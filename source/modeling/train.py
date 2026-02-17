"""
Training script for the Hierarchical Multimodal Attention Network.

Usage
-----
::

    python -m source.modeling.train \\
        --data_model  data_model.yaml \\
        --config      config.yaml \\
        --au_descriptions_dir  /path/to/au_descriptions/ \\
        --output_dir  source/modeling/checkpoints \\
        --epochs 30 \\
        --batch_size 2 \\
        --lr 2e-5

The script will:
1. Build train / val / test data loaders using the therapist-based splits
   from ``config.yaml``.
2. Train the model with MSE loss (NaN-masked for missing targets).
3. Log per-epoch metrics (MSE, MAE, R²) for train and validation.
4. Save the best checkpoint (lowest val MSE) and a final checkpoint.
5. Evaluate on the test set after training.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer

# Allow running as ``python -m source.modeling.train`` from project root
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from source.modeling.model import HierarchicalMultimodalAttentionNetwork
from source.modeling.dataset import create_dataloaders


def setup_logger(level: str = "INFO") -> logging.Logger:
    """Configure process logger to emit to stderr (captured by SLURM .err)."""
    logger = logging.getLogger("hman_train")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Masked MSE loss
# ---------------------------------------------------------------------------

def masked_mse_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """MSE loss that ignores ``NaN`` targets.

    Parameters
    ----------
    preds   : (batch, num_targets)
    targets : (batch, num_targets)  – may contain ``NaN``

    Returns
    -------
    Scalar loss (mean over valid elements).  Returns ``0`` when no valid
    targets exist (should not happen in practice).
    """
    valid = ~torch.isnan(targets)
    if not valid.any():
        return preds.new_tensor(0.0)
    return nn.functional.mse_loss(preds[valid], targets[valid])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
) -> dict[str, float]:
    """Compute MSE, MAE, and R² over non-NaN entries.

    Parameters
    ----------
    all_preds, all_targets : (N, num_targets)
    """
    valid = ~np.isnan(all_targets)
    if not valid.any():
        return {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    p = all_preds[valid]
    t = all_targets[valid]
    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    ss_res = np.sum((p - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Train / evaluate one epoch
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    log_interval: int = 20,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []
    optimizer.zero_grad()

    grad_norm_value = float("nan")
    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)

        preds, _ = model(
            batch["speech_input_ids"],
            batch["speech_attention_mask"],
            batch["au_input_ids"],
            batch["au_attention_mask"],
            batch["turn_counts"],
            batch["turn_mask"],
        )

        loss = masked_mse_loss(preds, batch["targets"]) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norm_value = float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(batch["targets"].detach().cpu().numpy())

        if ((step + 1) % max(log_interval, 1) == 0) or ((step + 1) == len(loader)):
            gpu_mem_gb = 0.0
            if device.type == "cuda":
                gpu_mem_gb = torch.cuda.memory_allocated(device) / 1024**3
            running_loss = total_loss / max(n_batches, 1)
            logger.info(
                "Epoch %d/%d | Train step %d/%d | loss=%.6f | running_loss=%.6f | grad_norm=%.4f | lr=%.3e | gpu_mem=%.2fGiB",
                epoch,
                total_epochs,
                step + 1,
                len(loader),
                float(loss.item() * grad_accum_steps),
                float(running_loss),
                grad_norm_value,
                optimizer.param_groups[0]["lr"],
                gpu_mem_gb,
            )

    all_preds_np = np.concatenate(all_preds, axis=0)
    all_targets_np = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(all_preds_np, all_targets_np)
    metrics["avg_loss"] = total_loss / max(n_batches, 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    logger: logging.Logger,
    stage: str = "val",
    log_interval: int = 20,
) -> tuple[dict[str, float], list[dict]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []
    all_attention: list[dict] = []

    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)
        preds, attn_w = model(
            batch["speech_input_ids"],
            batch["speech_attention_mask"],
            batch["au_input_ids"],
            batch["au_attention_mask"],
            batch["turn_counts"],
            batch["turn_mask"],
        )
        loss = masked_mse_loss(preds, batch["targets"])
        total_loss += loss.item()
        n_batches += 1
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch["targets"].cpu().numpy())

        # Collect attention for interpretability
        for i, meta in enumerate(batch["metadata"]):
            n = batch["turn_counts"][i].item()
            all_attention.append({
                **meta,
                "attention_weights": attn_w[i, :n].cpu().tolist(),
                "prediction": preds[i].cpu().tolist(),
                "target": batch["targets"][i].cpu().tolist(),
            })

        if ((step + 1) % max(log_interval, 1) == 0) or ((step + 1) == len(loader)):
            gpu_mem_gb = 0.0
            if device.type == "cuda":
                gpu_mem_gb = torch.cuda.memory_allocated(device) / 1024**3
            logger.info(
                "%s step %d/%d | batch_loss=%.6f | running_loss=%.6f | gpu_mem=%.2fGiB",
                stage.upper(),
                step + 1,
                len(loader),
                float(loss.item()),
                float(total_loss / max(n_batches, 1)),
                gpu_mem_gb,
            )

    all_preds_np = np.concatenate(all_preds, axis=0)
    all_targets_np = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(all_preds_np, all_targets_np)
    metrics["avg_loss"] = total_loss / max(n_batches, 1)
    return metrics, all_attention


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Hierarchical Multimodal Attention Network for BLRI prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data_model", type=Path, required=True, help="Path to data_model.yaml")
    p.add_argument("--config", type=Path, required=True, help="Path to config.yaml (split definitions)")
    p.add_argument("--au_descriptions_dir", type=Path, required=True, help="Directory with AU-description JSONs")

    # Output
    p.add_argument("--output_dir", type=Path, default=Path("source/modeling/checkpoints"), help="Where to save model checkpoints and logs")

    # Model
    p.add_argument("--bert_model", type=str, default="distilbert-base-uncased", help="HuggingFace BERT variant for turn encoding")
    p.add_argument("--fusion_dim", type=int, default=256, help="Turn-level fusion dimensionality")
    p.add_argument("--gru_hidden", type=int, default=128, help="GRU hidden size per direction")
    p.add_argument("--gru_layers", type=int, default=1, help="Number of stacked GRU layers")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    p.add_argument("--bert_sub_batch", type=int, default=64, help="Max turns per BERT forward pass (GPU memory control)")

    # Training
    p.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=2, help="Batch size (sessions per batch)")
    p.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    p.add_argument("--warmup_epochs", type=int, default=3, help="Linear warmup epochs")
    p.add_argument("--freeze_bert_epochs", type=int, default=2, help="Freeze BERT backbone for the first N epochs")
    p.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    p.add_argument("--max_token_length", type=int, default=128, help="Max token sequence length for BERT")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (0 = disabled)")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--log_level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING")
    p.add_argument("--log_interval", type=int, default=20, help="Batch logging interval for train/val/test")

    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info("Starting HMAN training")
    logger.info("Arguments: %s", json.dumps({k: str(v) for k, v in vars(args).items()}, indent=None))
    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("PyTorch: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("CUDA device count: %d", torch.cuda.device_count() if torch.cuda.is_available() else 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU[0]: %s", torch.cuda.get_device_name(0))
        logger.info("GPU[0] Memory: %.1f GiB", props.total_memory / 1024**3)

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", args.output_dir)

    # Tokeniser
    logger.info("Loading tokenizer: %s", args.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    logger.info("Tokenizer loaded")

    # Dataloaders
    logger.info("Building datasets and dataloaders")
    loaders = create_dataloaders(
        data_model_path=args.data_model,
        au_descriptions_dir=args.au_descriptions_dir,
        config_path=args.config,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_token_length=args.max_token_length,
        num_workers=args.num_workers,
    )
    logger.info(
        "DataLoader sizes | train: %d batches (%d sessions) | val: %d batches (%d sessions) | test: %d batches (%d sessions)",
        len(loaders["train"]), len(loaders["train"].dataset),
        len(loaders["val"]), len(loaders["val"].dataset),
        len(loaders["test"]), len(loaders["test"].dataset),
    )

    if len(loaders["train"].dataset) == 0:
        raise RuntimeError("Train split is empty after filtering. Check transcript/AU availability and split config.")
    if len(loaders["val"].dataset) == 0:
        logger.warning("Validation split is empty; validation metrics/checkpointing may fail.")
    if len(loaders["test"].dataset) == 0:
        logger.warning("Test split is empty; test evaluation will fail.")

    # Model
    logger.info("Initialising model (backbone=%s)", args.bert_model)
    model = HierarchicalMultimodalAttentionNetwork(
        bert_model_name=args.bert_model,
        fusion_dim=args.fusion_dim,
        gru_hidden_dim=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        num_targets=2,
        bert_sub_batch=args.bert_sub_batch,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s (trainable: %s)", f"{n_params:,}", f"{n_train:,}")

    # Optimiser and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    # Training loop
    logger.info("%s", "=" * 70)
    logger.info("TRAINING")
    logger.info("%s", "=" * 70)

    best_val_mse = float("inf")
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Freeze / unfreeze BERT
        if epoch <= args.freeze_bert_epochs:
            model.freeze_bert()
            logger.info("Epoch %d: BERT backbone frozen", epoch)
        elif epoch == args.freeze_bert_epochs + 1:
            model.unfreeze_bert()
            logger.info("Epoch %d: BERT backbone unfrozen", epoch)

        # Train
        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, device,
            logger=logger,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
        )

        # Validate
        val_metrics, _ = evaluate(
            model,
            loaders["val"],
            device,
            logger=logger,
            stage="val",
            log_interval=args.log_interval,
        )

        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        # Log
        log = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_metrics["avg_loss"],
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "val_loss": val_metrics["avg_loss"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "elapsed_s": elapsed,
        }
        history.append(log)

        logger.info(
            "Epoch %d/%d done | lr=%.3e | train(mse=%.4f mae=%.4f r2=%+.4f) | val(mse=%.4f mae=%.4f r2=%+.4f) | %.1fs",
            epoch,
            args.epochs,
            lr_now,
            train_metrics["mse"],
            train_metrics["mae"],
            train_metrics["r2"],
            val_metrics["mse"],
            val_metrics["mae"],
            val_metrics["r2"],
            elapsed,
        )

        # Checkpointing
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            patience_counter = 0
            ckpt_path = args.output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": best_val_mse,
                "args": vars(args),
            }, ckpt_path)
            logger.info("New best model saved: %s (val_mse=%.6f)", ckpt_path, best_val_mse)
        else:
            patience_counter += 1
            logger.info("No val improvement. patience_counter=%d/%d", patience_counter, args.patience)

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            logger.info("Early stopping triggered (patience=%d)", args.patience)
            break

    # Save final checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_mse": val_metrics["mse"],
        "args": vars(args),
    }, args.output_dir / "final_model.pt")

    # Save training history
    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info("Saved training history: %s", args.output_dir / "training_history.json")

    # --------------- Test evaluation ----------------------------------------
    logger.info("%s", "=" * 70)
    logger.info("TEST EVALUATION")
    logger.info("%s", "=" * 70)

    # Load best model
    best_ckpt = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    logger.info("Loaded best model from epoch %s (val_mse=%.6f)", best_ckpt["epoch"], best_ckpt["val_mse"])

    test_metrics, test_attention = evaluate(
        model,
        loaders["test"],
        device,
        logger=logger,
        stage="test",
        log_interval=args.log_interval,
    )

    logger.info(
        "Test metrics | mse=%.4f | mae=%.4f | r2=%+.4f",
        test_metrics["mse"],
        test_metrics["mae"],
        test_metrics["r2"],
    )

    # Save test predictions and attention weights for interpretability
    with open(args.output_dir / "test_predictions.json", "w") as f:
        json.dump(test_attention, f, indent=2, default=str)
    logger.info("Saved test predictions/attention: %s", args.output_dir / "test_predictions.json")

    logger.info("Done")


if __name__ == "__main__":
    main()
