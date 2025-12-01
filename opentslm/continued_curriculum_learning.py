#
# Continued Curriculum Learning for PsyTSLM
#


import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import json
import argparse
from typing import List, Optional, Dict, Any, Callable
from time_series_datasets.psychotherapy.psychotherapyCoTQADataset import PsychotherapyCoTQADataset
from time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from huggingface_hub import hf_hub_download, snapshot_download

from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from model.llm.OpenTSLMSP import OpenTSLMSP
from model.projector.MLPProjector import MLPProjector
import datetime
from logger import get_logger, set_global_verbose

from model_config import (
    BATCH_SIZE,
    EARLY_STOP_PAT,
    GRAD_CLIP_NORM,
    LR_ENCODER,
    LR_PROJECTOR,
    NUM_EPOCHS,
    PATCH_SIZE,
    WARMUP_FRAC,
    WEIGHT_DECAY,
)


class ContinuedCurriculumTrainer:
    """
    Continued curriculum learning trainer for OpenTSLM models.
    
    This trainer loads pre-trained models from HuggingFace Hub and continues
    training on stage6_synchrony_cot (psychotherapy).
    
    The key difference from CurriculumTrainer is that this trainer:
    1. Loads model weights from a HuggingFace Hub repository
    2. Only trains stage6_synchrony_cot
    3. Preserves all hyperparameters and training logic from curriculum_learning.py
    """

    def _sanitize_llm_id(self, llm_id: str) -> str:
        """Sanitize llm_id for use in directory names."""
        if not llm_id:
            return "unknown_llm"
        name = llm_id.split("/")[-1]
        name = name.replace(".", "_").replace("-", "_")
        while "__" in name:
            name = name.replace("__", "_")
        return name

    def __init__(
        self,
        model_type: str,
        hf_repo_id: str,
        hf_checkpoint_filename: str = "best_model.pt",
        device: str = None,
        gradient_checkpointing: bool = False,
        dist_url: str = "env://",
        dist_backend: str = "nccl",
        local_rank: int = int(os.environ.get("LOCAL_RANK", 0)),
        llm_id: str = None,
        cache_dir: str = None,
    ):
        """
        Initialize the continued curriculum trainer.

        Args:
            model_type: Either 'OpenTSLMSP' or 'OpenTSLMFlamingo'
            hf_repo_id: HuggingFace Hub repository ID (e.g., 'username/model-name')
            hf_checkpoint_filename: Checkpoint filename in the HF repo (default: 'best_model.pt')
            device: Device to use for training ('cuda', 'mps', or 'cpu')
            gradient_checkpointing: Enable gradient checkpointing
            dist_url: URL used to set up distributed training
            dist_backend: Distributed backend
            local_rank: Local GPU rank
            llm_id: LLM model ID (e.g., 'meta-llama/Llama-3.2-1B')
            cache_dir: Directory to cache HuggingFace downloads
        """
        self.model_type = model_type
        self.device = device or self._get_device()
        self.hf_repo_id = hf_repo_id
        self.hf_checkpoint_filename = hf_checkpoint_filename
        self.llm_id = llm_id
        self.llm_id_safe = self._sanitize_llm_id(llm_id)
        self.cache_dir = cache_dir

        if self.device == "mps":
            print(
                "🚨 Warning: Using MPS, might not be fully compatible with the model. Use CUDA for best results."
            )

        # Distributed training parameters
        self.gradient_checkpointing = gradient_checkpointing
        self.dist_url = dist_url
        self.dist_backend = dist_backend
        self.local_rank = local_rank

        # Initialize distributed training if needed
        self.rank = 0
        self.world_size = 1
        if self._should_use_distributed():
            self._init_distributed()

        # Initialize model first (with random weights)
        self.model = self._initialize_model()
        
        # Load pre-trained weights from HuggingFace Hub
        self._load_hf_checkpoint()
        
        self.results_dir = os.path.join("results", self.llm_id_safe, self.model_type)
        self._create_results_dir()

    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _initialize_model(self):
        """Initialize the specified model type."""
        if self.model_type == "OpenTSLMSP":
            model = OpenTSLMSP(llm_id=self.llm_id, device=self.device).to(self.device)
        elif self.model_type == "OpenTSLMFlamingo":
            model = OpenTSLMFlamingo(
                cross_attn_every_n_layers=1,
                gradient_checkpointing=self.gradient_checkpointing,
                llm_id=self.llm_id,
                device=self.device,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Use DDP for multi-GPU training
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            )
            if self.rank == 0:
                print(f"Wrapped {self.model_type} with DDP for distributed training")

        return model

    def _load_hf_checkpoint(self):
        """Load pre-trained checkpoint from HuggingFace Hub."""
        if self.rank == 0:
            print(f"\n📥 Loading checkpoint from HuggingFace Hub:")
            print(f"   Repository: {self.hf_repo_id}")
            print(f"   Filename: {self.hf_checkpoint_filename}")
        
        try:
            # Download checkpoint from HuggingFace Hub
            checkpoint_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=self.hf_checkpoint_filename,
                cache_dir=self.cache_dir,
            )
            
            if self.rank == 0:
                print(f"   Downloaded to: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # Get the underlying model (handles DDP wrapping)
            model = self._get_model()
            
            if self.model_type == "OpenTSLMSP":
                model.encoder.load_state_dict(checkpoint["encoder_state"])
                model.projector.load_state_dict(checkpoint["projector_state"])
                
                # Load LoRA state if present
                try:
                    loaded_count = model.load_lora_state_from_checkpoint(
                        checkpoint, allow_missing=True
                    )
                    if loaded_count > 0 and self.rank == 0:
                        print(f"   📥 Loaded LoRA adapters: {loaded_count} parameters")
                except RuntimeError as e:
                    if self.rank == 0:
                        print(f"   ⚠️  LoRA state not loaded: {e}")
            else:
                # Handle OpenTSLMFlamingo
                model_state = checkpoint["model_state"]
                if hasattr(self.model, "module"):
                    model_state = {f"module.{k}": v for k, v in model_state.items()}
                
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    model_state, strict=False
                )
                if missing_keys and self.rank == 0:
                    print(f"   ⚠️  Missing keys: {len(missing_keys)}")
                if unexpected_keys and self.rank == 0:
                    print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)}")
            
            if self.rank == 0:
                print(f"   ✅ Checkpoint loaded successfully!")
                if "epoch" in checkpoint:
                    print(f"   📊 Pre-trained at epoch: {checkpoint['epoch']}")
                if "val_loss" in checkpoint:
                    print(f"   📊 Pre-trained val loss: {checkpoint['val_loss']:.4f}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from HuggingFace Hub: {e}")

    def _create_results_dir(self):
        """Create the results directory structure."""
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create stage6 directories
        stage_dir = os.path.join(self.results_dir, "stage6_synchrony_cot")
        os.makedirs(stage_dir, exist_ok=True)
        os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(stage_dir, "results"), exist_ok=True)

    def _get_optimizer(
        self,
        batch_size: int = None,
        lr_encoder: float = None,
        lr_projector: float = None,
        lr_base: float = None,
    ):
        """Get optimizer for the model with configurable learning rates."""
        model = self._get_model()

        if self.model_type == "OpenTSLMSP":
            enc_params = list(model.encoder.parameters())
            proj_params = list(model.projector.projector.parameters())

            encoder_lr = lr_encoder if lr_encoder is not None else LR_ENCODER
            projector_lr = lr_projector if lr_projector is not None else LR_PROJECTOR

            param_groups = [
                {"params": enc_params, "lr": encoder_lr, "weight_decay": WEIGHT_DECAY},
                {"params": proj_params, "lr": projector_lr, "weight_decay": WEIGHT_DECAY},
            ]

            # Add LoRA parameters if enabled
            if hasattr(model, "lora_enabled") and model.lora_enabled:
                lora_params = model.get_lora_parameters()
                if lora_params:
                    param_groups.append({
                        "params": lora_params,
                        "lr": projector_lr,
                        "weight_decay": WEIGHT_DECAY,
                    })
                    if self.rank == 0:
                        print(f"📊 Learning rates for {self.model_type} (with LoRA):")
                        print(f"   Encoder LR: {encoder_lr:.2e}")
                        print(f"   Projector LR: {projector_lr:.2e}")
                        print(f"   LoRA LR: {projector_lr:.2e} ({len(lora_params)} parameters)")
            else:
                if self.rank == 0:
                    print(f"📊 Learning rates for {self.model_type}:")
                    print(f"   Encoder LR: {encoder_lr:.2e}")
                    print(f"   Projector LR: {projector_lr:.2e}")

            return AdamW(param_groups)
        else:
            # For Flamingo
            params_to_optimize = model.named_parameters()
            params_to_optimize = list(
                filter(
                    lambda x: x[1].requires_grad
                    and not getattr(x[1], "exclude_from_optimizer", False),
                    params_to_optimize,
                )
            )

            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                if "gated_cross_attn" in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)

            base_lr = lr_base if lr_base is not None else 2e-4

            if self.rank == 0:
                print(f"📊 Learning rate for {self.model_type}:")
                print(f"   Base LR: {base_lr:.2e}")

            return torch.optim.AdamW(
                [
                    {"params": params_with_wd, "weight_decay": 0.1},
                    {"params": params_without_wd, "weight_decay": 0.0},
                ],
                lr=base_lr,
            )

    def _merge_data_loaders(
        self,
        datasets: List[Dataset],
        shuffle: bool,
        batch_size: int,
        patch_size: int,
        distribute_data: bool = False,
    ) -> DataLoader:
        """Create a merged data loader from multiple datasets."""
        merged_ds = ConcatDataset(datasets)

        if distribute_data and dist.is_initialized():
            sampler = DistributedSampler(
                merged_ds, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle
            )
            return DataLoader(
                merged_ds,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=patch_size
                ),
            )
        else:
            return DataLoader(
                merged_ds,
                shuffle=shuffle,
                batch_size=batch_size,
                collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=patch_size
                ),
            )

    def _save_checkpoint(
        self, stage: str, epoch: int, val_loss: float, optimizer, scheduler
    ):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.results_dir, stage, "checkpoints")

        if dist.is_initialized() and self.rank != 0:
            return

        model = self._get_model()

        if self.model_type == "OpenTSLMSP":
            checkpoint = {
                "encoder_state": model.encoder.state_dict(),
                "projector_state": model.projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
                "hf_repo_id": self.hf_repo_id,  # Track source checkpoint
            }
            model.save_lora_state_to_checkpoint(checkpoint)
        else:
            model_state = model.state_dict()
            if hasattr(self.model, "module"):
                model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
            checkpoint = {
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
                "hf_repo_id": self.hf_repo_id,
            }

        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

        if self.rank == 0:
            import shutil
            total, used, free = shutil.disk_usage(checkpoint_dir)
            free_gb = free / (1024**3)
            print(f"💾 Disk space: {free_gb:.2f} GB free in {checkpoint_dir}")

        try:
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            if self.rank == 0:
                print(f"❌ Failed to save checkpoint: {e}")
                raise RuntimeError(f"Failed to save checkpoint: {e}")

    def _save_loss_history(
        self, stage: str, epoch: int, train_loss: float, val_loss: float
    ):
        """Save loss history to a file."""
        if dist.is_initialized() and self.rank != 0:
            return

        checkpoint_dir = os.path.join(self.results_dir, stage, "checkpoints")
        loss_history_file = os.path.join(checkpoint_dir, "loss_history.txt")

        os.makedirs(checkpoint_dir, exist_ok=True)

        if not os.path.exists(loss_history_file):
            with open(loss_history_file, "w") as f:
                f.write("Epoch\tTrain_Loss\tVal_Loss\n")
                f.write("-" * 30 + "\n")

        with open(loss_history_file, "a") as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")

    def _display_loss_history(self, stage: str):
        """Display the loss history for a stage if available."""
        if dist.is_initialized() and self.rank != 0:
            return

        checkpoint_dir = os.path.join(self.results_dir, stage, "checkpoints")
        loss_history_file = os.path.join(checkpoint_dir, "loss_history.txt")

        if os.path.exists(loss_history_file):
            try:
                with open(loss_history_file, "r") as f:
                    lines = f.readlines()

                if len(lines) > 2:
                    print(f"📊 Previous loss history for {stage}:")
                    print("   Epoch\tTrain_Loss\tVal_Loss")
                    print("   " + "-" * 30)

                    start_idx = max(2, len(lines) - 5)
                    for line in lines[start_idx:]:
                        if line.strip() and not line.startswith("-"):
                            parts = line.strip().split("\t")
                            if len(parts) == 3:
                                epoch, train_loss, val_loss = parts
                                print(f"   {epoch}\t{train_loss}\t{val_loss}")

                    if len(lines) > 7:
                        print(f"   ... and {len(lines) - 7} more epochs")
                    print()
            except Exception as e:
                print(f"⚠️  Could not read loss history: {e}")

    def _load_checkpoint(
        self, stage: str, optimizer, scheduler, eval_only: bool = False
    ):
        """Load model checkpoint for resuming training."""
        checkpoint_path = os.path.join(
            self.results_dir, stage, "checkpoints", "best_model.pt"
        )

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            model = self._get_model()

            if self.model_type == "OpenTSLMSP":
                model.encoder.load_state_dict(checkpoint["encoder_state"])
                model.projector.load_state_dict(checkpoint["projector_state"])

                try:
                    model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
                except RuntimeError as e:
                    if self.rank == 0:
                        print(f"❌ Failed to load LoRA state: {e}")
                    raise

                if not eval_only and optimizer is not None and "optimizer_state" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
            else:
                model_state = checkpoint["model_state"]
                if hasattr(self.model, "module"):
                    model_state = {f"module.{k}": v for k, v in model_state.items()}

                try:
                    self.model.load_state_dict(model_state, strict=False)
                except Exception as e:
                    raise RuntimeError(f"Failed to load model state: {e}")

                if not eval_only and optimizer is not None and "optimizer_state" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])

            if not eval_only and scheduler is not None and "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])

            return checkpoint.get("epoch", "?"), checkpoint.get("val_loss", float("inf"))
        return None, float("inf")

    def _evaluate_stage(
        self,
        stage: str,
        test_loader: DataLoader,
        stage_name: str,
        metric_func: Callable = None,
        epoch: int = None,
    ) -> Dict[str, Any]:
        """Evaluate model on test set."""
        self.model.eval()
        results = []
        max_new_tokens = 250

        results_file_rank = os.path.join(
            self.results_dir,
            stage_name,
            "results",
            f"test_predictions_rank_{self.rank if dist.is_initialized() else 0}.jsonl",
        )
        final_results_file = os.path.join(
            self.results_dir, stage_name, "results", "test_predictions.jsonl"
        )
        
        os.makedirs(os.path.dirname(results_file_rank), exist_ok=True)
        
        if self.rank == 0:
            print(f"[Eval] rank={self.rank}, world_size={self.world_size}")
            print(f"Saving per-rank test predictions to: {results_file_rank}")

        results_fp = open(results_file_rank, "w", encoding="utf-8")
        
        try:
            with torch.no_grad():
                for batch in tqdm(
                    test_loader, desc=f"Evaluating {stage_name}", disable=self.rank != 0
                ):
                    predictions = self._get_model().generate(
                        batch, max_new_tokens=max_new_tokens
                    )

                    for sample, pred in zip(batch, predictions):
                        result = {
                            "patient_id": sample.get("patient_id"),
                            "therapist_id": sample.get("therapist_id"),
                            "interview_type": sample.get("interview_type"),
                            "turn_index": sample.get("turn_index"),
                            "generated": pred,
                            "gold": sample["answer"]
                        }
                        results.append(result)
                        results_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                        results_fp.flush()
        finally:
            results_fp.close()

        if dist.is_initialized():
            dist.barrier()

        # Merge per-rank files
        if (not dist.is_initialized()) or (self.rank == 0):
            try:
                with open(final_results_file, "w", encoding="utf-8") as merged_fp:
                    num_ranks = self.world_size if dist.is_initialized() else 1
                    for r in range(num_ranks):
                        part_file = os.path.join(
                            self.results_dir,
                            stage_name,
                            "results",
                            f"test_predictions_rank_{r}.jsonl",
                        )
                        if os.path.exists(part_file):
                            with open(part_file, "r", encoding="utf-8") as pf:
                                for line in pf:
                                    merged_fp.write(line)
                if self.rank == 0:
                    print(f"Merged per-rank predictions into: {final_results_file}")
            finally:
                pass

        avg_test_loss = float("nan")

        metrics = {"test_loss": avg_test_loss}
        if epoch is not None:
            metrics["epoch"] = epoch
        if metric_func:
            if (not dist.is_initialized()) or (self.rank == 0):
                predictions = []
                gold_answers = []
                with open(final_results_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            predictions.append(obj.get("generated", ""))
                            gold_answers.append(obj.get("gold", ""))
                        except Exception:
                            continue
                additional_metrics = metric_func(predictions, gold_answers)
                metrics.update(additional_metrics)

        if (not dist.is_initialized()) or (self.rank == 0):
            metrics_file = os.path.join(
                self.results_dir, stage_name, "results", "metrics.json"
            )
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"✅ {stage_name} evaluation complete:")
            print(f"   Test predictions saved to: {final_results_file}")
            print(f"   Metrics saved to: {metrics_file}")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")

        if dist.is_initialized():
            dist.barrier()

        return metrics

    def _checkpoint_exists(self, stage: str) -> bool:
        """Check if a checkpoint exists for a specific stage."""
        checkpoint_path = os.path.join(
            self.results_dir, stage, "checkpoints", "best_model.pt"
        )
        return os.path.exists(checkpoint_path)

    def _enable_lora_if_needed(self, stage_name: str):
        """Enable LoRA for OpenTSLMSP models."""
        if self.model_type != "OpenTSLMSP":
            return

        model = self._get_model()

        # Enable LoRA for stage6_synchrony_cot
        if not getattr(model, "lora_enabled", False):
            if self.rank == 0:
                print(f"🔧 Enabling LoRA for {stage_name}")
            try:
                model.enable_lora(lora_r=16, lora_alpha=32, lora_dropout=0.0)
                if self.rank == 0:
                    print(f"✅ LoRA enabled for {stage_name}")
            except Exception as e:
                if self.rank == 0:
                    print(f"❌ Failed to enable LoRA: {e}")
                    print("   Continuing without LoRA...")
        else:
            if self.rank == 0:
                print(f"✅ LoRA already enabled for {stage_name}")

    def _get_model(self):
        """Get the underlying model (handles DDP wrapping)."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _should_use_distributed(self) -> bool:
        """Check if distributed training should be used."""
        return ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1) or (
            "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) >= 0
        )

    def _init_distributed(self):
        """Initialize distributed training."""
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        elif "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(
            backend=self.dist_backend,
            init_method=self.dist_url,
            world_size=self.world_size,
            rank=self.rank,
            timeout=datetime.timedelta(hours=999),
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)

        if self.rank == 0:
            print(f"Initialized distributed training with {self.world_size} GPUs")

    def stage6_synchrony_cot(
        self,
        batch_size: int = None,
        eval_only: bool = False,
        num_epochs: int = 30,
        lr_encoder: float = 2e-4,
        lr_projector: float = 1e-4,
        lr_base: float = 2e-4,
        feature_columns: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Stage 6: Chain-of-Thought Reasoning (Psychotherapy Synchrony).

        Configuration:
        - Epochs: 30 (default, configurable)
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only (chain-of-thought reasoning)
        - Features: Configurable AU columns (default: AU12_r, AU06_r, AU04_r, AU15_r)
        
        Args:
            batch_size: Batch size for training
            eval_only: Skip training, only run evaluation
            num_epochs: Number of training epochs
            lr_encoder: Learning rate for encoder (OpenTSLMSP)
            lr_projector: Learning rate for projector (OpenTSLMSP)
            lr_base: Base learning rate (OpenTSLMFlamingo)
            feature_columns: List of AU columns to use as features
        """
        stage_name = "stage6_synchrony_cot"
        
        if batch_size is None:
            batch_size = BATCH_SIZE
            
        # feature_columns=None means use ALL AU*_r columns from OpenFace
        # The loader will automatically extract all available AU regression columns

        if self.rank == 0:
            print(f"\n🚀 Starting {stage_name} Training with {self.model_type}")
            print(f"   Pre-trained from: {self.hf_repo_id}")
            if eval_only:
                print("🔍 EVAL-ONLY MODE: Skipping training, only running evaluation")
            print("=" * 60)
            print(f"📊 Stage Configuration:")
            print(f"   Epochs: {num_epochs}")
            if self.model_type == "OpenTSLMSP":
                print(f"   Encoder LR: {lr_encoder:.2e}")
                print(f"   Projector LR: {lr_projector:.2e}")
            else:
                print(f"   Base LR: {lr_base:.2e}")
            print(f"   Batch size per GPU: {batch_size}")
            print(f"   Feature columns: {feature_columns}")
            if self.world_size > 1:
                print(f"   Effective batch size: {batch_size * self.world_size}")
            print()

        if eval_only and not self._checkpoint_exists(stage_name):
            raise RuntimeError(
                f"Eval-only mode requires a checkpoint for {stage_name}, but none found."
            )

        # Enable LoRA if needed
        self._enable_lora_if_needed(stage_name)

        # Initialize optimizer
        optimizer = self._get_optimizer(batch_size, lr_encoder, lr_projector, lr_base)

        # Dataset kwargs
        dataset_kwargs = {'feature_columns': feature_columns}

        # Create data loaders
        train_loader = self._merge_data_loaders(
            [PsychotherapyCoTQADataset("train", EOS_TOKEN=self._get_model().get_eos_token(), **dataset_kwargs)],
            shuffle=True,
            batch_size=batch_size,
            patch_size=PATCH_SIZE,
            distribute_data=self.world_size > 1,
        )

        val_loader = self._merge_data_loaders(
            [PsychotherapyCoTQADataset("validation", EOS_TOKEN=self._get_model().get_eos_token(), **dataset_kwargs)],
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=False,
        )

        test_loader = self._merge_data_loaders(
            [PsychotherapyCoTQADataset("test", EOS_TOKEN=self._get_model().get_eos_token(), **dataset_kwargs)],
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=self.world_size > 1,
        )

        # Scheduler
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(WARMUP_FRAC * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        if self.rank == 0:
            print(f"📈 Total training steps: {total_steps}")
            print(f"🔥 Warmup steps: {warmup_steps}")

        # Load previous checkpoint if exists (for resuming)
        best_epoch, best_val_loss = self._load_checkpoint(
            stage_name, optimizer, scheduler, eval_only=eval_only
        )
        if best_epoch is not None:
            print(f"📂 Resuming {stage_name} from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
            self._display_loss_history(stage_name)
        else:
            print(f"🆕 Starting fresh training for {stage_name}")
            best_val_loss = float("inf")

        # Skip training loop if eval_only
        if eval_only:
            if self.rank == 0:
                print(f"⏭️  Skipping training loop (eval_only mode)")
            epoch = best_epoch
            epochs_no_improve = 0
        else:
            # Training loop
            epochs_no_improve = 0
            start_epoch = best_epoch + 1 if best_epoch is not None else 1
            
            for epoch in range(start_epoch, num_epochs + 1):
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

                # Training
                self.model.train()
                running_loss = 0.0
                prog = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch}/{num_epochs}",
                    disable=self.rank != 0,
                )
                
                for i, batch in enumerate(prog):
                    if epoch == start_epoch and i == 0:
                        print(f"[DEBUG] Batch {i} - batch size: {len(batch)}")
                        if isinstance(batch, list) and isinstance(batch[0], dict):
                            for k, v in batch[0].items():
                                if hasattr(v, "shape"):
                                    print(f"[DEBUG] Sample key '{k}' shape: {v.shape}")
                        print(
                            torch.cuda.memory_summary()
                            if torch.cuda.is_available()
                            else "No CUDA"
                        )
                    
                    optimizer.zero_grad()
                    loss = self._get_model().compute_loss(batch)
                    loss.backward()

                    clip_grad_norm_(self._get_model().parameters(), GRAD_CLIP_NORM)

                    optimizer.step()
                    scheduler.step()

                    running_loss += loss.item()

                    if i % 100 == 0:
                        torch.cuda.empty_cache()

                    if self.rank == 0:
                        prog.set_postfix(
                            loss=f"{loss.item():.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        )

                avg_train_loss = running_loss / len(train_loader)
                if self.rank == 0:
                    tqdm.write(f"Epoch {epoch} — train loss: {avg_train_loss:.4f}")

                # Validation
                val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm(
                        val_loader,
                        desc=f"Validating {stage_name}",
                        disable=self.rank != 0,
                    ):
                        val_loss += self._get_model().compute_loss(batch).item()

                avg_val_loss = val_loss / len(val_loader)

                if dist.is_initialized():
                    val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    avg_val_loss = val_loss_tensor.item() / self.world_size

                if self.rank == 0:
                    tqdm.write(f"Epoch {epoch} — val   loss: {avg_val_loss:.4f}")
                    tqdm.write(f"Epoch {epoch} — best  loss: {best_val_loss:.4f}")

                self._save_loss_history(stage_name, epoch, avg_train_loss, avg_val_loss)

                # Early stopping
                should_save = avg_val_loss + 1e-4 < best_val_loss
                if dist.is_initialized():
                    save_tensor = torch.tensor(1 if should_save else 0, device=self.device)
                    dist.all_reduce(save_tensor, op=dist.ReduceOp.SUM)
                    should_save = save_tensor.item() > 0

                if should_save:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    self._save_checkpoint(stage_name, epoch, avg_val_loss, optimizer, scheduler)
                    if self.rank == 0:
                        tqdm.write("✔️  New best model saved.\n")
                else:
                    epochs_no_improve += 1
                    if self.rank == 0:
                        tqdm.write(f"No improvement for {epochs_no_improve}/{EARLY_STOP_PAT} epochs.\n")

                    if epochs_no_improve >= EARLY_STOP_PAT:
                        if self.rank == 0:
                            tqdm.write(f"\nEarly stopping triggered after {epoch} epochs.")
                        break

                if dist.is_initialized():
                    best_loss_tensor = torch.tensor(best_val_loss, device=self.device)
                    epochs_tensor = torch.tensor(epochs_no_improve, device=self.device)
                    dist.broadcast(best_loss_tensor, src=0)
                    dist.broadcast(epochs_tensor, src=0)
                    best_val_loss = best_loss_tensor.item()
                    epochs_no_improve = int(epochs_tensor.item())

        # Load best model and evaluate
        best_epoch, _ = self._load_checkpoint(stage_name, optimizer, scheduler)
        if best_epoch is not None:
            if self.rank == 0:
                print(f"📂 Loaded best checkpoint from epoch {best_epoch} for evaluation.")

        if self.rank == 0:
            print(f"🏁 Training completed for {stage_name}")

        metrics = self._evaluate_stage(stage_name, test_loader, stage_name, None, best_epoch)

        # Mark as completed
        metrics["completed"] = True
        metrics["hf_repo_id"] = self.hf_repo_id
        metrics_file = os.path.join(self.results_dir, stage_name, "results", "metrics.json")
        if self.rank == 0:
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Continued Curriculum Learning for OpenTSLM Models - Stage 6 Synchrony CoT"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["OpenTSLMSP", "OpenTSLMFlamingo"],
        required=True,
        help="Model type to train",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        required=True,
        help="HuggingFace Hub repository ID containing the pre-trained model (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--hf_checkpoint_filename",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename in the HuggingFace repo (default: 'best_model.pt')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training (default: use value from model_config.py)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--lr_encoder",
        type=float,
        default=2e-4,
        help="Learning rate for encoder (OpenTSLMSP, default: 2e-4)",
    )
    parser.add_argument(
        "--lr_projector",
        type=float,
        default=1e-4,
        help="Learning rate for projector (OpenTSLMSP, default: 1e-4)",
    )
    parser.add_argument(
        "--lr_base",
        type=float,
        default=2e-4,
        help="Base learning rate (OpenTSLMFlamingo, default: 2e-4)",
    )
    parser.add_argument(
        "--feature_columns",
        nargs="+",
        default=None,
        help="AU feature columns to use (default: None = all AU*_r columns from OpenFace)",
    )
    parser.add_argument(
        "--eval_only",
        default=False,
        action="store_true",
        help="Skip training and only run evaluation (requires existing checkpoint)",
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="LLM model ID (e.g., 'meta-llama/Llama-3.2-1B')",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache HuggingFace downloads",
    )

    # Distributed training arguments
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="URL used to set up distributed training",
    )
    parser.add_argument(
        "--dist_backend",
        default="nccl",
        type=str,
        help="Distributed backend",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local GPU rank",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    set_global_verbose(args.verbose)
    logger = get_logger(verbose=args.verbose)

    # Initialize trainer
    trainer = ContinuedCurriculumTrainer(
        model_type=args.model,
        hf_repo_id=args.hf_repo_id,
        hf_checkpoint_filename=args.hf_checkpoint_filename,
        device=args.device,
        gradient_checkpointing=args.gradient_checkpointing,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend,
        local_rank=args.local_rank,
        llm_id=args.llm_id,
        cache_dir=args.cache_dir,
    )

    # Run stage6_synchrony_cot
    results = trainer.stage6_synchrony_cot(
        batch_size=args.batch_size,
        eval_only=args.eval_only,
        num_epochs=args.num_epochs,
        lr_encoder=args.lr_encoder,
        lr_projector=args.lr_projector,
        lr_base=args.lr_base,
        feature_columns=args.feature_columns,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("Stage 6 Synchrony CoT Results:")
    logger.info("=" * 60)
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
