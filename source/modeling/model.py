"""
Hierarchical Multimodal Attention Network for psychotherapy BLRI prediction.

Components
----------
1. **TurnEncoder**      – Shared DistilBERT backbone encodes speech summaries and
                          AU descriptions separately, then fuses them via a learned gate.
2. **SessionEncoder**   – Bidirectional GRU aggregates the sequence of fused turn vectors.
3. **TurnAttention**    – Additive (Bahdanau-style) attention computes per-turn weights
                          so that predictions are interpretable.
4. **Output Head**      – Linear regression head predicting BLRI scores
                          (patient-perspective ``Pr`` and interviewer-perspective ``In``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


# ---------------------------------------------------------------------------
# Gated Fusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """Element-wise gated fusion of two modality vectors.

    ``gate = σ(W [x₁ ‖ x₂] + b)``
    ``fused = gate ⊙ x₁ + (1 − gate) ⊙ x₂``
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim * 2, dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(torch.cat([x1, x2], dim=-1)))
        return gate * x1 + (1.0 - gate) * x2


# ---------------------------------------------------------------------------
# Turn Encoder
# ---------------------------------------------------------------------------

class TurnEncoder(nn.Module):
    """Encodes a (speech-summary, AU-description) pair into a single vector.

    A **shared** transformer backbone produces CLS embeddings for each
    modality.  Separate linear projections map them to ``fusion_dim``,
    followed by :class:`GatedFusion`.

    Parameters
    ----------
    bert_model_name : str
        HuggingFace model identifier (default ``distilbert-base-uncased``).
    fusion_dim : int
        Dimensionality of fused turn embeddings.
    dropout : float
        Dropout probability after projection.
    """

    def __init__(
        self,
        bert_model_name: str = "distilbert-base-uncased",
        fusion_dim: int = 256,
        dropout: float = 0.1,
        n_au_numeric: int | None = None,
    ):
        """
        Parameters
        ----------
        n_au_numeric : int | None
            When set, bypasses BERT for the AU stream and instead projects a
            float vector of this dimensionality directly into ``fusion_dim``.
            Use this for the AU-numbers ablation so numeric values are not
            corrupted by subword tokenisation.
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size

        self.speech_proj = nn.Sequential(
            nn.Linear(bert_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if n_au_numeric is not None:
            # Numeric AU path: float vector → projection (no BERT needed for AU)
            self.au_proj = nn.Sequential(
                nn.Linear(n_au_numeric, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self._au_numeric = True
        else:
            # Text AU path: BERT CLS → projection
            self.au_proj = nn.Sequential(
                nn.Linear(bert_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self._au_numeric = False

        self.fusion = GatedFusion(fusion_dim)

    # -- helpers -------------------------------------------------------------

    def _cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the [CLS] embedding from the shared backbone."""
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        speech_input_ids: torch.Tensor,
        speech_attention_mask: torch.Tensor,
        au_input_ids: torch.Tensor | None = None,
        au_attention_mask: torch.Tensor | None = None,
        au_numeric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        speech_input_ids, speech_attention_mask : (total_turns, seq_len)
        au_input_ids, au_attention_mask         : (total_turns, seq_len) – AU-text path
        au_numeric                              : (total_turns, n_au_numeric) – AU-numbers path

        Returns
        -------
        fused : (total_turns, fusion_dim)
        """
        speech_cls = self._cls_embedding(speech_input_ids, speech_attention_mask)
        speech_h = self.speech_proj(speech_cls)

        if self._au_numeric:
            if au_numeric is None:
                raise ValueError("au_numeric must be provided when n_au_numeric was set at construction.")
            au_h = self.au_proj(au_numeric)
        else:
            if au_input_ids is None or au_attention_mask is None:
                raise ValueError("au_input_ids/au_attention_mask must be provided for AU-text path.")
            au_cls = self._cls_embedding(au_input_ids, au_attention_mask)
            au_h = self.au_proj(au_cls)

        return self.fusion(speech_h, au_h)


# ---------------------------------------------------------------------------
# Session Encoder
# ---------------------------------------------------------------------------

class SessionEncoder(nn.Module):
    """Bidirectional GRU over the sequence of turn vectors.

    Uses :func:`pack_padded_sequence` so padded turns are never processed.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each turn vector (= ``fusion_dim``).
    hidden_dim : int
        GRU hidden size **per direction** (output is ``hidden_dim * 2``).
    num_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout between GRU layers (only used if ``num_layers > 1``).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        turn_embeddings: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        turn_embeddings : (batch, max_turns, input_dim)
        lengths         : (batch,)  – actual number of turns per session

        Returns
        -------
        outputs : (batch, max_turns, hidden_dim * 2)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            turn_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        outputs, _ = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs


# ---------------------------------------------------------------------------
# Turn Attention
# ---------------------------------------------------------------------------

class TurnAttention(nn.Module):
    """Additive (Bahdanau-style) attention over turn representations.

    Produces a session-level context vector **and** per-turn attention weights
    that can be inspected for interpretability.

    ``score_i = v^T tanh(W h_i + b)``
    ``α = softmax(score, masked)``
    ``context = Σ α_i h_i``

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the turn representations (BiGRU output).
    attention_dim : int | None
        Internal projection size.  ``None`` → same as *hidden_dim*.
    """

    def __init__(self, hidden_dim: int, attention_dim: int | None = None):
        super().__init__()
        attention_dim = attention_dim or hidden_dim
        self.proj = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        turn_repr: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        turn_repr : (batch, max_turns, hidden_dim)
        mask      : (batch, max_turns)  – ``True`` for valid turns

        Returns
        -------
        context : (batch, hidden_dim)
        weights : (batch, max_turns)  – attention distribution
        """
        scores = self.v(torch.tanh(self.proj(turn_repr))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        # Guard against all-masked rows (single-turn edge case)
        weights = weights.masked_fill(~mask, 0.0)
        context = (weights.unsqueeze(-1) * turn_repr).sum(dim=1)
        return context, weights


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class HierarchicalMultimodalAttentionNetwork(nn.Module):
    """End-to-end hierarchical model.

    Turn Encoder → Session Encoder → Attention → BLRI Regression Head.

    Parameters
    ----------
    bert_model_name : str
        HuggingFace identifier for the shared text backbone.
    fusion_dim : int
        Turn-level embedding size after gated fusion.
    gru_hidden_dim : int
        GRU hidden size per direction.
    gru_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout probability used throughout.
    num_targets : int
        Number of regression targets (default 2 → BLRI_Pr, BLRI_In).
    bert_sub_batch : int
        Maximum number of turns to process in a single BERT forward pass to
        keep GPU memory bounded.  ``0`` → process all at once.
    """

    def __init__(
        self,
        bert_model_name: str = "distilbert-base-uncased",
        fusion_dim: int = 256,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.1,
        num_targets: int = 2,
        bert_sub_batch: int = 64,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.bert_sub_batch = bert_sub_batch

        self.turn_encoder = TurnEncoder(bert_model_name, fusion_dim, dropout)
        self.session_encoder = SessionEncoder(
            fusion_dim, gru_hidden_dim, num_layers=gru_layers, dropout=dropout,
        )
        bigru_out = gru_hidden_dim * 2
        self.attention = TurnAttention(bigru_out)
        self.output_head = nn.Sequential(
            nn.LayerNorm(bigru_out),
            nn.Linear(bigru_out, gru_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, num_targets),
        )

    # -- internal BERT batching ----------------------------------------------

    def _encode_turns_batched(
        self,
        speech_ids: torch.Tensor,
        speech_mask: torch.Tensor,
        au_ids: torch.Tensor,
        au_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode all turns, optionally in sub-batches to save memory."""
        total = speech_ids.size(0)
        if self.bert_sub_batch <= 0 or total <= self.bert_sub_batch:
            return self.turn_encoder(speech_ids, speech_mask, au_input_ids=au_ids, au_attention_mask=au_mask)

        parts: list[torch.Tensor] = []
        for start in range(0, total, self.bert_sub_batch):
            end = min(start + self.bert_sub_batch, total)
            parts.append(
                self.turn_encoder(
                    speech_ids[start:end],
                    speech_mask[start:end],
                    au_input_ids=au_ids[start:end],
                    au_attention_mask=au_mask[start:end],
                )
            )
        return torch.cat(parts, dim=0)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        speech_input_ids: torch.Tensor,
        speech_attention_mask: torch.Tensor,
        au_input_ids: torch.Tensor,
        au_attention_mask: torch.Tensor,
        turn_counts: torch.Tensor,
        turn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        speech_input_ids       : (total_turns, max_tok)
        speech_attention_mask  : (total_turns, max_tok)
        au_input_ids           : (total_turns, max_tok)
        au_attention_mask      : (total_turns, max_tok)
        turn_counts            : (batch,)  – number of turns per session
        turn_mask              : (batch, max_turns)  – boolean mask

        Returns
        -------
        predictions       : (batch, num_targets)
        attention_weights  : (batch, max_turns)
        """
        # 1 ── Encode every turn across the batch
        turn_vecs = self._encode_turns_batched(
            speech_input_ids, speech_attention_mask,
            au_input_ids, au_attention_mask,
        )  # (total_turns, fusion_dim)

        # 2 ── Scatter back into (batch, max_turns, fusion_dim)
        batch_size = turn_counts.size(0)
        max_turns = turn_mask.size(1)
        turn_emb = turn_vecs.new_zeros(batch_size, max_turns, self.fusion_dim)

        idx = 0
        for i in range(batch_size):
            n = turn_counts[i].item()
            turn_emb[i, :n] = turn_vecs[idx : idx + n]
            idx += n

        # 3 ── Session encoder (BiGRU)
        session_repr = self.session_encoder(turn_emb, turn_counts)

        # 4 ── Turn-level attention
        context, attn_weights = self.attention(session_repr, turn_mask)

        # 5 ── Prediction
        preds = self.output_head(context)

        return preds, attn_weights

    # -- convenience ---------------------------------------------------------

    def freeze_bert(self) -> None:
        """Freeze the shared BERT backbone (useful for warm-up epochs)."""
        for p in self.turn_encoder.bert.parameters():
            p.requires_grad = False

    def unfreeze_bert(self) -> None:
        """Unfreeze the shared BERT backbone."""
        for p in self.turn_encoder.bert.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Single-task HMAN models (Regressor / Classifier)
# ---------------------------------------------------------------------------

class HMANBase(nn.Module):
    """Shared encoder stack used by HMANRegressor and HMANClassifier.

    Parameters
    ----------
    n_au_numeric : int | None
        When set, the AU stream uses a numeric float projection instead of BERT.
        Pass ``N_AUS * 2`` (patient + therapist concatenated) for AU-numbers ablation.
    """

    def __init__(
        self,
        bert_model_name: str,
        fusion_dim: int,
        gru_hidden_dim: int,
        gru_layers: int,
        dropout: float,
        bert_sub_batch: int,
        n_au_numeric: int | None = None,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.bert_sub_batch = bert_sub_batch
        self.n_au_numeric = n_au_numeric

        self.turn_encoder = TurnEncoder(
            bert_model_name, fusion_dim, dropout, n_au_numeric=n_au_numeric
        )
        self.session_encoder = SessionEncoder(
            fusion_dim, gru_hidden_dim, num_layers=gru_layers, dropout=dropout
        )

        bigru_out = gru_hidden_dim * 2
        self.attention = TurnAttention(bigru_out)
        self.shared = nn.Sequential(
            nn.LayerNorm(bigru_out),
            nn.Linear(bigru_out, gru_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._gru_hidden_dim = gru_hidden_dim

    def _encode_turns_batched(
        self,
        speech_ids: torch.Tensor,
        speech_mask: torch.Tensor,
        au_input_ids: torch.Tensor | None,
        au_attention_mask: torch.Tensor | None,
        au_numeric: torch.Tensor | None,
    ) -> torch.Tensor:
        total = speech_ids.size(0)

        def _encode_slice(sl: slice) -> torch.Tensor:
            return self.turn_encoder(
                speech_ids[sl],
                speech_mask[sl],
                au_input_ids=au_input_ids[sl] if au_input_ids is not None else None,
                au_attention_mask=au_attention_mask[sl] if au_attention_mask is not None else None,
                au_numeric=au_numeric[sl] if au_numeric is not None else None,
            )

        if self.bert_sub_batch <= 0 or total <= self.bert_sub_batch:
            return _encode_slice(slice(None))

        parts: list[torch.Tensor] = []
        for start in range(0, total, self.bert_sub_batch):
            parts.append(_encode_slice(slice(start, min(start + self.bert_sub_batch, total))))
        return torch.cat(parts, dim=0)

    def encode_context(
        self,
        speech_input_ids: torch.Tensor,
        speech_attention_mask: torch.Tensor,
        turn_counts: torch.Tensor,
        turn_mask: torch.Tensor,
        au_input_ids: torch.Tensor | None = None,
        au_attention_mask: torch.Tensor | None = None,
        au_numeric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        turn_vecs = self._encode_turns_batched(
            speech_input_ids, speech_attention_mask,
            au_input_ids, au_attention_mask, au_numeric,
        )  # (total_turns, fusion_dim)

        batch_size = turn_counts.size(0)
        max_turns = turn_mask.size(1)
        turn_emb = turn_vecs.new_zeros(batch_size, max_turns, self.fusion_dim)

        idx = 0
        for i in range(batch_size):
            n = int(turn_counts[i].item())
            turn_emb[i, :n] = turn_vecs[idx: idx + n]
            idx += n

        session_repr = self.session_encoder(turn_emb, turn_counts)
        context, _ = self.attention(session_repr, turn_mask)
        return context

    def freeze_bert(self) -> None:
        for p in self.turn_encoder.bert.parameters():
            p.requires_grad = False

    def unfreeze_bert(self) -> None:
        for p in self.turn_encoder.bert.parameters():
            p.requires_grad = True


class HMANRegressor(HMANBase):
    """Single-task regression head on top of HMANBase."""

    def __init__(
        self,
        bert_model_name: str,
        fusion_dim: int,
        gru_hidden_dim: int,
        gru_layers: int,
        dropout: float,
        bert_sub_batch: int,
        n_au_numeric: int | None = None,
    ):
        super().__init__(bert_model_name, fusion_dim, gru_hidden_dim, gru_layers,
                         dropout, bert_sub_batch, n_au_numeric)
        self.head = nn.Linear(gru_hidden_dim, 1)

    def forward(
        self,
        speech_input_ids: torch.Tensor,
        speech_attention_mask: torch.Tensor,
        turn_counts: torch.Tensor,
        turn_mask: torch.Tensor,
        au_input_ids: torch.Tensor | None = None,
        au_attention_mask: torch.Tensor | None = None,
        au_numeric: torch.Tensor | None = None,
    ) -> torch.Tensor:  # (batch,)
        context = self.encode_context(
            speech_input_ids, speech_attention_mask, turn_counts, turn_mask,
            au_input_ids, au_attention_mask, au_numeric,
        )
        return self.head(self.shared(context)).squeeze(-1)


class HMANClassifier(HMANBase):
    """Single-task classification head on top of HMANBase."""

    def __init__(
        self,
        bert_model_name: str,
        fusion_dim: int,
        gru_hidden_dim: int,
        gru_layers: int,
        dropout: float,
        bert_sub_batch: int,
        n_classes: int,
        n_au_numeric: int | None = None,
    ):
        super().__init__(bert_model_name, fusion_dim, gru_hidden_dim, gru_layers,
                         dropout, bert_sub_batch, n_au_numeric)
        self.head = nn.Linear(gru_hidden_dim, n_classes)

    def forward(
        self,
        speech_input_ids: torch.Tensor,
        speech_attention_mask: torch.Tensor,
        turn_counts: torch.Tensor,
        turn_mask: torch.Tensor,
        au_input_ids: torch.Tensor | None = None,
        au_attention_mask: torch.Tensor | None = None,
        au_numeric: torch.Tensor | None = None,
    ) -> torch.Tensor:  # (batch, n_classes)
        context = self.encode_context(
            speech_input_ids, speech_attention_mask, turn_counts, turn_mask,
            au_input_ids, au_attention_mask, au_numeric,
        )
        return self.head(self.shared(context))
