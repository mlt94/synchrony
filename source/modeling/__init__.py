"""
Hierarchical Multimodal Attention Network for psychotherapy BLRI prediction.

Architecture:
    1. Turn Encoder     – Shared DistilBERT + Gated Fusion of speech & AU modalities
    2. Session Encoder  – Bidirectional GRU over the turn sequence
    3. Attention Layer   – Learned attention weights per turn (interpretable)
    4. Output Head       – Linear regressor for BLRI scores (Pr / In)
"""

from .model import (
    HierarchicalMultimodalAttentionNetwork,
    TurnEncoder,
    SessionEncoder,
    TurnAttention,
    GatedFusion,
)
from .dataset import (
    PsychotherapyDataset,
    BucketBatchSampler,
    create_collate_fn,
    create_dataloaders,
)

__all__ = [
    "HierarchicalMultimodalAttentionNetwork",
    "TurnEncoder",
    "SessionEncoder",
    "TurnAttention",
    "GatedFusion",
    "PsychotherapyDataset",
    "BucketBatchSampler",
    "create_collate_fn",
    "create_dataloaders",
]
