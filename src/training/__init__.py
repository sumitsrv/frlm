"""
Training module.

Three-phase training pipeline:
  Phase 1 - Router pre-training (binary classification)
  Phase 2 - Retrieval head training (contrastive InfoNCE)
  Phase 3 - Joint fine-tuning (combined loss)

Provides trainers, datasets, and utilities for the complete pipeline.
"""

from src.training.dataset import JointDataset, RetrievalDataset, RouterDataset
from src.training.joint_trainer import JointTrainer
from src.training.retrieval_trainer import RetrievalTrainer
from src.training.router_trainer import RouterTrainer
from src.training.utils import (
    CheckpointManager,
    EarlyStopping,
    GradientAccumulator,
    LearningRateScheduler,
    MetricsLogger,
    TrainingState,
    finish_wandb,
    init_wandb,
)

__all__ = [
    # Trainers
    "RouterTrainer",
    "RetrievalTrainer",
    "JointTrainer",
    # Datasets
    "RouterDataset",
    "RetrievalDataset",
    "JointDataset",
    # Utilities
    "TrainingState",
    "CheckpointManager",
    "MetricsLogger",
    "EarlyStopping",
    "GradientAccumulator",
    "LearningRateScheduler",
    "init_wandb",
    "finish_wandb",
]