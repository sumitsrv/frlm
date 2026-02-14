"""
Training module.

Three-phase training pipeline:
  Phase 1 - Router pre-training (binary classification)
  Phase 2 - Retrieval head training (contrastive InfoNCE)
  Phase 3 - Joint fine-tuning (combined loss)
"""

from src.training.router_trainer import RouterTrainer
from src.training.retrieval_trainer import RetrievalTrainer
from src.training.joint_trainer import JointTrainer

__all__ = [
    "RouterTrainer",
    "RetrievalTrainer",
    "JointTrainer",
]