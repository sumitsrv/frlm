"""
Model module.

FRLM architecture combining BioMedLM backbone with router head,
retrieval head (semantic + granularity + temporal), generation head,
and multi-task loss functions.
"""

from src.model.backbone import BackboneOutput, BioMedLMBackbone
from src.model.frlm import FRLMModel, FRLMOutput
from src.model.generation_head import GenerationHead
from src.model.losses import (
    FRLMCombinedLoss,
    GenerationLoss,
    InfoNCELoss,
    RouterLoss,
)
from src.model.retrieval_head import QuerySignature, RetrievalHead
from src.model.router_head import RouterHead

__all__ = [
    "BackboneOutput",
    "BioMedLMBackbone",
    "FRLMCombinedLoss",
    "FRLMModel",
    "FRLMOutput",
    "GenerationHead",
    "GenerationLoss",
    "InfoNCELoss",
    "QuerySignature",
    "RetrievalHead",
    "RouterHead",
    "RouterLoss",
]