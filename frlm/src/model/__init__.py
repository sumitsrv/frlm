"""
Model module.

FRLM architecture combining BioMedLM backbone with router head,
retrieval head (semantic + granularity + temporal), and generation head.
"""

from src.model.frlm import FRLMModel
from src.model.backbone import BioMedLMBackbone
from src.model.router_head import RouterHead
from src.model.retrieval_head import RetrievalHead
from src.model.generation_head import GenerationHead

__all__ = [
    "FRLMModel",
    "BioMedLMBackbone",
    "RouterHead",
    "RetrievalHead",
    "GenerationHead",
]