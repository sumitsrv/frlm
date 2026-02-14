"""
Inference module.

Full inference pipeline combining router, retrieval, and generation
with FastAPI serving endpoint.
"""

from src.inference.pipeline import (
    FRLMResponse,
    InferencePipeline,
    RetrievedFact,
    RouterDecision,
)
from src.inference.server import create_app

__all__ = [
    "InferencePipeline",
    "FRLMResponse",
    "RetrievedFact",
    "RouterDecision",
    "create_app",
]