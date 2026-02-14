"""
Inference module.

Full inference pipeline combining router, retrieval, and generation
with FastAPI serving endpoint.
"""

from src.inference.pipeline import InferencePipeline

__all__ = [
    "InferencePipeline",
]