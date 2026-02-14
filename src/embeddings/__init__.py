"""
Embeddings module.

Frozen SapBERT encoder for fact embedding, FAISS index management
with IVF-PQ on GPU, and hierarchical multi-level indexing.
"""

from src.embeddings.sapbert import SapBERTEncoder
from src.embeddings.faiss_index import FAISSFactIndex
from src.embeddings.hierarchical import HierarchicalIndex

__all__ = [
    "SapBERTEncoder",
    "FAISSFactIndex",
    "HierarchicalIndex",
]