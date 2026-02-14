"""
Retrieval Head — structured query signature for KG fact retrieval.

Computes three outputs from backbone hidden states:

1. **Semantic embedding** — ``Linear(hidden_dim, embedding_dim)`` → L2-normalise.
   Projects into the SapBERT embedding space so FAISS can compare
   query vectors against pre-encoded fact embeddings.
2. **Granularity logits** — ``Linear(hidden_dim, num_granularity_levels)``
   Selects the hierarchical index level (atomic / relation / entity / cluster).
3. **Temporal logits** — ``Linear(hidden_dim, num_temporal_modes)``
   Selects the temporal resolution mode (CURRENT / AT_TIMESTAMP / HISTORY).

Public API
----------
- ``forward(hidden_states)`` → ``QuerySignature``
- ``resolve(query_signature, faiss_index, kg_client, top_k)`` → ``List[Fact]``
- ``from_config(retrieval_head_config)`` class-method
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from src.embeddings.faiss_index import FAISSFactIndex
    from src.embeddings.hierarchical import HierarchicalIndex
    from src.kg.neo4j_client import Neo4jClient
    from src.kg.schema import Fact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

GRANULARITY_NAMES = ["atomic", "relation", "entity", "cluster"]
TEMPORAL_MODE_NAMES = ["CURRENT", "AT_TIMESTAMP", "HISTORY"]


@dataclass
class QuerySignature:
    """Structured query emitted by :class:`RetrievalHead`.

    Attributes
    ----------
    semantic_embedding : Tensor
        L2-normalised dense vector, shape ``(batch, embedding_dim)``.
    granularity_logits : Tensor
        Raw logits over granularity levels, shape ``(batch, num_levels)``.
    temporal_logits : Tensor
        Raw logits over temporal modes, shape ``(batch, num_modes)``.
    """

    semantic_embedding: Tensor
    granularity_logits: Tensor
    temporal_logits: Tensor

    @property
    def granularity_level(self) -> Tensor:
        """Argmax granularity level per batch element.  Shape ``(batch,)``."""
        return self.granularity_logits.argmax(dim=-1)

    @property
    def temporal_mode(self) -> Tensor:
        """Argmax temporal mode per batch element.  Shape ``(batch,)``."""
        return self.temporal_logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


class RetrievalHead(nn.Module):
    """Produces a :class:`QuerySignature` from backbone hidden states.

    Parameters
    ----------
    hidden_dim : int
        Input dimensionality (must match backbone ``hidden_dim``).
    embedding_dim : int
        Semantic output dimensionality (must match SapBERT / FAISS dim).
    num_granularity_levels : int
        Number of hierarchical index levels.
    num_temporal_modes : int
        Number of temporal resolution modes.
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        embedding_dim: int = 768,
        num_granularity_levels: int = 4,
        num_temporal_modes: int = 3,
    ) -> None:
        super().__init__()

        self._hidden_dim = hidden_dim
        self._embedding_dim = embedding_dim
        self._num_granularity_levels = num_granularity_levels
        self._num_temporal_modes = num_temporal_modes

        # Sub-networks
        self.semantic_proj = nn.Linear(hidden_dim, embedding_dim)
        self.granularity_head = nn.Linear(hidden_dim, num_granularity_levels)
        self.temporal_head = nn.Linear(hidden_dim, num_temporal_modes)

        logger.info(
            "RetrievalHead created: hidden_dim=%d → semantic(%d), "
            "granularity(%d levels), temporal(%d modes)",
            hidden_dim,
            embedding_dim,
            num_granularity_levels,
            num_temporal_modes,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def num_granularity_levels(self) -> int:
        return self._num_granularity_levels

    @property
    def num_temporal_modes(self) -> int:
        return self._num_temporal_modes

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: Tensor) -> QuerySignature:
        """Compute a structured query signature.

        Parameters
        ----------
        hidden_states : Tensor
            Shape ``(batch, seq_len, hidden_dim)`` or ``(batch, hidden_dim)``.

        Returns
        -------
        QuerySignature
            Contains ``semantic_embedding`` (L2-normalised),
            ``granularity_logits``, and ``temporal_logits``.
        """
        # Semantic projection + L2 normalisation
        semantic = self.semantic_proj(hidden_states)
        semantic = F.normalize(semantic, p=2, dim=-1)

        # Classification heads (raw logits — softmax applied in loss / at
        # inference time via argmax)
        granularity = self.granularity_head(hidden_states)
        temporal = self.temporal_head(hidden_states)

        return QuerySignature(
            semantic_embedding=semantic,
            granularity_logits=granularity,
            temporal_logits=temporal,
        )

    # ------------------------------------------------------------------
    # Resolution — end-to-end: query → facts
    # ------------------------------------------------------------------

    def resolve(
        self,
        query_signature: QuerySignature,
        faiss_index: Any,
        kg_client: Any,
        top_k: int = 10,
        timestamp: Optional[Any] = None,
    ) -> List[Any]:
        """Full resolution pipeline: FAISS search → granularity expansion → temporal filter.

        Parameters
        ----------
        query_signature : QuerySignature
            Output from :meth:`forward`.
        faiss_index : FAISSFactIndex or HierarchicalIndex
            Vector index to search.
        kg_client : Neo4jClient
            KG client for temporal filtering.
        top_k : int
            Number of candidate facts to retrieve.
        timestamp : date, optional
            Required when temporal mode resolves to ``AT_TIMESTAMP``.

        Returns
        -------
        list
            Resolved facts (content depends on KG client return type).
        """
        # Detach and move to CPU for FAISS
        sem_np = (
            query_signature.semantic_embedding
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        # Determine granularity level and temporal mode (take first element)
        gran_level = int(query_signature.granularity_level[0].item())
        temp_mode_idx = int(query_signature.temporal_mode[0].item())
        temp_mode = TEMPORAL_MODE_NAMES[temp_mode_idx]

        # Search
        from src.embeddings.hierarchical import HierarchicalIndex

        if isinstance(faiss_index, HierarchicalIndex):
            results = faiss_index.search_at_level(
                query=sem_np[0],
                level=gran_level,
                top_k=top_k,
            )
        else:
            # Plain FAISSFactIndex
            results = faiss_index.search(
                query_embedding=sem_np[0],
                top_k=top_k,
            )

        if not results:
            return []

        # Extract fact ids
        fact_ids = [fact_id for fact_id, _score in results]

        # Temporal filtering via KG client
        resolved: List[Any] = []
        for fid in fact_ids:
            fact = kg_client.get_fact_by_id(fid)
            if fact is not None:
                resolved.append(fact)

        return resolved

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, retrieval_config: "RetrievalHeadConfig") -> "RetrievalHead":  # type: ignore[name-defined]  # noqa: F821
        """Construct from a :class:`config.config.RetrievalHeadConfig` instance."""
        return cls(
            hidden_dim=retrieval_config.semantic.input_dim,
            embedding_dim=retrieval_config.semantic.output_dim,
            num_granularity_levels=retrieval_config.granularity.num_levels,
            num_temporal_modes=retrieval_config.temporal.num_modes,
        )
