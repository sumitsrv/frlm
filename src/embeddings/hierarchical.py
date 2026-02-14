"""
Hierarchical Multi-Level FAISS Index.

Maintains four independent ``FAISSFactIndex`` instances — one per
granularity level defined in the FRLM architecture:

- **Level 0 (atomic)**: one embedding per individual fact
  (direct SapBERT encoding).
- **Level 1 (relation)**: one embedding per unique *(subject, object)*
  entity pair — mean-pool of all atomic embeddings for that pair.
- **Level 2 (entity)**: one embedding per unique entity — mean-pool
  of all atomic embeddings involving that entity.
- **Level 3 (cluster)**: one embedding per curated ``FactCluster`` —
  mean-pool of constituent atomic embeddings.

Public API
----------
- ``build_all_levels(facts, atomic_embeddings, clusters)``
- ``search_at_level(query, level, top_k)``
- ``resolve(query, level, temporal_mode, timestamp, kg_client)``
- ``save(directory)`` / ``load(directory)``
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.embeddings.faiss_index import FAISSFactIndex

if TYPE_CHECKING:
    from src.kg.schema import Fact, FactCluster

logger = logging.getLogger(__name__)

# Canonical level identifiers
LEVEL_ATOMIC = 0
LEVEL_RELATION = 1
LEVEL_ENTITY = 2
LEVEL_CLUSTER = 3

LEVEL_NAMES: Dict[int, str] = {
    LEVEL_ATOMIC: "atomic",
    LEVEL_RELATION: "relation",
    LEVEL_ENTITY: "entity",
    LEVEL_CLUSTER: "cluster",
}
NUM_LEVELS = len(LEVEL_NAMES)


def _mean_pool(vectors: List[np.ndarray]) -> np.ndarray:
    """Compute L2-normalised mean of a list of vectors."""
    stacked = np.vstack(vectors).astype(np.float32)
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean /= norm
    return mean


class HierarchicalIndex:
    """Four-level hierarchical FAISS index over SapBERT fact embeddings.

    Parameters
    ----------
    embedding_dim : int
        Vector dimensionality (default 768).
    index_type : str
        FAISS factory string (e.g. ``"IVF4096,PQ64"``).
        For small datasets the caller may pass ``"Flat"`` instead.
    metric : str
        ``"L2"`` or ``"IP"``.
    nprobe : int
        Number of IVF cells to visit during search.
    use_gpu : bool
        Whether to place indices on GPU.
    gpu_id : int
        GPU device ordinal.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "IVF4096,PQ64",
        metric: str = "L2",
        nprobe: int = 64,
        use_gpu: bool = False,
        gpu_id: int = 0,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._index_type = index_type
        self._metric = metric
        self._nprobe = nprobe
        self._use_gpu = use_gpu
        self._gpu_id = gpu_id

        self._levels: Dict[int, FAISSFactIndex] = {}

        # Auxiliary look-ups populated during build
        # Level 1: pair_key → list of atomic fact_ids
        self._pair_to_facts: Dict[str, List[str]] = {}
        # Level 2: entity_canonical_id → list of atomic fact_ids
        self._entity_to_facts: Dict[str, List[str]] = {}
        # Level 3: cluster_id → list of atomic fact_ids
        self._cluster_to_facts: Dict[str, List[str]] = {}

        logger.info(
            "HierarchicalIndex created: dim=%d, type=%s, metric=%s",
            embedding_dim,
            index_type,
            metric,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pair_key(subject_id: str, object_id: str) -> str:
        """Stable key for a (subject, object) entity pair."""
        return f"{subject_id}||{object_id}"

    def _make_faiss_index(self, n_vectors: int) -> FAISSFactIndex:
        """Create a ``FAISSFactIndex`` with an appropriate factory string.

        For very small datasets (< 4 × nlist) we fall back to a flat index
        because IVF training requires at least *nlist* vectors.
        """
        idx_type = self._index_type
        # Heuristic: IVF needs at least 39 * nlist vectors for reliable training.
        # Fall back to Flat for tiny sets.
        if "IVF" in idx_type:
            # Extract nlist from the factory string (e.g. IVF4096,PQ64 → 4096)
            try:
                nlist_str = idx_type.split("IVF")[1].split(",")[0]
                nlist = int(nlist_str)
            except (IndexError, ValueError):
                nlist = 4096
            if n_vectors < max(nlist, 256):
                idx_type = "Flat"
                logger.info(
                    "Too few vectors (%d) for %s; falling back to Flat",
                    n_vectors,
                    self._index_type,
                )

        return FAISSFactIndex(
            embedding_dim=self._embedding_dim,
            index_type=idx_type,
            metric=self._metric,
            nprobe=self._nprobe,
            use_gpu=self._use_gpu,
            gpu_id=self._gpu_id,
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_all_levels(
        self,
        facts: Sequence["Fact"],
        atomic_embeddings: np.ndarray,
        clusters: Optional[Sequence["FactCluster"]] = None,
    ) -> None:
        """Build all four hierarchical index levels.

        Parameters
        ----------
        facts : Sequence[Fact]
            The fact objects, aligned 1-to-1 with *atomic_embeddings*.
        atomic_embeddings : np.ndarray
            Shape ``(len(facts), embedding_dim)``, dtype ``float32``.
        clusters : Sequence[FactCluster] | None
            Optional pre-built clusters for Level 3.  If *None*, Level 3
            is skipped (empty index).
        """
        n = len(facts)
        if atomic_embeddings.shape[0] != n:
            raise ValueError(
                f"facts length ({n}) != embeddings rows ({atomic_embeddings.shape[0]})"
            )

        atomic_embeddings = np.ascontiguousarray(
            atomic_embeddings, dtype=np.float32
        )

        # Map fact_id → index
        fid_to_idx = {f.fact_id: i for i, f in enumerate(facts)}

        # ---- Level 0: atomic ----------------------------------------
        logger.info("Building Level 0 (atomic): %d vectors", n)
        level0 = self._make_faiss_index(n)
        fact_ids_0 = [f.fact_id for f in facts]
        if n > 0:
            level0.build_index(atomic_embeddings, fact_ids_0)
        self._levels[LEVEL_ATOMIC] = level0

        # ---- Level 1: relation (entity pair) -------------------------
        pair_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
        pair_fact_map: Dict[str, List[str]] = defaultdict(list)

        for i, fact in enumerate(facts):
            pk = self._pair_key(
                fact.subject.canonical_id, fact.object.canonical_id
            )
            pair_embeddings[pk].append(atomic_embeddings[i])
            pair_fact_map[pk].append(fact.fact_id)

        pair_keys = sorted(pair_embeddings.keys())
        pair_vecs = np.array(
            [_mean_pool(pair_embeddings[pk]) for pk in pair_keys],
            dtype=np.float32,
        ) if pair_keys else np.empty((0, self._embedding_dim), dtype=np.float32)

        logger.info("Building Level 1 (relation): %d pair vectors", len(pair_keys))
        level1 = self._make_faiss_index(len(pair_keys))
        if len(pair_keys) > 0:
            level1.build_index(pair_vecs, pair_keys)
        self._levels[LEVEL_RELATION] = level1
        self._pair_to_facts = dict(pair_fact_map)

        # ---- Level 2: entity -----------------------------------------
        entity_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
        entity_fact_map: Dict[str, List[str]] = defaultdict(list)

        for i, fact in enumerate(facts):
            for eid in (fact.subject.canonical_id, fact.object.canonical_id):
                entity_embeddings[eid].append(atomic_embeddings[i])
                entity_fact_map[eid].append(fact.fact_id)

        entity_ids = sorted(entity_embeddings.keys())
        entity_vecs = np.array(
            [_mean_pool(entity_embeddings[eid]) for eid in entity_ids],
            dtype=np.float32,
        ) if entity_ids else np.empty((0, self._embedding_dim), dtype=np.float32)

        logger.info("Building Level 2 (entity): %d entity vectors", len(entity_ids))
        level2 = self._make_faiss_index(len(entity_ids))
        if len(entity_ids) > 0:
            level2.build_index(entity_vecs, entity_ids)
        self._levels[LEVEL_ENTITY] = level2
        self._entity_to_facts = dict(entity_fact_map)

        # ---- Level 3: cluster ----------------------------------------
        cluster_ids_list: List[str] = []
        cluster_vecs_list: List[np.ndarray] = []
        cluster_fact_map: Dict[str, List[str]] = {}

        if clusters:
            for cluster in clusters:
                cid = cluster.cluster_id or cluster.anchor_entity
                if not cid:
                    continue
                member_indices = [
                    fid_to_idx[f.fact_id]
                    for f in cluster.facts
                    if f.fact_id in fid_to_idx
                ]
                if not member_indices:
                    continue
                member_embs = [atomic_embeddings[i] for i in member_indices]
                cluster_ids_list.append(cid)
                cluster_vecs_list.append(_mean_pool(member_embs))
                cluster_fact_map[cid] = [
                    facts[i].fact_id for i in member_indices
                ]

        cluster_vecs = (
            np.array(cluster_vecs_list, dtype=np.float32)
            if cluster_vecs_list
            else np.empty((0, self._embedding_dim), dtype=np.float32)
        )

        logger.info("Building Level 3 (cluster): %d cluster vectors", len(cluster_ids_list))
        level3 = self._make_faiss_index(len(cluster_ids_list))
        if len(cluster_ids_list) > 0:
            level3.build_index(cluster_vecs, cluster_ids_list)
        self._levels[LEVEL_CLUSTER] = level3
        self._cluster_to_facts = cluster_fact_map

        logger.info(
            "Hierarchical index complete — L0=%d, L1=%d, L2=%d, L3=%d",
            level0.ntotal,
            level1.ntotal,
            level2.ntotal,
            level3.ntotal,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_at_level(
        self,
        query: np.ndarray,
        level: int,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Search the index at a specific granularity level.

        Parameters
        ----------
        query : np.ndarray
            Shape ``(embedding_dim,)`` or ``(1, embedding_dim)``.
        level : int
            0–3 (atomic, relation, entity, cluster).
        top_k : int
            Number of results.

        Returns
        -------
        List[Tuple[str, float]]
            ``(id, distance)`` pairs.  The id semantics depend on level:
            - Level 0: fact_id
            - Level 1: pair key (``subject_id||object_id``)
            - Level 2: entity canonical_id
            - Level 3: cluster_id
        """
        if level not in self._levels:
            raise ValueError(
                f"Level {level} not built. Available: {list(self._levels)}"
            )
        return self._levels[level].search(query, top_k=top_k)

    def expand_to_fact_ids(
        self,
        ids: List[str],
        level: int,
    ) -> List[str]:
        """Expand level-specific ids to underlying atomic fact_ids.

        Parameters
        ----------
        ids : List[str]
            Identifiers returned by ``search_at_level``.
        level : int
            The granularity level the ids belong to.

        Returns
        -------
        List[str]
            De-duplicated list of atomic fact_ids.
        """
        if level == LEVEL_ATOMIC:
            return list(ids)

        lookup: Dict[str, List[str]]
        if level == LEVEL_RELATION:
            lookup = self._pair_to_facts
        elif level == LEVEL_ENTITY:
            lookup = self._entity_to_facts
        elif level == LEVEL_CLUSTER:
            lookup = self._cluster_to_facts
        else:
            raise ValueError(f"Unknown level: {level}")

        seen: set[str] = set()
        result: List[str] = []
        for key in ids:
            for fid in lookup.get(key, []):
                if fid not in seen:
                    seen.add(fid)
                    result.append(fid)
        return result

    def resolve(
        self,
        query: np.ndarray,
        level: int,
        temporal_mode: str = "CURRENT",
        timestamp: Optional[date] = None,
        kg_client: Optional[Any] = None,
        top_k: int = 10,
    ) -> List["Fact"]:
        """Full resolution: FAISS search → granularity expansion → temporal filtering.

        Parameters
        ----------
        query : np.ndarray
            Shape ``(embedding_dim,)``.
        level : int
            Granularity level (0–3).
        temporal_mode : str
            ``"CURRENT"``, ``"AT_TIMESTAMP"``, or ``"HISTORY"``.
        timestamp : date | None
            Required when *temporal_mode* is ``"AT_TIMESTAMP"``.
        kg_client : Neo4jClient | None
            If provided, facts are fetched from Neo4j and temporally resolved.
            If *None*, only fact_ids are returned (wrapped in a stub).
        top_k : int
            Number of nearest neighbours at the given level.

        Returns
        -------
        List[Fact]
            Temporally-resolved facts.
        """
        from src.kg.temporal import TemporalResolver

        search_results = self.search_at_level(query, level=level, top_k=top_k)
        ids = [sid for sid, _ in search_results]
        fact_ids = self.expand_to_fact_ids(ids, level)

        if kg_client is None:
            logger.warning(
                "No kg_client provided; returning fact_ids without temporal resolution"
            )
            return []

        # Fetch facts from KG
        facts: List["Fact"] = []
        for fid in fact_ids:
            fact = kg_client.get_fact_by_id(fid)
            if fact is not None:
                facts.append(fact)

        # Apply temporal resolution
        resolver = TemporalResolver()
        resolved = resolver.resolve(
            facts,
            mode=temporal_mode,
            timestamp=timestamp,
        )
        return resolved

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save all level indices to a directory.

        Creates ``<directory>/level_<N>_<name>`` base-paths for each level.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        import json as _json

        for level, idx in self._levels.items():
            name = LEVEL_NAMES.get(level, str(level))
            base = directory / f"level_{level}_{name}"
            if idx.ntotal == 0:
                logger.info("Skipping empty Level %d (%s)", level, name)
                continue
            idx.save_index(base)

        # Save auxiliary mappings
        aux = {
            "pair_to_facts": self._pair_to_facts,
            "entity_to_facts": self._entity_to_facts,
            "cluster_to_facts": self._cluster_to_facts,
        }
        aux_path = directory / "hierarchical_aux.json"
        with open(aux_path, "w", encoding="utf-8") as fh:
            _json.dump(aux, fh)

        logger.info("Hierarchical index saved to %s", directory)

    def load(self, directory: str | Path) -> None:
        """Load all level indices from a directory."""
        directory = Path(directory)
        import json as _json

        for level in range(NUM_LEVELS):
            name = LEVEL_NAMES[level]
            base = directory / f"level_{level}_{name}"
            faiss_path = base.with_suffix(".faiss")
            if not faiss_path.exists():
                logger.info("Level %d (%s) index not found, skipping", level, name)
                continue
            idx = self._make_faiss_index(0)  # n_vectors unused for load
            idx.load_index(base)
            self._levels[level] = idx

        # Load auxiliary mappings
        aux_path = directory / "hierarchical_aux.json"
        if aux_path.exists():
            with open(aux_path, "r", encoding="utf-8") as fh:
                aux = _json.load(fh)
            self._pair_to_facts = aux.get("pair_to_facts", {})
            self._entity_to_facts = aux.get("entity_to_facts", {})
            self._cluster_to_facts = aux.get("cluster_to_facts", {})

        logger.info(
            "Hierarchical index loaded from %s (%d levels)",
            directory,
            len(self._levels),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return per-level statistics."""
        return {
            LEVEL_NAMES.get(level, str(level)): idx.index_stats()
            for level, idx in sorted(self._levels.items())
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: object) -> "HierarchicalIndex":
        """Construct from a ``FAISSConfig`` Pydantic model.

        Parameters
        ----------
        cfg : FAISSConfig
            ``config.config.FAISSConfig`` (or object with matching attrs).
        """
        return cls(
            embedding_dim=cfg.embedding_dim,  # type: ignore[attr-defined]
            index_type=cfg.index_type,  # type: ignore[attr-defined]
            metric=cfg.metric,  # type: ignore[attr-defined]
            nprobe=cfg.nprobe,  # type: ignore[attr-defined]
            use_gpu=cfg.use_gpu,  # type: ignore[attr-defined]
            gpu_id=cfg.gpu_id,  # type: ignore[attr-defined]
        )
