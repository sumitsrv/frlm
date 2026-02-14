"""
FAISS Fact Index — IVF-PQ vector index over SapBERT fact embeddings.

Provides:
- ``build_index``  — train + populate from numpy arrays
- ``search`` / ``search_batch`` — top-k nearest-neighbour retrieval
- ``mine_hard_negatives`` — hard-negative mining for contrastive training
- ``save_index`` / ``load_index`` — persistence (index + fact-id mapping)
- ``index_stats`` — diagnostics
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover – CI may lack faiss
    faiss = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class FAISSFactIndex:
    """Manages a FAISS IVF-PQ index with a parallel ``faiss_idx → fact_id`` mapping.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the vectors (must match SapBERT output = 768).
    index_type : str
        FAISS factory string, e.g. ``"IVF4096,PQ64"``.
    metric : str
        ``"L2"`` or ``"IP"`` (inner product).
    nprobe : int
        Number of IVF cells to visit at search time.
    use_gpu : bool
        Move the index to GPU after building.
    gpu_id : int
        CUDA device ordinal when *use_gpu* is True.
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
        if faiss is None:
            raise ImportError(
                "faiss is required for FAISSFactIndex. "
                "Install with: pip install faiss-cpu  (or faiss-gpu)"
            )

        self._embedding_dim = embedding_dim
        self._index_type = index_type
        self._metric_name = metric
        self._nprobe = nprobe
        self._use_gpu = use_gpu
        self._gpu_id = gpu_id

        self._metric = (
            faiss.METRIC_INNER_PRODUCT if metric.upper() == "IP" else faiss.METRIC_L2
        )

        # Populated by build_index / load_index
        self._index: Optional[Any] = None
        self._fact_ids: List[str] = []
        self._fact_id_to_idx: Dict[str, int] = {}

        logger.info(
            "FAISSFactIndex created: dim=%d, type=%s, metric=%s, nprobe=%d, gpu=%s",
            embedding_dim,
            index_type,
            metric,
            nprobe,
            use_gpu,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Whether the underlying FAISS index has been trained."""
        return self._index is not None and self._index.is_trained

    @property
    def ntotal(self) -> int:
        """Number of vectors stored in the index."""
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(
        self,
        embeddings: np.ndarray,
        fact_ids: List[str],
        train_sample_size: Optional[int] = None,
    ) -> None:
        """Train and populate the FAISS index.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(N, embedding_dim)``, dtype ``float32``.
        fact_ids : List[str]
            SHA-256 fact identifiers, same length as *embeddings*.
        train_sample_size : int | None
            Number of vectors to use for IVF/PQ training.
            Defaults to ``min(N, 500_000)``.

        Raises
        ------
        ValueError
            If shapes or lengths are inconsistent.
        """
        n, d = embeddings.shape
        if d != self._embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._embedding_dim}, got {d}"
            )
        if len(fact_ids) != n:
            raise ValueError(
                f"fact_ids length ({len(fact_ids)}) != embeddings rows ({n})"
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        logger.info(
            "Building FAISS index: type=%s, n=%d, dim=%d", self._index_type, n, d
        )

        # Create the index
        index = faiss.index_factory(d, self._index_type, self._metric)

        # Train
        if train_sample_size is None:
            train_sample_size = min(n, 500_000)
        train_sample_size = min(train_sample_size, n)

        if train_sample_size < n:
            rng = np.random.default_rng(42)
            train_indices = rng.choice(n, size=train_sample_size, replace=False)
            train_vectors = embeddings[train_indices]
        else:
            train_vectors = embeddings

        logger.info("Training index on %d vectors …", len(train_vectors))
        index.train(train_vectors)

        # Add vectors
        logger.info("Adding %d vectors to index …", n)
        index.add(embeddings)

        # Set nprobe (on the IVF layer if present)
        try:
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", self._nprobe)
        except Exception:
            logger.debug("Could not set nprobe on index (may not be IVF-based)")

        # Optionally move to GPU
        if self._use_gpu and hasattr(faiss, "index_cpu_to_gpu"):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, self._gpu_id, index)
                logger.info("Index moved to GPU %d", self._gpu_id)
            except Exception:
                logger.warning("GPU transfer failed; keeping index on CPU")

        self._index = index
        self._fact_ids = list(fact_ids)
        self._fact_id_to_idx = {fid: i for i, fid in enumerate(fact_ids)}

        logger.info(
            "Index built: ntotal=%d, is_trained=%s",
            self._index.ntotal,
            self._index.is_trained,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if self._index is None or not self._index.is_trained:
            raise RuntimeError(
                "Index is not built/loaded. Call build_index() or load_index() first."
            )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Search for the *top_k* nearest facts.

        Parameters
        ----------
        query_embedding : np.ndarray
            Shape ``(embedding_dim,)`` or ``(1, embedding_dim)``.
        top_k : int
            Number of results.

        Returns
        -------
        List[Tuple[str, float]]
            ``(fact_id, distance)`` pairs sorted by ascending distance.
        """
        self._ensure_ready()

        q = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        distances, indices = self._index.search(q, top_k)

        results: List[Tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue  # FAISS returns -1 for insufficient results
            results.append((self._fact_ids[int(idx)], float(dist)))
        return results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """Batch search: one result list per query.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Shape ``(num_queries, embedding_dim)``.
        top_k : int
            Number of results per query.

        Returns
        -------
        List[List[Tuple[str, float]]]
            Outer list has ``num_queries`` elements.
        """
        self._ensure_ready()

        q = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        distances, indices = self._index.search(q, top_k)

        all_results: List[List[Tuple[str, float]]] = []
        for row_idx in range(q.shape[0]):
            row: List[Tuple[str, float]] = []
            for idx, dist in zip(indices[row_idx], distances[row_idx]):
                if idx < 0:
                    continue
                row.append((self._fact_ids[int(idx)], float(dist)))
            all_results.append(row)
        return all_results

    # ------------------------------------------------------------------
    # Hard-negative mining
    # ------------------------------------------------------------------

    def mine_hard_negatives(
        self,
        query_embedding: np.ndarray,
        positive_fact_id: str,
        num_negatives: int = 20,
        top_k_candidates: int = 50,
    ) -> List[str]:
        """Mine hard negatives from the index.

        Retrieve the *top_k_candidates* nearest neighbours, exclude the
        positive, and sample *num_negatives* weighted by similarity
        (closer = more likely to be sampled).

        Parameters
        ----------
        query_embedding : np.ndarray
            Shape ``(embedding_dim,)``.
        positive_fact_id : str
            The correct fact id (excluded from negatives).
        num_negatives : int
            Number of hard negatives to return.
        top_k_candidates : int
            How many neighbours to consider.

        Returns
        -------
        List[str]
            Fact ids of the selected hard negatives.
        """
        self._ensure_ready()

        candidates = self.search(query_embedding, top_k=top_k_candidates)

        # Filter out the positive
        filtered = [
            (fid, dist) for fid, dist in candidates if fid != positive_fact_id
        ]

        if not filtered:
            return []

        fact_ids_arr = [fid for fid, _ in filtered]
        distances_arr = np.array([d for _, d in filtered], dtype=np.float32)

        # Convert distances to similarity weights (closer ↔ higher weight).
        # For L2 distance, use inverse; for IP, distances are similarities already.
        if self._metric_name.upper() == "IP":
            weights = distances_arr.copy()
        else:
            # L2: smaller distance = harder negative = higher weight
            weights = 1.0 / (distances_arr + 1e-8)

        # Normalise to probability distribution
        weights = weights / weights.sum()

        num_to_sample = min(num_negatives, len(fact_ids_arr))
        rng = np.random.default_rng()
        chosen_indices = rng.choice(
            len(fact_ids_arr),
            size=num_to_sample,
            replace=False,
            p=weights,
        )
        return [fact_ids_arr[i] for i in chosen_indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path) -> None:
        """Save the FAISS index and the fact-id mapping to disk.

        Creates two files:
        - ``<path>.faiss`` — the binary FAISS index
        - ``<path>.meta.json`` — the fact-id list and config

        Parameters
        ----------
        path : str | Path
            Base path (without extension).
        """
        self._ensure_ready()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss_path = path.with_suffix(".faiss")
        meta_path = path.with_suffix(".meta.json")

        # If on GPU, copy back to CPU for serialisation
        cpu_index = self._index
        if hasattr(faiss, "index_gpu_to_cpu"):
            try:
                cpu_index = faiss.index_gpu_to_cpu(self._index)
            except Exception:
                pass  # already on CPU

        # Atomic write for the FAISS file
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".faiss.tmp"
        )
        os.close(tmp_fd)
        try:
            faiss.write_index(cpu_index, tmp_path)
            os.replace(tmp_path, str(faiss_path))
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        # Metadata (fact ids + config)
        meta = {
            "embedding_dim": self._embedding_dim,
            "index_type": self._index_type,
            "metric": self._metric_name,
            "nprobe": self._nprobe,
            "ntotal": self.ntotal,
            "fact_ids": self._fact_ids,
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".meta.tmp"
        )
        os.close(tmp_fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            os.replace(tmp_path, str(meta_path))
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        logger.info(
            "Index saved: %s (%d vectors) + %s",
            faiss_path,
            self.ntotal,
            meta_path,
        )

    def load_index(self, path: str | Path) -> None:
        """Load a previously saved index.

        Parameters
        ----------
        path : str | Path
            Base path (without extension).  Expects ``<path>.faiss``
            and ``<path>.meta.json``.
        """
        path = Path(path)
        faiss_path = path.with_suffix(".faiss")
        meta_path = path.with_suffix(".meta.json")

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        # Load metadata
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        self._fact_ids = meta["fact_ids"]
        self._fact_id_to_idx = {fid: i for i, fid in enumerate(self._fact_ids)}
        self._embedding_dim = meta.get("embedding_dim", self._embedding_dim)
        self._index_type = meta.get("index_type", self._index_type)
        self._metric_name = meta.get("metric", self._metric_name)
        self._nprobe = meta.get("nprobe", self._nprobe)

        # Load FAISS index
        index = faiss.read_index(str(faiss_path))

        # Set nprobe
        try:
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", self._nprobe)
        except Exception:
            pass

        # Optionally move to GPU
        if self._use_gpu and hasattr(faiss, "index_cpu_to_gpu"):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, self._gpu_id, index)
                logger.info("Loaded index moved to GPU %d", self._gpu_id)
            except Exception:
                logger.warning("GPU transfer failed; keeping on CPU")

        self._index = index

        logger.info(
            "Index loaded from %s: ntotal=%d, trained=%s",
            faiss_path,
            self.ntotal,
            self.is_trained,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def index_stats(self) -> Dict[str, Any]:
        """Return diagnostic information about the current index.

        Returns
        -------
        Dict[str, Any]
            Keys: ``ntotal``, ``embedding_dim``, ``index_type``, ``metric``,
            ``nprobe``, ``is_trained``, ``num_fact_ids``.
        """
        return {
            "ntotal": self.ntotal,
            "embedding_dim": self._embedding_dim,
            "index_type": self._index_type,
            "metric": self._metric_name,
            "nprobe": self._nprobe,
            "is_trained": self.is_trained,
            "num_fact_ids": len(self._fact_ids),
        }

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def fact_id_for_index(self, idx: int) -> Optional[str]:
        """Return the fact_id for a FAISS internal index, or *None*."""
        if 0 <= idx < len(self._fact_ids):
            return self._fact_ids[idx]
        return None

    def index_for_fact_id(self, fact_id: str) -> Optional[int]:
        """Return the FAISS internal index for a fact_id, or *None*."""
        return self._fact_id_to_idx.get(fact_id)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: object) -> "FAISSFactIndex":
        """Construct from a ``FAISSConfig`` Pydantic model.

        Parameters
        ----------
        cfg : FAISSConfig
            ``config.config.FAISSConfig`` (or any object with matching attrs).

        Returns
        -------
        FAISSFactIndex
        """
        return cls(
            embedding_dim=cfg.embedding_dim,  # type: ignore[attr-defined]
            index_type=cfg.index_type,  # type: ignore[attr-defined]
            metric=cfg.metric,  # type: ignore[attr-defined]
            nprobe=cfg.nprobe,  # type: ignore[attr-defined]
            use_gpu=cfg.use_gpu,  # type: ignore[attr-defined]
            gpu_id=cfg.gpu_id,  # type: ignore[attr-defined]
        )
