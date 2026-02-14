"""
Training Datasets for the three FRLM training phases.

- :class:`RouterDataset`    — Phase 1: token-level factual / linguistic labels
- :class:`RetrievalDataset` — Phase 2: fact retrieval with hard negatives
- :class:`JointDataset`     — Phase 3: combined router + retrieval + generation

All datasets support lazy / streamed loading from disk so that arbitrarily
large corpora can be processed without exhausting RAM.  Each class stores
on-disk paths and reads individual examples on demand via :func:`__getitem__`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ===========================================================================
# RouterDataset — Phase 1
# ===========================================================================


class RouterDataset(Dataset):
    """Token-level binary labels for router pre-training.

    Each example on disk is a JSON file (or one entry in a JSON-Lines
    file) with at least::

        {
            "input_ids":      [int, ...],
            "attention_mask": [int, ...],
            "router_labels":  [int, ...]    # 1 = retrieval, 0 = generation
        }

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``*.json`` or ``*.jsonl`` label files.
    max_seq_length : int
        Pad / truncate to this length.
    files : list[Path], optional
        Pre-selected list of files (overrides directory scan).
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        max_seq_length: int = 1024,
        files: Optional[List[Path]] = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._max_seq_length = max_seq_length

        if files is not None:
            self._files = list(files)
        else:
            self._files = self._scan_files()

        self._index: Optional[List[Tuple[Path, int]]] = None
        self._build_index()

        logger.info(
            "RouterDataset: %d examples from %d files (max_seq=%d)",
            len(self), len(self._files), max_seq_length,
        )

    # ------------------------------------------------------------------ scan

    def _scan_files(self) -> List[Path]:
        if not self._data_dir.exists():
            logger.warning("Data dir does not exist: %s", self._data_dir)
            return []
        globs = list(self._data_dir.glob("*.json")) + list(
            self._data_dir.glob("*.jsonl")
        )
        return sorted(globs)

    def _build_index(self) -> None:
        """Build a lightweight index: ``(file_path, line_offset)`` per example.

        For ``.json`` files each file is one example. For ``.jsonl`` each
        line is one example.
        """
        self._index = []
        for path in self._files:
            if path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    for idx, _ in enumerate(f):
                        self._index.append((path, idx))
            else:
                # Single-example JSON
                self._index.append((path, 0))

    # ------------------------------------------------------------------ len / getitem

    def __len__(self) -> int:
        return len(self._index) if self._index else 0

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        path, line_idx = self._index[idx]  # type: ignore[index]
        raw = self._read_example(path, line_idx)
        return self._process(raw)

    def _read_example(self, path: Path, line_idx: int) -> Dict[str, Any]:
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == line_idx:
                        return json.loads(line)
            raise IndexError(f"Line {line_idx} not found in {path}")
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _process(self, raw: Dict[str, Any]) -> Dict[str, Tensor]:
        seq_len = self._max_seq_length

        input_ids = raw["input_ids"][:seq_len]
        attention_mask = raw.get("attention_mask", [1] * len(input_ids))[:seq_len]
        router_labels = raw["router_labels"][:seq_len]

        # Pad
        pad_len = seq_len - len(input_ids)
        input_ids = input_ids + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        router_labels = router_labels + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "router_labels": torch.tensor(router_labels, dtype=torch.float),
        }


# ===========================================================================
# RetrievalDataset — Phase 2
# ===========================================================================


class RetrievalDataset(Dataset):
    """Fact-retrieval examples with pre-computed embeddings.

    Each on-disk example contains::

        {
            "input_ids":             [int, ...],
            "attention_mask":        [int, ...],
            "span_mask":             [int, ...],   # 1 = retrieval position
            "positive_embedding":    [float, ...], # (emb_dim,)
            "negative_embeddings":   [[float, ...], ...],  # (num_neg, emb_dim)
            "fact_id":               str           # optional ground-truth fact id
        }

    Alternatively, embeddings may be stored as memory-mapped ``.npy`` files
    referenced by a ``positive_embedding_path`` / ``negative_embeddings_path``
    field (see :meth:`_load_embedding`).

    Parameters
    ----------
    data_dir : str or Path
        Directory containing example files.
    max_seq_length : int
        Pad / truncate sequences.
    embedding_dim : int
        Expected dimensionality of fact embeddings.
    num_negatives : int
        Number of hard-negative embeddings per example (pad / truncate).
    files : list[Path], optional
        Pre-selected list of files.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        max_seq_length: int = 1024,
        embedding_dim: int = 768,
        num_negatives: int = 20,
        files: Optional[List[Path]] = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._max_seq_length = max_seq_length
        self._embedding_dim = embedding_dim
        self._num_negatives = num_negatives

        if files is not None:
            self._files = list(files)
        else:
            self._files = sorted(
                list(self._data_dir.glob("*.json"))
                + list(self._data_dir.glob("*.jsonl"))
            ) if self._data_dir.exists() else []

        self._index: List[Tuple[Path, int]] = []
        self._build_index()

        logger.info(
            "RetrievalDataset: %d examples, emb_dim=%d, num_neg=%d",
            len(self), embedding_dim, num_negatives,
        )

    def _build_index(self) -> None:
        self._index = []
        for path in self._files:
            if path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    for idx, _ in enumerate(f):
                        self._index.append((path, idx))
            else:
                self._index.append((path, 0))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        path, line_idx = self._index[idx]
        raw = self._read_example(path, line_idx)
        return self._process(raw)

    def _read_example(self, path: Path, line_idx: int) -> Dict[str, Any]:
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == line_idx:
                        return json.loads(line)
            raise IndexError(f"Line {line_idx} not found in {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_embedding(
        self, raw: Dict[str, Any], key: str, path_key: str
    ) -> np.ndarray:
        """Load an embedding from inline JSON or a ``.npy`` memory-map."""
        if key in raw and raw[key] is not None:
            return np.asarray(raw[key], dtype=np.float32)
        if path_key in raw:
            return np.load(raw[path_key], mmap_mode="r").astype(np.float32)
        return np.zeros(self._embedding_dim, dtype=np.float32)

    def _process(self, raw: Dict[str, Any]) -> Dict[str, Tensor]:
        seq = self._max_seq_length
        edim = self._embedding_dim
        n_neg = self._num_negatives

        input_ids = raw["input_ids"][:seq]
        attn = raw.get("attention_mask", [1] * len(input_ids))[:seq]
        span_mask = raw.get("span_mask", [0] * len(input_ids))[:seq]

        pad = seq - len(input_ids)
        input_ids = input_ids + [0] * pad
        attn = attn + [0] * pad
        span_mask = span_mask + [0] * pad

        # Positive embedding
        pos_emb = self._load_embedding(raw, "positive_embedding", "positive_embedding_path")
        if pos_emb.ndim == 0 or pos_emb.shape[0] != edim:
            pos_emb = np.zeros(edim, dtype=np.float32)

        # Negative embeddings
        neg_emb = self._load_embedding(raw, "negative_embeddings", "negative_embeddings_path")
        if neg_emb.ndim == 1:
            neg_emb = neg_emb.reshape(1, -1)
        # Pad / truncate to (num_negatives, emb_dim)
        if neg_emb.shape[0] < n_neg:
            padding = np.zeros((n_neg - neg_emb.shape[0], edim), dtype=np.float32)
            neg_emb = np.concatenate([neg_emb, padding], axis=0)
        else:
            neg_emb = neg_emb[:n_neg]
        if neg_emb.shape[1] != edim:
            neg_emb = np.zeros((n_neg, edim), dtype=np.float32)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "span_mask": torch.tensor(span_mask, dtype=torch.float),
            "positive_embedding": torch.from_numpy(pos_emb.copy()),
            "negative_embeddings": torch.from_numpy(neg_emb.copy()),
        }


# ===========================================================================
# JointDataset — Phase 3
# ===========================================================================


class JointDataset(Dataset):
    """Combined dataset for joint fine-tuning.

    Each on-disk example contains all fields needed for the three loss
    components::

        {
            "input_ids":             [int, ...],
            "attention_mask":        [int, ...],
            "router_labels":         [int, ...],
            "span_mask":             [int, ...],
            "positive_embedding":    [float, ...] or null,
            "negative_embeddings":   [[float, ...], ...] or null,
            "token_labels":          [int, ...]   # next-token ids for generation loss
        }

    Parameters
    ----------
    data_dir : str or Path
    max_seq_length : int
    embedding_dim : int
    num_negatives : int
    files : list[Path], optional
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        max_seq_length: int = 1024,
        embedding_dim: int = 768,
        num_negatives: int = 20,
        files: Optional[List[Path]] = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._max_seq_length = max_seq_length
        self._embedding_dim = embedding_dim
        self._num_negatives = num_negatives

        if files is not None:
            self._files = list(files)
        else:
            self._files = sorted(
                list(self._data_dir.glob("*.json"))
                + list(self._data_dir.glob("*.jsonl"))
            ) if self._data_dir.exists() else []

        self._index: List[Tuple[Path, int]] = []
        self._build_index()

        logger.info(
            "JointDataset: %d examples, emb_dim=%d, num_neg=%d",
            len(self), embedding_dim, num_negatives,
        )

    def _build_index(self) -> None:
        self._index = []
        for path in self._files:
            if path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    for idx, _ in enumerate(f):
                        self._index.append((path, idx))
            else:
                self._index.append((path, 0))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        path, line_idx = self._index[idx]
        raw = self._read_example(path, line_idx)
        return self._process(raw)

    def _read_example(self, path: Path, line_idx: int) -> Dict[str, Any]:
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == line_idx:
                        return json.loads(line)
            raise IndexError(f"Line {line_idx} not found in {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_embedding(
        self, raw: Dict[str, Any], key: str, path_key: str
    ) -> np.ndarray:
        if key in raw and raw[key] is not None:
            return np.asarray(raw[key], dtype=np.float32)
        if path_key in raw:
            return np.load(raw[path_key], mmap_mode="r").astype(np.float32)
        return np.zeros(self._embedding_dim, dtype=np.float32)

    def _process(self, raw: Dict[str, Any]) -> Dict[str, Tensor]:
        seq = self._max_seq_length
        edim = self._embedding_dim
        n_neg = self._num_negatives

        input_ids = raw["input_ids"][:seq]
        attn = raw.get("attention_mask", [1] * len(input_ids))[:seq]
        router_labels = raw.get("router_labels", [0] * len(input_ids))[:seq]
        span_mask = raw.get("span_mask", router_labels[:])[:seq]
        token_labels = raw.get("token_labels", [-100] * len(input_ids))[:seq]

        pad = seq - len(input_ids)
        input_ids = input_ids + [0] * pad
        attn = attn + [0] * pad
        router_labels = router_labels + [0] * pad
        span_mask = span_mask + [0] * pad
        token_labels = token_labels + [-100] * pad

        # Positive embedding
        pos_emb = self._load_embedding(raw, "positive_embedding", "positive_embedding_path")
        if pos_emb.ndim == 0 or pos_emb.shape[-1] != edim:
            pos_emb = np.zeros(edim, dtype=np.float32)

        # Negative embeddings
        neg_raw = raw.get("negative_embeddings")
        if neg_raw is not None:
            neg_emb = np.asarray(neg_raw, dtype=np.float32)
        elif "negative_embeddings_path" in raw:
            neg_emb = np.load(raw["negative_embeddings_path"], mmap_mode="r").astype(np.float32)
        else:
            neg_emb = np.zeros((n_neg, edim), dtype=np.float32)

        if neg_emb.ndim == 1:
            neg_emb = neg_emb.reshape(1, -1)
        if neg_emb.shape[0] < n_neg:
            padding = np.zeros((n_neg - neg_emb.shape[0], edim), dtype=np.float32)
            neg_emb = np.concatenate([neg_emb, padding], axis=0)
        else:
            neg_emb = neg_emb[:n_neg]
        if neg_emb.shape[1] != edim:
            neg_emb = np.zeros((n_neg, edim), dtype=np.float32)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "router_labels": torch.tensor(router_labels, dtype=torch.float),
            "span_mask": torch.tensor(span_mask, dtype=torch.float),
            "positive_embedding": torch.from_numpy(pos_emb.copy()),
            "negative_embeddings": torch.from_numpy(neg_emb.copy()),
            "token_labels": torch.tensor(token_labels, dtype=torch.long),
        }
