"""
SapBERT Frozen Encoder — embeds KG facts into a 768-dim L2-normalised space.

The encoder wraps ``cambridgeltl/SapBERT-from-PubMedBERT-fulltext``, which is
loaded once in ``eval()`` mode with **all parameters frozen**.  It is never
trained; its sole purpose is to produce the target embedding space that the
retrieval head learns to project into.

Public API
----------
- ``encode_fact(fact)`` → 768-dim numpy vector
- ``encode_facts_batch(facts, batch_size)`` → (N, 768) numpy array
- ``encode_query(text)`` → 768-dim numpy vector
- ``encode_texts_batch(texts, batch_size)`` → (N, 768) numpy array
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from src.kg.schema import Fact

logger = logging.getLogger(__name__)


def _fact_to_text(fact: "Fact") -> str:
    """Build the canonical text representation of a *Fact* for embedding.

    Format: ``"{subject_label} {relation_type} {object_label}"``
    """
    return (
        f"{fact.subject.label} "
        f"{fact.relation.type.value} "
        f"{fact.object.label}"
    )


class SapBERTEncoder:
    """Frozen SapBERT encoder for knowledge-graph fact embedding.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum token length passed to the tokenizer.
    pool_strategy : str
        One of ``"cls"`` (default), ``"mean"``, ``"max"``.
    device : str | None
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect).
    dtype : str
        ``"float16"`` or ``"float32"``.
    """

    # Class-level constants
    EMBEDDING_DIM: int = 768
    VALID_POOL_STRATEGIES = {"cls", "mean", "max"}

    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        max_length: int = 64,
        pool_strategy: str = "cls",
        device: Optional[str] = None,
        dtype: str = "float32",
    ) -> None:
        if pool_strategy not in self.VALID_POOL_STRATEGIES:
            raise ValueError(
                f"pool_strategy must be one of {self.VALID_POOL_STRATEGIES}, "
                f"got '{pool_strategy}'"
            )

        self._model_name = model_name
        self._max_length = max_length
        self._pool_strategy = pool_strategy

        # Resolve device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        # Resolve dtype
        self._dtype = torch.float16 if dtype == "float16" else torch.float32

        logger.info(
            "Loading SapBERT encoder: model=%s, device=%s, dtype=%s, pool=%s",
            model_name,
            self._device,
            self._dtype,
            pool_strategy,
        )

        # Load tokenizer & model, freeze everything
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.to(device=self._device, dtype=self._dtype)
        self._model.eval()

        # Freeze all parameters
        for param in self._model.parameters():
            param.requires_grad = False

        logger.info(
            "SapBERT encoder ready — %d parameters (all frozen)",
            sum(p.numel() for p in self._model.parameters()),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def pool_strategy(self) -> str:
        return self._pool_strategy

    # ------------------------------------------------------------------
    # Core pooling
    # ------------------------------------------------------------------

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the configured pooling strategy.

        Parameters
        ----------
        last_hidden_state : Tensor
            Shape ``(batch, seq_len, hidden_dim)``.
        attention_mask : Tensor
            Shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Shape ``(batch, hidden_dim)``, L2-normalised.
        """
        if self._pool_strategy == "cls":
            pooled = last_hidden_state[:, 0, :]
        elif self._pool_strategy == "mean":
            # Mask padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / count
        elif self._pool_strategy == "max":
            # Replace padded positions with -inf so they don't win the max
            mask_expanded = attention_mask.unsqueeze(-1).bool()
            filled = last_hidden_state.masked_fill(~mask_expanded, -1e9)
            pooled = filled.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool_strategy: {self._pool_strategy}")

        # L2 normalise
        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    # ------------------------------------------------------------------
    # Text-level encoding (internal)
    # ------------------------------------------------------------------

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a single batch of texts (must fit in memory).

        Returns ``(len(texts), 768)`` float32 numpy array.
        """
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        pooled = self._pool(outputs.last_hidden_state, inputs["attention_mask"])
        return pooled.cpu().float().numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_fact(self, fact: "Fact") -> np.ndarray:
        """Encode a single *Fact* → 768-dim L2-normalised vector.

        Parameters
        ----------
        fact : Fact
            A ``src.kg.schema.Fact`` instance.

        Returns
        -------
        np.ndarray
            Shape ``(768,)`` with dtype ``float32``.
        """
        text = _fact_to_text(fact)
        return self._encode_texts([text])[0]

    def encode_facts_batch(
        self,
        facts: Sequence["Fact"],
        batch_size: int = 256,
    ) -> np.ndarray:
        """Encode a list of *Fact* objects in batches.

        Parameters
        ----------
        facts : Sequence[Fact]
            Facts to embed.
        batch_size : int
            Sub-batch size for the tokenizer / model forward pass.

        Returns
        -------
        np.ndarray
            Shape ``(len(facts), 768)`` with dtype ``float32``.
        """
        texts = [_fact_to_text(f) for f in facts]
        return self.encode_texts_batch(texts, batch_size=batch_size)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode an ad-hoc text query → 768-dim L2-normalised vector.

        Parameters
        ----------
        text : str
            Free-form query string.

        Returns
        -------
        np.ndarray
            Shape ``(768,)`` with dtype ``float32``.
        """
        return self._encode_texts([text])[0]

    def encode_texts_batch(
        self,
        texts: List[str],
        batch_size: int = 256,
    ) -> np.ndarray:
        """Encode arbitrary texts in micro-batches.

        Parameters
        ----------
        texts : List[str]
            Texts to embed.
        batch_size : int
            Maximum number of texts per forward pass.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), 768)`` with dtype ``float32``.
        """
        if not texts:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            emb = self._encode_texts(batch_texts)
            all_embeddings.append(emb)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Encoded batch %d–%d / %d", start, end, len(texts)
                )

        return np.vstack(all_embeddings)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: object) -> "SapBERTEncoder":
        """Construct from a ``SapBERTConfig`` Pydantic model.

        Parameters
        ----------
        cfg : SapBERTConfig
            ``config.config.SapBERTConfig`` (or any object with
            matching attributes).

        Returns
        -------
        SapBERTEncoder
        """
        return cls(
            model_name=cfg.model_name,  # type: ignore[attr-defined]
            max_length=cfg.max_length,  # type: ignore[attr-defined]
            pool_strategy=cfg.pool_strategy,  # type: ignore[attr-defined]
            device=cfg.device,  # type: ignore[attr-defined]
            dtype=cfg.dtype,  # type: ignore[attr-defined]
        )
