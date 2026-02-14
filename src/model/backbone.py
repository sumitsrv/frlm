"""
BioMedLM Backbone — decoder-only transformer wrapper.

Wraps the ``stanford-crfm/BioMedLM`` (GPT-2 XL variant) via
HuggingFace ``transformers`` and exposes a clean interface for
the FRLM composite model.

Public API
----------
- ``forward(input_ids, attention_mask)``
    → ``BackboneOutput(last_hidden_state, all_hidden_states)``
- ``get_hidden_dim()`` → ``int``
- ``get_embedding_weight()`` → ``Tensor``  (for weight tying)
- ``from_config(backbone_config)`` class-method
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class BackboneOutput:
    """Container returned by :meth:`BioMedLMBackbone.forward`.

    Attributes
    ----------
    last_hidden_state : Tensor
        Hidden states from the final transformer layer.
        Shape ``(batch, seq_len, hidden_dim)``.
    all_hidden_states : tuple[Tensor, ...]
        Hidden states from **every** layer (including the embedding
        layer), each of shape ``(batch, seq_len, hidden_dim)``.
        Only populated when ``output_hidden_states=True`` on the
        underlying HuggingFace model (always enabled here).
    """

    last_hidden_state: Tensor
    all_hidden_states: Tuple[Tensor, ...]


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class BioMedLMBackbone(nn.Module):
    """Thin wrapper around HuggingFace ``GPT2LMHeadModel`` for BioMedLM.

    Parameters
    ----------
    model_name : str
        HuggingFace hub identifier or local path.
    hidden_dim : int
        Expected hidden dimensionality (used for sanity check only).
    freeze_backbone : bool
        If *True* all backbone parameters are frozen (``requires_grad=False``).
    gradient_checkpointing : bool
        Enable gradient-checkpointing to trade compute for memory.
    dtype : str
        One of ``"float32"``, ``"float16"``, ``"bfloat16"``.
    """

    # Maps config string → torch dtype
    _DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(
        self,
        model_name: str = "stanford-crfm/BioMedLM",
        hidden_dim: int = 2560,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = True,
        dtype: str = "float32",
    ) -> None:
        super().__init__()

        self._model_name = model_name
        self._hidden_dim = hidden_dim
        self._freeze = freeze_backbone
        self._gradient_checkpointing = gradient_checkpointing
        self._dtype_str = dtype

        torch_dtype = self._DTYPE_MAP.get(dtype, torch.float32)

        # Lazy import so unit tests can mock without heavy deps
        from transformers import GPT2Model  # type: ignore[import-untyped]

        logger.info("Loading backbone model: %s (dtype=%s)", model_name, dtype)
        self.transformer: GPT2Model = GPT2Model.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )

        # Always emit hidden states from every layer
        self.transformer.config.output_hidden_states = True

        # Sanity-check hidden dimension
        actual_dim = self.transformer.config.hidden_size
        if actual_dim != hidden_dim:
            logger.warning(
                "Expected hidden_dim=%d but model reports %d — using model value",
                hidden_dim,
                actual_dim,
            )
            self._hidden_dim = actual_dim

        # Gradient checkpointing
        if gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for backbone")

        # Freeze
        if freeze_backbone:
            self._freeze_parameters()

        n_params = sum(p.numel() for p in self.transformer.parameters())
        n_trainable = sum(
            p.numel() for p in self.transformer.parameters() if p.requires_grad
        )
        logger.info(
            "Backbone loaded: %s — %.1fM params (%.1fM trainable)",
            model_name,
            n_params / 1e6,
            n_trainable / 1e6,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_hidden_dim(self) -> int:
        """Return the hidden dimensionality of the transformer."""
        return self._hidden_dim

    def get_embedding_weight(self) -> Tensor:
        """Return the token-embedding weight matrix for weight tying."""
        return self.transformer.wte.weight  # type: ignore[union-attr]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size reported by the underlying model."""
        return int(self.transformer.config.vocab_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> BackboneOutput:
        """Run the backbone transformer.

        Parameters
        ----------
        input_ids : Tensor
            Integer token ids, shape ``(batch, seq_len)``.
        attention_mask : Tensor, optional
            Binary mask, shape ``(batch, seq_len)``.

        Returns
        -------
        BackboneOutput
            Dataclass with ``last_hidden_state`` and ``all_hidden_states``.
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return BackboneOutput(
            last_hidden_state=outputs.last_hidden_state,
            all_hidden_states=outputs.hidden_states,
        )

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _freeze_parameters(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        logger.info("All backbone parameters frozen")

    def freeze(self) -> None:
        """Public method to freeze backbone weights."""
        self._freeze = True
        self._freeze_parameters()

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        self._freeze = False
        for param in self.transformer.parameters():
            param.requires_grad = True
        logger.info("All backbone parameters unfrozen")

    def freeze_layers(self, n_layers: int) -> None:
        """Freeze the first *n_layers* transformer blocks.

        Useful for gradual unfreezing during fine-tuning.
        """
        blocks = self.transformer.h  # nn.ModuleList of transformer blocks
        for i, block in enumerate(blocks):
            if i < n_layers:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True
        # Always freeze embeddings when partially freezing
        if n_layers > 0:
            for param in self.transformer.wte.parameters():
                param.requires_grad = False
            for param in self.transformer.wpe.parameters():
                param.requires_grad = False
        logger.info("Froze first %d / %d transformer layers", n_layers, len(blocks))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, backbone_config: "BackboneConfig") -> "BioMedLMBackbone":  # type: ignore[name-defined]  # noqa: F821
        """Construct from a :class:`config.config.BackboneConfig` instance.

        Parameters
        ----------
        backbone_config : BackboneConfig
            Backbone section of the FRLM config.
        """
        return cls(
            model_name=backbone_config.name,
            hidden_dim=backbone_config.hidden_dim,
            freeze_backbone=backbone_config.freeze_backbone,
            gradient_checkpointing=backbone_config.gradient_checkpointing,
            dtype=backbone_config.dtype,
        )
