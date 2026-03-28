"""
Generation Head — standard language-model head for next-token prediction.

Architecture: ``Linear(hidden_dim, vocab_size)`` with optional weight
tying to the backbone embedding matrix.

Public API
----------
- ``forward(hidden_states)`` → ``Tensor``  logits ``(batch, seq_len, vocab_size)``
- ``tie_weights(embedding_weight)``  — share weights with backbone
- ``from_config(gen_config, backbone)`` class-method
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class GenerationHead(nn.Module):
    """Standard LM head that projects hidden states to vocabulary logits.

    Parameters
    ----------
    hidden_dim : int
        Input dimensionality (must match backbone ``hidden_dim``).
    vocab_size : int
        Size of the output vocabulary.
    tie_weights_flag : bool
        If *True*, ``tie_weights`` should be called after construction
        so that the projection shares its weight with the backbone
        token embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        vocab_size: int = 50257,
        tie_weights_flag: bool = True,
    ) -> None:
        super().__init__()

        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size
        self._tied = False

        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)

        logger.info(
            "GenerationHead created: %d → %d (tie_weights=%s)",
            hidden_dim,
            vocab_size,
            tie_weights_flag,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def is_tied(self) -> bool:
        """Whether the projection weight is tied to backbone embeddings."""
        return self._tied

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits.

        Parameters
        ----------
        hidden_states : Tensor
            Shape ``(batch, seq_len, hidden_dim)`` or ``(batch, hidden_dim)``.

        Returns
        -------
        Tensor
            Logits with shape ``(..., vocab_size)``.
        """
        if hidden_states.dtype == torch.float16:
            # Upcast to FP32 for the large vocabulary projection.
            # With hidden_dim=2560 and vocab_size=50257, FP16 dot-products
            # can overflow (max ~65504), producing NaN in cross_entropy.
            # Gradients are automatically cast back to FP16 by autograd.
            return torch.nn.functional.linear(
                hidden_states.float(), self.proj.weight.float(),
            )
        return self.proj(hidden_states)

    # ------------------------------------------------------------------
    # Weight tying
    # ------------------------------------------------------------------

    def tie_weights(self, embedding_weight: Tensor) -> None:
        """Share the projection weight with the backbone embedding matrix.

        Parameters
        ----------
        embedding_weight : Tensor
            ``backbone.get_embedding_weight()`` — shape ``(vocab_size, hidden_dim)``.
        """
        self.proj.weight = nn.Parameter(embedding_weight, requires_grad=embedding_weight.requires_grad)
        self._tied = True
        logger.info("GenerationHead weights tied to backbone embedding matrix")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        gen_config: "GenerationHeadConfig",  # type: ignore[name-defined]  # noqa: F821
        backbone: Optional["BioMedLMBackbone"] = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> "GenerationHead":
        """Construct from a :class:`config.config.GenerationHeadConfig` instance.

        Parameters
        ----------
        gen_config : GenerationHeadConfig
            Generation head section of the FRLM config.
        backbone : BioMedLMBackbone, optional
            If provided **and** ``gen_config.tie_weights`` is True the
            head's projection weight is tied to the backbone embeddings.
        """
        head = cls(
            hidden_dim=gen_config.input_dim,
            vocab_size=gen_config.output_dim,
            tie_weights_flag=gen_config.tie_weights,
        )

        if gen_config.tie_weights and backbone is not None:
            head.tie_weights(backbone.get_embedding_weight())

        return head
