"""
Router Head — binary classifier that decides retrieval vs generation.

Architecture (from the FRLM spec):

    Linear(hidden_dim, 256) → ReLU → Dropout(0.1) → Linear(256, 1)

``forward`` returns raw logits (before sigmoid); ``predict`` applies
sigmoid to produce probabilities in [0, 1].  A probability above
*threshold* (default 0.5) means **retrieval**; below means
**generation**.

Public API
----------
- ``forward(hidden_states)``  → ``Tensor`` (batch, seq_len, 1)
- ``predict(hidden_states)``  → ``Tensor`` (batch, seq_len, 1)
- ``decide(hidden_states, threshold=None)`` → ``BoolTensor``
- ``from_config(router_config)`` class-method
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class RouterHead(nn.Module):
    """Binary classification head: retrieval (1) vs generation (0).

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of backbone hidden states (input to the router).
    intermediate_dim : int
        Width of the hidden layer (default 256).
    dropout : float
        Dropout probability between the two linear layers.
    threshold : float
        Default decision boundary for :meth:`decide`.
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        intermediate_dim: int = 256,
        dropout: float = 0.1,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()

        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._threshold = threshold

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
        )

        logger.info(
            "RouterHead created: %d → %d → 1 (dropout=%.2f, threshold=%.2f)",
            hidden_dim,
            intermediate_dim,
            dropout,
            threshold,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Current decision threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {value}")
        self._threshold = value

    # ------------------------------------------------------------------
    # Forward / predict / decide
    # ------------------------------------------------------------------

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Produce raw logits (before sigmoid).

        Parameters
        ----------
        hidden_states : Tensor
            Shape ``(batch, seq_len, hidden_dim)`` or ``(batch, hidden_dim)``.

        Returns
        -------
        Tensor
            Logits, same leading dimensions with final dim = 1.
        """
        return self.net(hidden_states)

    def predict(self, hidden_states: Tensor) -> Tensor:
        """Produce probabilities via sigmoid.

        Returns
        -------
        Tensor
            Values in [0, 1], same shape as ``forward`` output.
        """
        return torch.sigmoid(self.forward(hidden_states))

    def decide(
        self,
        hidden_states: Tensor,
        threshold: Optional[float] = None,
    ) -> Tensor:
        """Return a boolean mask: *True* where router selects retrieval.

        Parameters
        ----------
        hidden_states : Tensor
            Backbone hidden states.
        threshold : float, optional
            Override the default threshold for this call.

        Returns
        -------
        BoolTensor
            Shape ``(batch, seq_len)`` — True = retrieval.
        """
        t = threshold if threshold is not None else self._threshold
        probs = self.predict(hidden_states)
        return (probs.squeeze(-1) > t)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, router_config: "RouterHeadConfig") -> "RouterHead":  # type: ignore[name-defined]  # noqa: F821
        """Construct from a :class:`config.config.RouterHeadConfig` instance."""
        return cls(
            hidden_dim=router_config.input_dim,
            intermediate_dim=router_config.hidden_dim,
            dropout=router_config.dropout,
            threshold=router_config.threshold,
        )
