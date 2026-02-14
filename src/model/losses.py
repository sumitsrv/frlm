"""
FRLM Loss Functions.

Implements the three individual losses and the combined objective:

    L_total = λ_router · L_router + λ_retrieval · L_retrieval + λ_generation · L_generation

Individual losses
-----------------
- :class:`InfoNCELoss` — contrastive loss with temperature & hard negatives
- :class:`RouterLoss`  — binary cross-entropy (optional class weighting)
- :class:`GenerationLoss` — cross-entropy (optional label smoothing)

Combined
--------
- :class:`FRLMCombinedLoss` — applies the three losses with a ``router_mask``
  that gates which positions contribute to retrieval vs generation loss.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ===========================================================================
# InfoNCE (contrastive) loss
# ===========================================================================


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with temperature scaling and hard negatives.

    For a single query the loss is:

        -log( exp(sim(q, p) / τ) / Σ_i exp(sim(q, n_i) / τ) )

    where *p* is the positive and the sum runs over the positive **and**
    all negatives.

    Parameters
    ----------
    temperature : float
        Scaling temperature τ (default 0.07).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        query_emb: Tensor,
        positive_emb: Tensor,
        negative_embs: Tensor,
        temperature: Optional[float] = None,
    ) -> Tensor:
        """Compute InfoNCE loss.

        Parameters
        ----------
        query_emb : Tensor
            Query embeddings, shape ``(batch, dim)``.
        positive_emb : Tensor
            Positive fact embeddings, shape ``(batch, dim)``.
        negative_embs : Tensor
            Negative fact embeddings, shape ``(batch, num_neg, dim)``.
        temperature : float, optional
            Override instance temperature for this call.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        tau = temperature if temperature is not None else self.temperature

        # Similarities: query · positive  → (batch,)
        pos_sim = (query_emb * positive_emb).sum(dim=-1) / tau  # (batch,)

        # query · negatives  → (batch, num_neg)
        neg_sim = torch.bmm(
            negative_embs,
            query_emb.unsqueeze(-1),
        ).squeeze(-1) / tau  # (batch, num_neg)

        # Logits: positive is index 0
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch, 1+num_neg)

        # Labels: positive is always index 0
        labels = torch.zeros(
            logits.size(0), dtype=torch.long, device=logits.device
        )

        return F.cross_entropy(logits, labels)


# ===========================================================================
# Router loss
# ===========================================================================


class RouterLoss(nn.Module):
    """Binary cross-entropy loss for the router head.

    Parameters
    ----------
    pos_weight : float, optional
        Weight for the positive (retrieval) class.  ``None`` means equal
        weighting.
    label_smoothing : float
        Amount of label smoothing (default 0.0).
    """

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self._label_smoothing = label_smoothing
        self._pos_weight_value = pos_weight

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute masked BCE loss.

        Parameters
        ----------
        logits : Tensor
            Raw router logits, shape ``(batch, seq_len)`` or ``(batch, seq_len, 1)``.
        labels : Tensor
            Binary ground-truth, same shape as *logits* (after squeeze).
        mask : Tensor, optional
            Boolean or float mask — positions with 0 are ignored.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        logits = logits.squeeze(-1)
        labels = labels.float().squeeze(-1)

        # Label smoothing
        if self._label_smoothing > 0:
            labels = labels * (1 - self._label_smoothing) + 0.5 * self._label_smoothing

        # Pos weight
        pw = None
        if self._pos_weight_value is not None:
            pw = torch.tensor(
                [self._pos_weight_value],
                device=logits.device,
                dtype=logits.dtype,
            )

        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pw, reduction="none"
        )

        if mask is not None:
            mask = mask.float().squeeze(-1)
            loss = loss * mask
            denom = mask.sum().clamp(min=1.0)
            return loss.sum() / denom

        return loss.mean()


# ===========================================================================
# Generation loss
# ===========================================================================


class GenerationLoss(nn.Module):
    """Cross-entropy loss for next-token prediction.

    Parameters
    ----------
    label_smoothing : float
        Amount of label smoothing (default 0.0).
    ignore_index : int
        Token id to ignore in the loss (typically padding, default -100).
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self._label_smoothing = label_smoothing
        self._ignore_index = ignore_index

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute masked cross-entropy loss.

        Parameters
        ----------
        logits : Tensor
            Shape ``(batch, seq_len, vocab_size)``.
        labels : Tensor
            Shape ``(batch, seq_len)`` with token ids.
        mask : Tensor, optional
            Boolean mask — positions with 0/False are set to ``ignore_index``.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        # Apply mask by overwriting labels
        if mask is not None:
            labels = labels.clone()
            labels[~mask.bool()] = self._ignore_index

        # Reshape for cross_entropy: (batch*seq_len, vocab) vs (batch*seq_len,)
        flat_labels = labels.reshape(-1)

        # If every position is ignore_index, return zero to avoid NaN
        if (flat_labels == self._ignore_index).all():
            return logits.new_tensor(0.0)

        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            flat_labels,
            ignore_index=self._ignore_index,
            label_smoothing=self._label_smoothing,
        )
        return loss


# ===========================================================================
# Combined loss
# ===========================================================================


class FRLMCombinedLoss(nn.Module):
    """Combines the three FRLM losses with configurable weights.

    .. math::

        L_{total} = \\lambda_r L_{router} + \\lambda_{ret} L_{retrieval}
                    + \\lambda_{gen} L_{generation}

    Parameters
    ----------
    router_weight : float
        λ_router (default 1.0).
    retrieval_weight : float
        λ_retrieval (default 2.0).
    generation_weight : float
        λ_generation (default 1.0).
    contrastive_temperature : float
        Temperature for InfoNCE (default 0.07).
    router_pos_weight : float, optional
        Positive-class weight for router BCE loss.
    router_label_smoothing : float
        Label smoothing for router (default 0.0).
    generation_label_smoothing : float
        Label smoothing for generation CE (default 0.0).
    """

    def __init__(
        self,
        router_weight: float = 1.0,
        retrieval_weight: float = 2.0,
        generation_weight: float = 1.0,
        contrastive_temperature: float = 0.07,
        router_pos_weight: Optional[float] = None,
        router_label_smoothing: float = 0.0,
        generation_label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        self.lambda_router = router_weight
        self.lambda_retrieval = retrieval_weight
        self.lambda_generation = generation_weight

        self.router_loss_fn = RouterLoss(
            pos_weight=router_pos_weight,
            label_smoothing=router_label_smoothing,
        )
        self.retrieval_loss_fn = InfoNCELoss(temperature=contrastive_temperature)
        self.generation_loss_fn = GenerationLoss(
            label_smoothing=generation_label_smoothing,
        )

        logger.info(
            "FRLMCombinedLoss created: λ_router=%.2f, λ_retrieval=%.2f, "
            "λ_generation=%.2f, τ=%.3f",
            router_weight,
            retrieval_weight,
            generation_weight,
            contrastive_temperature,
        )

    def forward(
        self,
        router_logits: Tensor,
        router_labels: Tensor,
        query_emb: Optional[Tensor] = None,
        positive_emb: Optional[Tensor] = None,
        negative_embs: Optional[Tensor] = None,
        gen_logits: Optional[Tensor] = None,
        gen_labels: Optional[Tensor] = None,
        router_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """Compute the combined FRLM loss.

        Parameters
        ----------
        router_logits : Tensor
            Raw router logits ``(batch, seq_len)`` or ``(batch, seq_len, 1)``.
        router_labels : Tensor
            Binary ground-truth ``(batch, seq_len)``.
        query_emb : Tensor, optional
            Query embeddings from retrieval head ``(N_ret, dim)``.
        positive_emb : Tensor, optional
            Positive fact embeddings ``(N_ret, dim)``.
        negative_embs : Tensor, optional
            Negative embeddings ``(N_ret, num_neg, dim)``.
        gen_logits : Tensor, optional
            Generation logits ``(batch, seq_len, vocab_size)``.
        gen_labels : Tensor, optional
            Token labels ``(batch, seq_len)``.
        router_mask : Tensor, optional
            ``1`` = retrieval position, ``0`` = generation position.
            Used to gate which positions contribute to generation loss.
        attention_mask : Tensor, optional
            Padding mask ``(batch, seq_len)`` — loss computed only where 1.

        Returns
        -------
        total_loss : Tensor
            Scalar weighted sum of all active losses.
        loss_dict : dict[str, Tensor]
            Individual loss components for logging.
        """
        loss_dict: Dict[str, Tensor] = {}
        device = router_logits.device
        total = torch.tensor(0.0, device=device)

        # ----- Router loss (always computed) ----
        router_loss_mask = attention_mask
        l_router = self.router_loss_fn(router_logits, router_labels, mask=router_loss_mask)
        loss_dict["router_loss"] = l_router
        total = total + self.lambda_router * l_router

        # ----- Retrieval loss (if embeddings are provided) ----
        if query_emb is not None and positive_emb is not None and negative_embs is not None:
            if query_emb.size(0) > 0:
                l_retrieval = self.retrieval_loss_fn(query_emb, positive_emb, negative_embs)
            else:
                l_retrieval = torch.tensor(0.0, device=device)
            loss_dict["retrieval_loss"] = l_retrieval
            total = total + self.lambda_retrieval * l_retrieval

        # ----- Generation loss (if logits + labels are provided) ----
        if gen_logits is not None and gen_labels is not None:
            # Generation only on non-retrieval positions
            gen_mask = None
            if router_mask is not None:
                # Invert: generation positions are where router_mask == 0
                gen_mask = (1 - router_mask.float())
                if attention_mask is not None:
                    gen_mask = gen_mask * attention_mask.float()
            elif attention_mask is not None:
                gen_mask = attention_mask.float()

            l_generation = self.generation_loss_fn(gen_logits, gen_labels, mask=gen_mask)
            loss_dict["generation_loss"] = l_generation
            total = total + self.lambda_generation * l_generation

        loss_dict["total_loss"] = total
        return total, loss_dict

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, loss_config: "LossConfig", training_config: Optional[Any] = None) -> "FRLMCombinedLoss":  # type: ignore[name-defined]  # noqa: F821
        """Construct from :class:`config.config.LossConfig`.

        Parameters
        ----------
        loss_config : LossConfig
            Loss section of the FRLM config.
        training_config : TrainingConfig, optional
            Training section — used for router ``pos_weight`` and label
            smoothing if available.
        """
        router_pos_weight = None
        router_ls = 0.0
        gen_ls = 0.0

        if training_config is not None:
            router_pos_weight = getattr(training_config.router, "pos_weight", None)
            router_ls = getattr(training_config.router, "label_smoothing", 0.0)

        return cls(
            router_weight=loss_config.router_weight,
            retrieval_weight=loss_config.retrieval_weight,
            generation_weight=loss_config.generation_weight,
            contrastive_temperature=loss_config.contrastive_temperature,
            router_pos_weight=router_pos_weight,
            router_label_smoothing=router_ls,
            generation_label_smoothing=gen_ls,
        )
