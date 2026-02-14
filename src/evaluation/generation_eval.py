"""
Generation Evaluator — metrics for the FRLM generation head.

Computes:
- Perplexity on non-factual (generation) spans only
- Full-sequence perplexity for baseline comparison
- Cross-entropy loss decomposition
- Baseline comparison with the un-augmented BioMedLM backbone

Public API
----------
- ``evaluate(model, dataloader, config)`` → ``GenerationResults``
- ``compute_perplexity(model, dataloader)`` → ``float``
- ``compare_with_baseline(frlm_model, baseline_model, dataloader)``
  → ``BaselineComparison``
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ===========================================================================
# Result containers
# ===========================================================================


@dataclass
class PerplexityResult:
    """Perplexity broken down by span type."""

    overall: float = 0.0
    generation_spans: float = 0.0
    retrieval_spans: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "overall_perplexity": self.overall,
            "generation_span_perplexity": self.generation_spans,
            "retrieval_span_perplexity": self.retrieval_spans,
        }


@dataclass
class BaselineComparison:
    """Comparison between FRLM and baseline BioMedLM."""

    frlm_perplexity: float = 0.0
    baseline_perplexity: float = 0.0
    perplexity_reduction: float = 0.0
    frlm_loss: float = 0.0
    baseline_loss: float = 0.0
    loss_reduction: float = 0.0

    @property
    def improvement_pct(self) -> float:
        """Perplexity improvement as a percentage."""
        if self.baseline_perplexity == 0:
            return 0.0
        return (
            (self.baseline_perplexity - self.frlm_perplexity)
            / self.baseline_perplexity
            * 100.0
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "frlm_perplexity": self.frlm_perplexity,
            "baseline_perplexity": self.baseline_perplexity,
            "perplexity_reduction": self.perplexity_reduction,
            "improvement_pct": round(self.improvement_pct, 2),
            "frlm_loss": self.frlm_loss,
            "baseline_loss": self.baseline_loss,
            "loss_reduction": self.loss_reduction,
        }


@dataclass
class GenerationResults:
    """Complete generation evaluation results."""

    perplexity: PerplexityResult = field(default_factory=PerplexityResult)
    cross_entropy_loss: float = 0.0
    num_tokens: int = 0
    num_generation_tokens: int = 0
    num_retrieval_tokens: int = 0
    baseline_comparison: Optional[BaselineComparison] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        result.update(self.perplexity.to_dict())
        result["cross_entropy_loss"] = self.cross_entropy_loss
        result["num_tokens"] = self.num_tokens
        result["num_generation_tokens"] = self.num_generation_tokens
        result["num_retrieval_tokens"] = self.num_retrieval_tokens
        if self.baseline_comparison is not None:
            result["baseline_comparison"] = self.baseline_comparison.to_dict()
        return result


# ===========================================================================
# Core metric functions
# ===========================================================================


def compute_perplexity(
    total_loss: float,
    num_tokens: int,
) -> float:
    """Compute perplexity from total cross-entropy loss and token count.

    Parameters
    ----------
    total_loss : float
        Sum of per-token cross-entropy losses.
    num_tokens : int
        Number of tokens (non-padding).

    Returns
    -------
    float
        Perplexity = exp(avg_loss). Returns inf if num_tokens == 0.
    """
    if num_tokens == 0:
        return float("inf")
    avg_loss = total_loss / num_tokens
    try:
        return math.exp(avg_loss)
    except OverflowError:
        return float("inf")


def compute_token_level_loss(
    gen_logits: Tensor,
    labels: Tensor,
    mask: Optional[Tensor] = None,
    ignore_index: int = -100,
) -> tuple[float, int]:
    """Compute sum of CE losses and number of valid tokens.

    Parameters
    ----------
    gen_logits : Tensor
        Shape ``(batch, seq_len, vocab_size)``.
    labels : Tensor
        Shape ``(batch, seq_len)``.
    mask : Tensor, optional
        Boolean mask ``(batch, seq_len)`` — True = include this token.
    ignore_index : int
        Token ID to ignore.

    Returns
    -------
    tuple of (float, int)
        (sum_of_losses, count_of_valid_tokens)
    """
    # Shift for causal LM: predict next token
    shift_logits = gen_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Per-token loss
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    )
    loss = loss.view(shift_labels.shape)

    # Valid token mask
    valid = shift_labels != ignore_index
    if mask is not None:
        # Mask is (batch, seq_len) — shift to match
        shifted_mask = mask[:, 1:].contiguous()
        valid = valid & shifted_mask

    total_loss = float(loss[valid].sum().item())
    num_tokens = int(valid.sum().item())

    return total_loss, num_tokens


# ===========================================================================
# Evaluator class
# ===========================================================================


class GenerationEvaluator:
    """Full generation evaluation pipeline.

    Parameters
    ----------
    max_length : int
        Maximum sequence length for evaluation.
    temperature : float
        Sampling temperature (only used in qualitative eval).
    device : str
        Device for model inference.
    """

    def __init__(
        self,
        max_length: int = 256,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.max_length = max_length
        self.temperature = temperature
        self.device = device

    @classmethod
    def from_config(cls, eval_config: Any) -> "GenerationEvaluator":
        """Create from a :class:`GenerationEvalConfig`."""
        return cls(
            max_length=eval_config.max_length,
            temperature=eval_config.temperature,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: Any,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> GenerationResults:
        """Run generation evaluation over the dataloader.

        Each batch should yield dicts with keys:
            - ``input_ids``: ``(batch, seq_len)``
            - ``attention_mask``: ``(batch, seq_len)``
            - ``labels``: ``(batch, seq_len)`` with -100 for padding
            - ``router_labels``: ``(batch, seq_len)`` binary — 1=retrieval, 0=generation (optional)

        Parameters
        ----------
        model : FRLMModel
            FRLM model in eval mode.
        dataloader : DataLoader
            Test set.
        max_samples : int, optional
            Cap on number of samples.

        Returns
        -------
        GenerationResults
        """
        model.eval()
        model.to(self.device)

        total_loss_all = 0.0
        total_tokens_all = 0
        total_loss_gen = 0.0
        total_tokens_gen = 0
        total_loss_ret = 0.0
        total_tokens_ret = 0
        sample_count = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            router_labels = batch.get("router_labels")

            bsz = input_ids.size(0)
            if max_samples and sample_count + bsz > max_samples:
                # Trim batch
                remaining = max_samples - sample_count
                input_ids = input_ids[:remaining]
                attention_mask = attention_mask[:remaining]
                labels = labels[:remaining]
                if router_labels is not None:
                    router_labels = router_labels[:remaining]

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            gen_logits = output.gen_logits

            if gen_logits is None:
                sample_count += input_ids.size(0)
                continue

            # Overall loss
            loss_all, tok_all = compute_token_level_loss(gen_logits, labels)
            total_loss_all += loss_all
            total_tokens_all += tok_all

            # Split by router mask
            if router_labels is not None:
                router_mask = router_labels.to(self.device).bool()

                # Generation spans (router_labels == 0)
                gen_mask = ~router_mask
                loss_gen, tok_gen = compute_token_level_loss(
                    gen_logits, labels, mask=gen_mask
                )
                total_loss_gen += loss_gen
                total_tokens_gen += tok_gen

                # Retrieval spans (router_labels == 1)
                loss_ret, tok_ret = compute_token_level_loss(
                    gen_logits, labels, mask=router_mask
                )
                total_loss_ret += loss_ret
                total_tokens_ret += tok_ret

            sample_count += input_ids.size(0)
            if max_samples and sample_count >= max_samples:
                break

        # Compute perplexities
        ppl = PerplexityResult(
            overall=compute_perplexity(total_loss_all, total_tokens_all),
            generation_spans=compute_perplexity(total_loss_gen, total_tokens_gen),
            retrieval_spans=compute_perplexity(total_loss_ret, total_tokens_ret),
        )

        avg_loss = total_loss_all / total_tokens_all if total_tokens_all > 0 else 0.0

        results = GenerationResults(
            perplexity=ppl,
            cross_entropy_loss=avg_loss,
            num_tokens=total_tokens_all,
            num_generation_tokens=total_tokens_gen,
            num_retrieval_tokens=total_tokens_ret,
        )

        logger.info(
            "Generation evaluation complete: %d tokens, PPL=%.2f, "
            "gen_span_PPL=%.2f, ret_span_PPL=%.2f",
            results.num_tokens,
            ppl.overall,
            ppl.generation_spans,
            ppl.retrieval_spans,
        )
        return results

    @torch.no_grad()
    def compare_with_baseline(
        self,
        frlm_model: Any,
        baseline_model: Any,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> BaselineComparison:
        """Compare FRLM against baseline BioMedLM on generation quality.

        Parameters
        ----------
        frlm_model : FRLMModel
            FRLM model.
        baseline_model : BioMedLMBackbone or nn.Module
            Baseline model (backbone + generation head only).
        dataloader : DataLoader
            Shared test set.
        max_samples : int, optional
            Cap on sample count.

        Returns
        -------
        BaselineComparison
        """
        # Evaluate FRLM
        frlm_results = self.evaluate(frlm_model, dataloader, max_samples)

        # Evaluate baseline
        baseline_model.eval()
        baseline_model.to(self.device)
        baseline_total_loss = 0.0
        baseline_total_tokens = 0
        sample_count = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            bsz = input_ids.size(0)
            if max_samples and sample_count + bsz > max_samples:
                remaining = max_samples - sample_count
                input_ids = input_ids[:remaining]
                labels = labels[:remaining]

            # Baseline forward: just backbone + generation head
            baseline_output = baseline_model(input_ids=input_ids)
            if hasattr(baseline_output, "gen_logits") and baseline_output.gen_logits is not None:
                logits = baseline_output.gen_logits
            elif hasattr(baseline_output, "last_hidden_state"):
                logits = baseline_output.last_hidden_state
            else:
                logits = baseline_output

            loss_val, tok_count = compute_token_level_loss(logits, labels)
            baseline_total_loss += loss_val
            baseline_total_tokens += tok_count

            sample_count += input_ids.size(0)
            if max_samples and sample_count >= max_samples:
                break

        baseline_ppl = compute_perplexity(baseline_total_loss, baseline_total_tokens)
        baseline_avg_loss = (
            baseline_total_loss / baseline_total_tokens
            if baseline_total_tokens > 0
            else 0.0
        )

        comparison = BaselineComparison(
            frlm_perplexity=frlm_results.perplexity.overall,
            baseline_perplexity=baseline_ppl,
            perplexity_reduction=baseline_ppl - frlm_results.perplexity.overall,
            frlm_loss=frlm_results.cross_entropy_loss,
            baseline_loss=baseline_avg_loss,
            loss_reduction=baseline_avg_loss - frlm_results.cross_entropy_loss,
        )

        logger.info(
            "Baseline comparison: FRLM PPL=%.2f, Baseline PPL=%.2f, "
            "Improvement=%.1f%%",
            comparison.frlm_perplexity,
            comparison.baseline_perplexity,
            comparison.improvement_pct,
        )
        return comparison

    def evaluate_from_losses(
        self,
        token_losses: List[float],
        router_mask: Optional[List[bool]] = None,
    ) -> GenerationResults:
        """Evaluate from pre-computed per-token losses.

        Parameters
        ----------
        token_losses : list of float
            Per-token cross-entropy losses.
        router_mask : list of bool, optional
            True = retrieval token, False = generation token.

        Returns
        -------
        GenerationResults
        """
        if not token_losses:
            return GenerationResults()

        import numpy as np

        losses = np.array(token_losses)
        total_loss = float(losses.sum())
        num_tokens = len(losses)

        gen_loss = 0.0
        gen_tokens = 0
        ret_loss = 0.0
        ret_tokens = 0

        if router_mask is not None:
            mask = np.array(router_mask)
            gen_mask = ~mask
            gen_loss = float(losses[gen_mask].sum())
            gen_tokens = int(gen_mask.sum())
            ret_loss = float(losses[mask].sum())
            ret_tokens = int(mask.sum())

        ppl = PerplexityResult(
            overall=compute_perplexity(total_loss, num_tokens),
            generation_spans=compute_perplexity(gen_loss, gen_tokens),
            retrieval_spans=compute_perplexity(ret_loss, ret_tokens),
        )

        return GenerationResults(
            perplexity=ppl,
            cross_entropy_loss=total_loss / num_tokens if num_tokens else 0.0,
            num_tokens=num_tokens,
            num_generation_tokens=gen_tokens,
            num_retrieval_tokens=ret_tokens,
        )
