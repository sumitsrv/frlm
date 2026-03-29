"""
Retrieval Evaluator — metrics for the FRLM retrieval head.

Computes:
- Precision@k for configurable k values (1, 5, 10, 20)
- Mean Reciprocal Rank (MRR)
- Temporal accuracy (per-mode and overall)
- Granularity accuracy (per-level and overall)
- Stratified evaluation by relation type, entity frequency,
  and temporal complexity

Public API
----------
- ``evaluate(model, dataloader, faiss_index, kg_client, config)``
  → ``RetrievalResults``
- ``precision_at_k(predictions, ground_truth, k)`` → ``float``
- ``mean_reciprocal_rank(predictions, ground_truth)`` → ``float``
- ``temporal_accuracy(predicted_modes, true_modes)`` → ``dict``
- ``granularity_accuracy(predicted_levels, true_levels)`` → ``dict``
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.model.retrieval_head import (
    GRANULARITY_NAMES,
    TEMPORAL_MODE_NAMES,
    QuerySignature,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Result containers
# ===========================================================================


@dataclass
class PrecisionAtKResult:
    """Holds P@k values for multiple k."""

    values: Dict[int, float] = field(default_factory=dict)

    def __getitem__(self, k: int) -> float:
        return self.values[k]

    def to_dict(self) -> Dict[str, float]:
        return {f"P@{k}": v for k, v in sorted(self.values.items())}


@dataclass
class TemporalAccuracyResult:
    """Per-mode and overall temporal accuracy."""

    overall: float = 0.0
    per_mode: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"overall": self.overall, "per_mode": dict(self.per_mode)}


@dataclass
class GranularityAccuracyResult:
    """Per-level and overall granularity accuracy."""

    overall: float = 0.0
    per_level: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"overall": self.overall, "per_level": dict(self.per_level)}


@dataclass
class StratifiedResult:
    """Stratified evaluation breakdowns."""

    by_relation_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_entity_frequency: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_temporal_complexity: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "by_relation_type": dict(self.by_relation_type),
            "by_entity_frequency": dict(self.by_entity_frequency),
            "by_temporal_complexity": dict(self.by_temporal_complexity),
        }


@dataclass
class RetrievalResults:
    """Complete retrieval evaluation results."""

    precision_at_k: PrecisionAtKResult = field(default_factory=PrecisionAtKResult)
    mrr: float = 0.0
    temporal: TemporalAccuracyResult = field(default_factory=TemporalAccuracyResult)
    granularity: GranularityAccuracyResult = field(
        default_factory=GranularityAccuracyResult
    )
    stratified: StratifiedResult = field(default_factory=StratifiedResult)
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        result.update(self.precision_at_k.to_dict())
        result["MRR"] = self.mrr
        result["temporal"] = self.temporal.to_dict()
        result["granularity"] = self.granularity.to_dict()
        result["stratified"] = self.stratified.to_dict()
        result["num_samples"] = self.num_samples
        return result


# ===========================================================================
# Core metric functions
# ===========================================================================


def precision_at_k(
    predicted_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
    k: int,
) -> float:
    """Compute Precision@k.

    Parameters
    ----------
    predicted_ids : sequence of str
        Ordered predicted fact IDs (most relevant first).
    ground_truth_ids : sequence of str
        Set of relevant fact IDs.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        Fraction of top-k predictions that are relevant.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not ground_truth_ids:
        return 0.0

    gt_set = set(ground_truth_ids)
    top_k = predicted_ids[:k]
    hits = sum(1 for pid in top_k if pid in gt_set)
    return hits / k


def mean_reciprocal_rank(
    predicted_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a single query.

    Parameters
    ----------
    predicted_ids : sequence of str
        Ordered predicted fact IDs.
    ground_truth_ids : sequence of str
        Relevant fact IDs.

    Returns
    -------
    float
        1 / rank_of_first_relevant_result, or 0 if none found.
    """
    if not ground_truth_ids:
        return 0.0

    gt_set = set(ground_truth_ids)
    for rank, pid in enumerate(predicted_ids, start=1):
        if pid in gt_set:
            return 1.0 / rank
    return 0.0


def temporal_accuracy(
    predicted_modes: Sequence[int],
    true_modes: Sequence[int],
) -> TemporalAccuracyResult:
    """Compute temporal mode classification accuracy.

    Parameters
    ----------
    predicted_modes : sequence of int
        Predicted temporal mode indices (0=CURRENT, 1=AT_TIMESTAMP, 2=HISTORY).
    true_modes : sequence of int
        Ground-truth temporal mode indices.

    Returns
    -------
    TemporalAccuracyResult
        Overall and per-mode accuracy.
    """
    if len(predicted_modes) != len(true_modes):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted_modes)}, "
            f"true={len(true_modes)}"
        )
    if not predicted_modes:
        return TemporalAccuracyResult()

    pred_arr = np.array(predicted_modes)
    true_arr = np.array(true_modes)

    overall = float(np.mean(pred_arr == true_arr))

    per_mode: Dict[str, float] = {}
    for mode_idx, mode_name in enumerate(TEMPORAL_MODE_NAMES):
        mask = true_arr == mode_idx
        if mask.sum() > 0:
            per_mode[mode_name] = float(np.mean(pred_arr[mask] == true_arr[mask]))
        else:
            per_mode[mode_name] = 0.0

    return TemporalAccuracyResult(overall=overall, per_mode=per_mode)


def granularity_accuracy(
    predicted_levels: Sequence[int],
    true_levels: Sequence[int],
) -> GranularityAccuracyResult:
    """Compute granularity level classification accuracy.

    Parameters
    ----------
    predicted_levels : sequence of int
        Predicted granularity level indices (0=atomic, ..., 3=cluster).
    true_levels : sequence of int
        Ground-truth granularity level indices.

    Returns
    -------
    GranularityAccuracyResult
        Overall and per-level accuracy.
    """
    if len(predicted_levels) != len(true_levels):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted_levels)}, "
            f"true={len(true_levels)}"
        )
    if not predicted_levels:
        return GranularityAccuracyResult()

    pred_arr = np.array(predicted_levels)
    true_arr = np.array(true_levels)

    overall = float(np.mean(pred_arr == true_arr))

    per_level: Dict[str, float] = {}
    for level_idx, level_name in enumerate(GRANULARITY_NAMES):
        mask = true_arr == level_idx
        if mask.sum() > 0:
            per_level[level_name] = float(np.mean(pred_arr[mask] == true_arr[mask]))
        else:
            per_level[level_name] = 0.0

    return GranularityAccuracyResult(overall=overall, per_level=per_level)


# ===========================================================================
# Batch-level metric accumulators
# ===========================================================================


class _MetricAccumulator:
    """Accumulates P@k, MRR, temporal, and granularity metrics over batches."""

    def __init__(self, k_values: List[int]) -> None:
        self.k_values = k_values
        self._pk_totals: Dict[int, float] = {k: 0.0 for k in k_values}
        self._mrr_total: float = 0.0
        self._temporal_preds: List[int] = []
        self._temporal_true: List[int] = []
        self._granularity_preds: List[int] = []
        self._granularity_true: List[int] = []
        self._count: int = 0

        # Stratified accumulators
        self._by_relation: Dict[str, List[Tuple[List[str], List[str]]]] = defaultdict(
            list
        )
        self._by_entity_freq: Dict[str, List[Tuple[List[str], List[str]]]] = (
            defaultdict(list)
        )
        self._by_temporal_complexity: Dict[
            str, List[Tuple[List[str], List[str]]]
        ] = defaultdict(list)

    def update(
        self,
        predicted_ids: List[str],
        ground_truth_ids: List[str],
        predicted_temporal: Optional[int] = None,
        true_temporal: Optional[int] = None,
        predicted_granularity: Optional[int] = None,
        true_granularity: Optional[int] = None,
        relation_type: Optional[str] = None,
        entity_frequency_bin: Optional[str] = None,
        temporal_complexity: Optional[str] = None,
    ) -> None:
        """Update accumulators with a single query result."""
        self._count += 1

        for k in self.k_values:
            self._pk_totals[k] += precision_at_k(predicted_ids, ground_truth_ids, k)
        self._mrr_total += mean_reciprocal_rank(predicted_ids, ground_truth_ids)

        if predicted_temporal is not None and true_temporal is not None:
            self._temporal_preds.append(predicted_temporal)
            self._temporal_true.append(true_temporal)

        if predicted_granularity is not None and true_granularity is not None:
            self._granularity_preds.append(predicted_granularity)
            self._granularity_true.append(true_granularity)

        # Stratified
        pair = (predicted_ids, ground_truth_ids)
        if relation_type:
            self._by_relation[relation_type].append(pair)
        if entity_frequency_bin:
            self._by_entity_freq[entity_frequency_bin].append(pair)
        if temporal_complexity:
            self._by_temporal_complexity[temporal_complexity].append(pair)

    def compute(self) -> RetrievalResults:
        """Compute final aggregated metrics."""
        if self._count == 0:
            return RetrievalResults()

        # P@k
        pk = PrecisionAtKResult(
            values={k: v / self._count for k, v in self._pk_totals.items()}
        )

        # MRR
        mrr = self._mrr_total / self._count

        # Temporal
        temp = temporal_accuracy(self._temporal_preds, self._temporal_true)

        # Granularity
        gran = granularity_accuracy(self._granularity_preds, self._granularity_true)

        # Stratified
        stratified = self._compute_stratified()

        return RetrievalResults(
            precision_at_k=pk,
            mrr=mrr,
            temporal=temp,
            granularity=gran,
            stratified=stratified,
            num_samples=self._count,
        )

    def _compute_stratified(self) -> StratifiedResult:
        """Compute stratified breakdowns using a default k=1."""
        result = StratifiedResult()

        for category, buckets, target in [
            ("relation_type", self._by_relation, result.by_relation_type),
            ("entity_freq", self._by_entity_freq, result.by_entity_frequency),
            (
                "temporal_complexity",
                self._by_temporal_complexity,
                result.by_temporal_complexity,
            ),
        ]:
            for bucket_name, pairs in buckets.items():
                pk1_sum = sum(
                    precision_at_k(p, g, 1) for p, g in pairs
                )
                mrr_sum = sum(
                    mean_reciprocal_rank(p, g) for p, g in pairs
                )
                n = len(pairs)
                target[bucket_name] = {
                    "P@1": pk1_sum / n if n else 0.0,
                    "MRR": mrr_sum / n if n else 0.0,
                    "count": n,
                }

        return result


# ===========================================================================
# Evaluator class
# ===========================================================================


class RetrievalEvaluator:
    """Full retrieval evaluation pipeline.

    Parameters
    ----------
    k_values : list of int
        k values for Precision@k.
    compute_temporal : bool
        Whether to evaluate temporal mode accuracy.
    compute_granularity : bool
        Whether to evaluate granularity level accuracy.
    device : str
        Device for model inference.
    """

    def __init__(
        self,
        k_values: Optional[List[int]] = None,
        compute_temporal: bool = True,
        compute_granularity: bool = True,
        device: str = "cpu",
    ) -> None:
        self.k_values = k_values or [1, 5, 10, 20]
        self.compute_temporal = compute_temporal
        self.compute_granularity = compute_granularity
        self.device = device

    @classmethod
    def from_config(cls, eval_config: Any) -> "RetrievalEvaluator":
        """Create from a :class:`RetrievalEvalConfig`."""
        return cls(
            k_values=list(eval_config.k_values),
            compute_temporal=eval_config.temporal_accuracy,
            compute_granularity=eval_config.granularity_accuracy,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: Any,
        dataloader: DataLoader,
        faiss_index: Any,
        kg_client: Any,
        max_samples: Optional[int] = None,
    ) -> RetrievalResults:
        """Run retrieval evaluation over the dataloader.

        Each batch is expected to yield dicts with keys:
            - ``input_ids``: token IDs ``(batch, seq_len)``
            - ``attention_mask``: ``(batch, seq_len)``
            - ``ground_truth_fact_ids``: list of lists of fact IDs
            - ``ground_truth_temporal``: list of int (optional)
            - ``ground_truth_granularity``: list of int (optional)
            - ``relation_type``: list of str (optional, for stratification)
            - ``entity_frequency_bin``: list of str (optional)
            - ``temporal_complexity``: list of str (optional)

        Parameters
        ----------
        model : FRLMModel
            The FRLM model in eval mode.
        dataloader : DataLoader
            Test set dataloader.
        faiss_index : FAISSFactIndex or HierarchicalIndex
            Fact index for retrieval.
        kg_client : Neo4jClient
            KG client for temporal filtering.
        max_samples : int, optional
            Stop after this many samples.

        Returns
        -------
        RetrievalResults
            Complete evaluation results.
        """
        model.eval()
        model.to(self.device)
        accumulator = _MetricAccumulator(self.k_values)
        sample_count = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # ground_truth_fact_ids is required for retrieval evaluation.
            # Datasets that lack this field (e.g. RouterDataset) cannot be
            # used for retrieval metrics — skip the batch gracefully.
            if "ground_truth_fact_ids" not in batch:
                logger.debug(
                    "Batch missing 'ground_truth_fact_ids' — skipping "
                    "(dataloader may not contain retrieval annotations)."
                )
                continue
            gt_fact_ids_batch: List[List[str]] = batch["ground_truth_fact_ids"]

            # Forward through backbone + retrieval head
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            query_sig = output.query_signature

            if query_sig is None:
                # No retrieval positions in this batch
                continue

            # Pool per-token outputs → per-sample if the query_signature
            # has a sequence dimension, i.e. shape (batch, seq_len, dim).
            # resolve() expects (batch, dim) tensors.
            if query_sig.semantic_embedding.ndim == 3:
                # Use attention_mask (or span_mask if available) for pooling
                pool_mask = batch.get("span_mask")
                if pool_mask is not None:
                    pool_mask = pool_mask.to(self.device)
                    # Fall back to attention_mask if span_mask is all zeros
                    if pool_mask.sum() == 0:
                        pool_mask = attention_mask.float()
                else:
                    pool_mask = attention_mask.float()

                # Expand mask for broadcasting: (batch, seq_len) → (batch, seq_len, 1)
                mask_expanded = pool_mask.unsqueeze(-1)
                denom = mask_expanded.sum(dim=1).clamp(min=1e-8)  # (batch, 1)

                pooled_sem = (query_sig.semantic_embedding * mask_expanded).sum(dim=1) / denom
                pooled_sem = torch.nn.functional.normalize(pooled_sem, p=2, dim=-1)
                pooled_gran = (query_sig.granularity_logits * mask_expanded).sum(dim=1) / denom
                pooled_temp = (query_sig.temporal_logits * mask_expanded).sum(dim=1) / denom

                query_sig = QuerySignature(
                    semantic_embedding=pooled_sem,
                    granularity_logits=pooled_gran,
                    temporal_logits=pooled_temp,
                )

            # Resolve each sample in the batch
            bsz = input_ids.size(0)
            for i in range(bsz):
                if max_samples and sample_count >= max_samples:
                    break

                # Build single-sample query signature
                single_sig = QuerySignature(
                    semantic_embedding=query_sig.semantic_embedding[i : i + 1],
                    granularity_logits=query_sig.granularity_logits[i : i + 1],
                    temporal_logits=query_sig.temporal_logits[i : i + 1],
                )

                # Resolve through FAISS + KG
                facts = model.retrieval_head.resolve(
                    single_sig, faiss_index, kg_client, top_k=max(self.k_values)
                )
                predicted_ids = [f.fact_id for f in facts]
                gt_ids = gt_fact_ids_batch[i]

                # Temporal / granularity
                pred_temp = (
                    int(single_sig.temporal_mode[0].item())
                    if self.compute_temporal
                    else None
                )
                true_temp = (
                    batch.get("ground_truth_temporal", [None] * bsz)[i]
                    if self.compute_temporal
                    else None
                )
                pred_gran = (
                    int(single_sig.granularity_level[0].item())
                    if self.compute_granularity
                    else None
                )
                true_gran = (
                    batch.get("ground_truth_granularity", [None] * bsz)[i]
                    if self.compute_granularity
                    else None
                )

                accumulator.update(
                    predicted_ids=predicted_ids,
                    ground_truth_ids=gt_ids,
                    predicted_temporal=pred_temp,
                    true_temporal=true_temp,
                    predicted_granularity=pred_gran,
                    true_granularity=true_gran,
                    relation_type=batch.get("relation_type", [None] * bsz)[i],
                    entity_frequency_bin=batch.get("entity_frequency_bin", [None] * bsz)[i],
                    temporal_complexity=batch.get("temporal_complexity", [None] * bsz)[i],
                )
                sample_count += 1

            if max_samples and sample_count >= max_samples:
                break

        if sample_count == 0:
            logger.warning(
                "Retrieval evaluation produced 0 samples — the dataloader "
                "may lack 'ground_truth_fact_ids' annotations. "
                "Use a RetrievalDataset or add fact-id annotations to "
                "evaluate retrieval metrics."
            )

        results = accumulator.compute()
        logger.info(
            "Retrieval evaluation complete: %d samples, MRR=%.4f, %s",
            results.num_samples,
            results.mrr,
            results.precision_at_k.to_dict(),
        )
        return results

    def evaluate_from_predictions(
        self,
        predictions: List[Dict[str, Any]],
    ) -> RetrievalResults:
        """Evaluate from pre-computed prediction dicts.

        Each dict should contain:
            - ``predicted_fact_ids``: list of str
            - ``ground_truth_fact_ids``: list of str
            - ``predicted_temporal`` (optional): int
            - ``true_temporal`` (optional): int
            - ``predicted_granularity`` (optional): int
            - ``true_granularity`` (optional): int
            - ``relation_type`` (optional): str
            - ``entity_frequency_bin`` (optional): str
            - ``temporal_complexity`` (optional): str

        Parameters
        ----------
        predictions : list of dict
            Pre-computed predictions.

        Returns
        -------
        RetrievalResults
        """
        accumulator = _MetricAccumulator(self.k_values)

        for pred in predictions:
            accumulator.update(
                predicted_ids=pred["predicted_fact_ids"],
                ground_truth_ids=pred["ground_truth_fact_ids"],
                predicted_temporal=pred.get("predicted_temporal"),
                true_temporal=pred.get("true_temporal"),
                predicted_granularity=pred.get("predicted_granularity"),
                true_granularity=pred.get("true_granularity"),
                relation_type=pred.get("relation_type"),
                entity_frequency_bin=pred.get("entity_frequency_bin"),
                temporal_complexity=pred.get("temporal_complexity"),
            )

        results = accumulator.compute()
        logger.info(
            "Retrieval evaluation (from predictions): %d samples, MRR=%.4f",
            results.num_samples,
            results.mrr,
        )
        return results
