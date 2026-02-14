"""
Tests for Phase 8 — Evaluation Suite.

Tests cover:
- retrieval_eval: precision_at_k, mean_reciprocal_rank, temporal_accuracy,
  granularity_accuracy, _MetricAccumulator, PrecisionAtKResult,
  RetrievalEvaluator.evaluate_from_predictions
- generation_eval: compute_perplexity, compute_token_level_loss,
  PerplexityResult, BaselineComparison, GenerationEvaluator.evaluate_from_losses
- router_eval: ConfusionMatrix, confusion_matrix, confusion_matrix_from_arrays,
  calibration_error, compute_metrics_at_threshold, RouterEvaluator.evaluate_from_predictions
- end_to_end: compute_factual_accuracy, compute_temporal_consistency,
  EndToEndEvaluator.evaluate_from_predictions, EndToEndEvaluator.export_results
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- Retrieval eval imports ----
from src.evaluation.retrieval_eval import (
    GranularityAccuracyResult,
    PrecisionAtKResult,
    RetrievalEvaluator,
    RetrievalResults,
    StratifiedResult,
    TemporalAccuracyResult,
    _MetricAccumulator,
    granularity_accuracy,
    mean_reciprocal_rank,
    precision_at_k,
    temporal_accuracy,
)

# ---- Generation eval imports ----
from src.evaluation.generation_eval import (
    BaselineComparison,
    GenerationEvaluator,
    GenerationResults,
    PerplexityResult,
    compute_perplexity,
    compute_token_level_loss,
)

# ---- Router eval imports ----
from src.evaluation.router_eval import (
    ConfusionMatrix,
    ErrorAnalysis,
    RouterEvaluator,
    RouterResults,
    ThresholdResult,
    calibration_error,
    compute_metrics_at_threshold,
    confusion_matrix,
    confusion_matrix_from_arrays,
)

# ---- End-to-end eval imports ----
from src.evaluation.end_to_end import (
    EndToEndComparison,
    EndToEndEvaluator,
    EndToEndResults,
    FactualAccuracyResult,
    TemporalConsistencyResult,
    compute_factual_accuracy,
    compute_temporal_consistency,
)


# ====================================================================
# SECTION 1 — Retrieval Evaluation
# ====================================================================


class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    def test_perfect_precision(self) -> None:
        """All top-k predictions are relevant."""
        predicted = ["a", "b", "c", "d", "e"]
        ground_truth = ["a", "b", "c", "d", "e"]
        assert precision_at_k(predicted, ground_truth, k=5) == 1.0

    def test_zero_precision(self) -> None:
        """No predictions are relevant."""
        predicted = ["x", "y", "z"]
        ground_truth = ["a", "b", "c"]
        assert precision_at_k(predicted, ground_truth, k=3) == 0.0

    def test_partial_precision(self) -> None:
        """Half of top-k are relevant."""
        predicted = ["a", "x", "b", "y"]
        ground_truth = ["a", "b"]
        assert precision_at_k(predicted, ground_truth, k=4) == 0.5

    def test_k_larger_than_predictions(self) -> None:
        """k exceeds prediction list length."""
        predicted = ["a", "b"]
        ground_truth = ["a", "b", "c"]
        # Only 2 predictions, 2 hits out of k=5
        assert precision_at_k(predicted, ground_truth, k=5) == 2 / 5

    def test_k_equals_one(self) -> None:
        """P@1 with a correct first prediction."""
        predicted = ["a", "b", "c"]
        ground_truth = ["a"]
        assert precision_at_k(predicted, ground_truth, k=1) == 1.0

    def test_k_equals_one_miss(self) -> None:
        """P@1 with an incorrect first prediction."""
        predicted = ["x", "a", "b"]
        ground_truth = ["a"]
        assert precision_at_k(predicted, ground_truth, k=1) == 0.0

    def test_empty_ground_truth(self) -> None:
        """No relevant items gives 0.0."""
        predicted = ["a", "b"]
        assert precision_at_k(predicted, [], k=2) == 0.0

    def test_invalid_k_raises(self) -> None:
        """k <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            precision_at_k(["a"], ["a"], k=0)

    def test_negative_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            precision_at_k(["a"], ["a"], k=-1)

    def test_duplicate_predictions(self) -> None:
        """Duplicates in predictions only count once per rank."""
        predicted = ["a", "a", "a"]
        ground_truth = ["a"]
        assert precision_at_k(predicted, ground_truth, k=3) == 1.0


class TestMeanReciprocalRank:
    """Tests for mean_reciprocal_rank function."""

    def test_first_position(self) -> None:
        """Relevant item is at rank 1."""
        assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self) -> None:
        """Relevant item is at rank 2."""
        assert mean_reciprocal_rank(["x", "a", "c"], ["a"]) == 0.5

    def test_third_position(self) -> None:
        """Relevant item is at rank 3."""
        assert mean_reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_no_relevant_item(self) -> None:
        """No relevant items returns 0.0."""
        assert mean_reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0

    def test_empty_ground_truth(self) -> None:
        assert mean_reciprocal_rank(["a", "b"], []) == 0.0

    def test_empty_predictions(self) -> None:
        assert mean_reciprocal_rank([], ["a"]) == 0.0

    def test_multiple_relevant(self) -> None:
        """MRR only considers the first relevant hit."""
        assert mean_reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5


class TestTemporalAccuracy:
    """Tests for temporal_accuracy function."""

    def test_perfect_accuracy(self) -> None:
        predicted = [0, 1, 2, 0, 1]
        true = [0, 1, 2, 0, 1]
        result = temporal_accuracy(predicted, true)
        assert result.overall == 1.0
        assert result.per_mode["CURRENT"] == 1.0
        assert result.per_mode["AT_TIMESTAMP"] == 1.0
        assert result.per_mode["HISTORY"] == 1.0

    def test_zero_accuracy(self) -> None:
        predicted = [1, 2, 0]
        true = [0, 0, 1]
        result = temporal_accuracy(predicted, true)
        assert result.overall == 0.0

    def test_partial_accuracy(self) -> None:
        predicted = [0, 0, 1]
        true = [0, 1, 1]
        result = temporal_accuracy(predicted, true)
        assert result.overall == pytest.approx(2 / 3)

    def test_empty_inputs(self) -> None:
        result = temporal_accuracy([], [])
        assert result.overall == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            temporal_accuracy([0, 1], [0])

    def test_per_mode_missing_mode(self) -> None:
        """Mode not present in ground truth gets 0.0."""
        predicted = [0, 0, 0]
        true = [0, 0, 0]
        result = temporal_accuracy(predicted, true)
        assert result.per_mode["CURRENT"] == 1.0
        # AT_TIMESTAMP (1) and HISTORY (2) have no samples
        assert result.per_mode["AT_TIMESTAMP"] == 0.0
        assert result.per_mode["HISTORY"] == 0.0

    def test_to_dict(self) -> None:
        result = temporal_accuracy([0, 1], [0, 1])
        d = result.to_dict()
        assert "overall" in d
        assert "per_mode" in d


class TestGranularityAccuracy:
    """Tests for granularity_accuracy function."""

    def test_perfect_accuracy(self) -> None:
        predicted = [0, 1, 2, 3]
        true = [0, 1, 2, 3]
        result = granularity_accuracy(predicted, true)
        assert result.overall == 1.0

    def test_zero_accuracy(self) -> None:
        predicted = [3, 2, 1, 0]
        true = [0, 1, 2, 3]
        result = granularity_accuracy(predicted, true)
        assert result.overall == 0.0

    def test_empty_inputs(self) -> None:
        result = granularity_accuracy([], [])
        assert result.overall == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            granularity_accuracy([0], [0, 1])

    def test_per_level(self) -> None:
        predicted = [0, 0, 1, 1]
        true = [0, 1, 1, 2]
        result = granularity_accuracy(predicted, true)
        # atomic: true has idx 0 at position 0, pred=0 → correct → 1.0
        assert result.per_level["atomic"] == 1.0
        # relation: true has idx 1 at positions 1,2; pred=[0,1] → 1/2 = 0.5
        assert result.per_level["relation"] == 0.5

    def test_to_dict(self) -> None:
        result = granularity_accuracy([0, 1], [0, 1])
        d = result.to_dict()
        assert "overall" in d
        assert "per_level" in d


class TestPrecisionAtKResult:
    """Tests for PrecisionAtKResult dataclass."""

    def test_getitem(self) -> None:
        result = PrecisionAtKResult(values={1: 0.5, 5: 0.3})
        assert result[1] == 0.5
        assert result[5] == 0.3

    def test_to_dict(self) -> None:
        result = PrecisionAtKResult(values={1: 0.5, 10: 0.2})
        d = result.to_dict()
        assert d["P@1"] == 0.5
        assert d["P@10"] == 0.2


class TestMetricAccumulator:
    """Tests for _MetricAccumulator batch accumulation."""

    def test_single_sample(self) -> None:
        acc = _MetricAccumulator(k_values=[1, 5])
        acc.update(
            predicted_ids=["a", "b", "c"],
            ground_truth_ids=["a", "d"],
            predicted_temporal=0,
            true_temporal=0,
            predicted_granularity=1,
            true_granularity=1,
        )
        results = acc.compute()
        assert results.num_samples == 1
        assert results.precision_at_k[1] == 1.0  # "a" is correct
        assert results.mrr == 1.0  # "a" at rank 1

    def test_multiple_samples(self) -> None:
        acc = _MetricAccumulator(k_values=[1])
        # Sample 1: hit at position 1
        acc.update(["a", "b"], ["a"], 0, 0, 0, 0)
        # Sample 2: miss at position 1
        acc.update(["x", "a"], ["a"], 1, 1, 1, 1)
        results = acc.compute()
        assert results.num_samples == 2
        # P@1: (1.0 + 0.0) / 2 = 0.5
        assert results.precision_at_k[1] == 0.5
        # MRR: (1.0 + 0.5) / 2 = 0.75
        assert results.mrr == 0.75

    def test_empty_accumulator(self) -> None:
        acc = _MetricAccumulator(k_values=[1, 5])
        results = acc.compute()
        assert results.num_samples == 0
        assert results.mrr == 0.0

    def test_stratified_accumulation(self) -> None:
        acc = _MetricAccumulator(k_values=[1])
        acc.update(
            ["a", "b"], ["a"],
            relation_type="INHIBITS",
        )
        acc.update(
            ["x", "a"], ["a"],
            relation_type="INHIBITS",
        )
        acc.update(
            ["a"], ["a"],
            relation_type="TREATS",
        )
        results = acc.compute()
        assert "INHIBITS" in results.stratified.by_relation_type
        assert "TREATS" in results.stratified.by_relation_type
        assert results.stratified.by_relation_type["TREATS"]["P@1"] == 1.0


class TestRetrievalEvaluator:
    """Tests for RetrievalEvaluator class."""

    def test_default_k_values(self) -> None:
        evaluator = RetrievalEvaluator()
        assert evaluator.k_values == [1, 5, 10, 20]

    def test_custom_k_values(self) -> None:
        evaluator = RetrievalEvaluator(k_values=[1, 3, 7])
        assert evaluator.k_values == [1, 3, 7]

    def test_evaluate_from_predictions(self) -> None:
        evaluator = RetrievalEvaluator(k_values=[1, 5])
        predictions = [
            {
                "predicted_fact_ids": ["a", "b", "c", "d", "e"],
                "ground_truth_fact_ids": ["a", "c"],
                "predicted_temporal": 0,
                "true_temporal": 0,
                "predicted_granularity": 2,
                "true_granularity": 2,
            },
            {
                "predicted_fact_ids": ["x", "y", "z"],
                "ground_truth_fact_ids": ["y"],
                "predicted_temporal": 1,
                "true_temporal": 1,
                "predicted_granularity": 0,
                "true_granularity": 0,
            },
        ]
        results = evaluator.evaluate_from_predictions(predictions)
        assert results.num_samples == 2
        # P@1: (1.0 + 0.0) / 2 = 0.5
        assert results.precision_at_k[1] == pytest.approx(0.5)
        assert results.temporal.overall == 1.0
        assert results.granularity.overall == 1.0

    def test_evaluate_from_predictions_empty(self) -> None:
        evaluator = RetrievalEvaluator()
        results = evaluator.evaluate_from_predictions([])
        assert results.num_samples == 0

    def test_results_to_dict(self) -> None:
        evaluator = RetrievalEvaluator(k_values=[1])
        predictions = [
            {
                "predicted_fact_ids": ["a"],
                "ground_truth_fact_ids": ["a"],
            },
        ]
        results = evaluator.evaluate_from_predictions(predictions)
        d = results.to_dict()
        assert "P@1" in d
        assert "MRR" in d
        assert "num_samples" in d


class TestRetrievalResults:
    """Tests for RetrievalResults dataclass."""

    def test_to_dict_keys(self) -> None:
        results = RetrievalResults(
            precision_at_k=PrecisionAtKResult(values={1: 0.5}),
            mrr=0.75,
            num_samples=10,
        )
        d = results.to_dict()
        assert d["P@1"] == 0.5
        assert d["MRR"] == 0.75
        assert d["num_samples"] == 10


# ====================================================================
# SECTION 2 — Generation Evaluation
# ====================================================================


class TestComputePerplexity:
    """Tests for compute_perplexity function."""

    def test_zero_loss(self) -> None:
        assert compute_perplexity(0.0, 100) == pytest.approx(1.0)

    def test_known_perplexity(self) -> None:
        # PPL = exp(avg_loss) = exp(1.0) = e ≈ 2.718
        assert compute_perplexity(100.0, 100) == pytest.approx(math.e, rel=1e-3)

    def test_zero_tokens_returns_inf(self) -> None:
        result = compute_perplexity(5.0, 0)
        assert result == float("inf")

    def test_large_loss_returns_inf(self) -> None:
        result = compute_perplexity(1e10, 1)
        assert result == float("inf")

    def test_fractional_loss(self) -> None:
        # PPL = exp(0.5) ≈ 1.6487
        result = compute_perplexity(5.0, 10)
        assert result == pytest.approx(math.exp(0.5), rel=1e-3)


class TestComputeTokenLevelLoss:
    """Tests for compute_token_level_loss function."""

    VOCAB_SIZE = 128
    BATCH = 4
    SEQ_LEN = 16

    def test_basic_loss_computation(self) -> None:
        """Loss is non-negative and token count is correct."""
        logits = torch.randn(self.BATCH, self.SEQ_LEN, self.VOCAB_SIZE)
        labels = torch.randint(0, self.VOCAB_SIZE, (self.BATCH, self.SEQ_LEN))
        total_loss, num_tokens = compute_token_level_loss(logits, labels)
        assert total_loss > 0
        # Shifted: seq_len - 1 per sample
        assert num_tokens == self.BATCH * (self.SEQ_LEN - 1)

    def test_causal_shift(self) -> None:
        """Output should use shifted logits (predict next token)."""
        logits = torch.randn(2, 8, self.VOCAB_SIZE)
        labels = torch.randint(0, self.VOCAB_SIZE, (2, 8))
        _, num_tokens = compute_token_level_loss(logits, labels)
        # 2 * (8 - 1) = 14
        assert num_tokens == 14

    def test_ignore_index(self) -> None:
        """Tokens with ignore_index=-100 are not counted."""
        logits = torch.randn(1, 10, self.VOCAB_SIZE)
        labels = torch.full((1, 10), -100, dtype=torch.long)
        labels[0, 5] = 42  # Only 1 valid token in shifted labels
        # After shift, labels become [-100, -100, -100, -100, 42, -100, -100, -100, -100]
        total_loss, num_tokens = compute_token_level_loss(logits, labels)
        assert num_tokens == 1

    def test_with_mask(self) -> None:
        """External mask restricts computation."""
        logits = torch.randn(1, 10, self.VOCAB_SIZE)
        labels = torch.randint(0, self.VOCAB_SIZE, (1, 10))
        # Mask: only first 5 tokens (after shift, positions 0-3)
        mask = torch.zeros(1, 10, dtype=torch.bool)
        mask[0, :5] = True
        total_loss, num_tokens = compute_token_level_loss(logits, labels, mask=mask)
        # Shifted mask: mask[:, 1:] → [True, True, True, True, False, False, False, False, False]
        assert num_tokens == 4

    def test_all_ignored_gives_zero(self) -> None:
        logits = torch.randn(1, 5, self.VOCAB_SIZE)
        labels = torch.full((1, 5), -100, dtype=torch.long)
        total_loss, num_tokens = compute_token_level_loss(logits, labels)
        assert num_tokens == 0
        assert total_loss == 0.0


class TestPerplexityResult:
    """Tests for PerplexityResult dataclass."""

    def test_to_dict(self) -> None:
        ppl = PerplexityResult(overall=10.0, generation_spans=8.0, retrieval_spans=15.0)
        d = ppl.to_dict()
        assert d["overall_perplexity"] == 10.0
        assert d["generation_span_perplexity"] == 8.0
        assert d["retrieval_span_perplexity"] == 15.0


class TestBaselineComparison:
    """Tests for BaselineComparison dataclass."""

    def test_improvement_pct(self) -> None:
        comp = BaselineComparison(frlm_perplexity=80.0, baseline_perplexity=100.0)
        assert comp.improvement_pct == pytest.approx(20.0)

    def test_improvement_pct_zero_baseline(self) -> None:
        comp = BaselineComparison(frlm_perplexity=50.0, baseline_perplexity=0.0)
        assert comp.improvement_pct == 0.0

    def test_to_dict(self) -> None:
        comp = BaselineComparison(
            frlm_perplexity=80.0,
            baseline_perplexity=100.0,
            perplexity_reduction=20.0,
            frlm_loss=2.0,
            baseline_loss=3.0,
            loss_reduction=1.0,
        )
        d = comp.to_dict()
        assert "improvement_pct" in d
        assert d["frlm_perplexity"] == 80.0


class TestGenerationEvaluator:
    """Tests for GenerationEvaluator class."""

    def test_evaluate_from_losses_basic(self) -> None:
        evaluator = GenerationEvaluator()
        losses = [1.0, 2.0, 3.0, 4.0, 5.0]
        results = evaluator.evaluate_from_losses(losses)
        assert results.num_tokens == 5
        assert results.cross_entropy_loss == pytest.approx(3.0)
        expected_ppl = math.exp(15.0 / 5)
        assert results.perplexity.overall == pytest.approx(expected_ppl, rel=1e-3)

    def test_evaluate_from_losses_with_router_mask(self) -> None:
        evaluator = GenerationEvaluator()
        losses = [1.0, 2.0, 3.0, 4.0]
        router_mask = [False, False, True, True]  # last 2 are retrieval
        results = evaluator.evaluate_from_losses(losses, router_mask)
        assert results.num_tokens == 4
        assert results.num_generation_tokens == 2
        assert results.num_retrieval_tokens == 2
        # Gen spans: 1.0 + 2.0 = 3.0, PPL = exp(3.0/2) = exp(1.5)
        assert results.perplexity.generation_spans == pytest.approx(
            math.exp(1.5), rel=1e-3
        )
        # Ret spans: 3.0 + 4.0 = 7.0, PPL = exp(7.0/2) = exp(3.5)
        assert results.perplexity.retrieval_spans == pytest.approx(
            math.exp(3.5), rel=1e-3
        )

    def test_evaluate_from_losses_empty(self) -> None:
        evaluator = GenerationEvaluator()
        results = evaluator.evaluate_from_losses([])
        assert results.num_tokens == 0

    def test_generation_results_to_dict(self) -> None:
        results = GenerationResults(
            perplexity=PerplexityResult(overall=10.0),
            cross_entropy_loss=2.3,
            num_tokens=100,
        )
        d = results.to_dict()
        assert d["overall_perplexity"] == 10.0
        assert d["cross_entropy_loss"] == 2.3
        assert d["num_tokens"] == 100


# ====================================================================
# SECTION 3 — Router Evaluation
# ====================================================================


class TestConfusionMatrix:
    """Tests for ConfusionMatrix dataclass."""

    def test_properties(self) -> None:
        cm = ConfusionMatrix(tp=50, tn=40, fp=10, fn=5)
        assert cm.total == 105
        assert cm.accuracy == pytest.approx(90 / 105)
        assert cm.precision == pytest.approx(50 / 60)
        assert cm.recall == pytest.approx(50 / 55)
        assert cm.specificity == pytest.approx(40 / 50)

    def test_f1(self) -> None:
        cm = ConfusionMatrix(tp=50, tn=40, fp=10, fn=5)
        p = 50 / 60
        r = 50 / 55
        expected_f1 = 2 * p * r / (p + r)
        assert cm.f1 == pytest.approx(expected_f1)

    def test_all_zeros(self) -> None:
        cm = ConfusionMatrix()
        assert cm.total == 0
        assert cm.accuracy == 0.0
        assert cm.precision == 0.0
        assert cm.recall == 0.0
        assert cm.f1 == 0.0
        assert cm.specificity == 0.0

    def test_to_matrix(self) -> None:
        cm = ConfusionMatrix(tp=50, tn=40, fp=10, fn=5)
        matrix = cm.to_matrix()
        assert matrix == [[40, 10], [5, 50]]

    def test_to_dict_keys(self) -> None:
        cm = ConfusionMatrix(tp=1, tn=1, fp=1, fn=1)
        d = cm.to_dict()
        expected_keys = {"tp", "tn", "fp", "fn", "accuracy", "precision",
                         "recall", "f1", "specificity", "matrix"}
        assert set(d.keys()) == expected_keys

    def test_no_positive_predictions(self) -> None:
        """Precision is 0 when no positive predictions."""
        cm = ConfusionMatrix(tp=0, tn=100, fp=0, fn=5)
        assert cm.precision == 0.0


class TestConfusionMatrixFunction:
    """Tests for confusion_matrix() function."""

    def test_perfect_predictions(self) -> None:
        preds = [True, True, False, False]
        labels = [True, True, False, False]
        cm = confusion_matrix(preds, labels)
        assert cm.tp == 2
        assert cm.tn == 2
        assert cm.fp == 0
        assert cm.fn == 0

    def test_all_wrong(self) -> None:
        preds = [True, True, False, False]
        labels = [False, False, True, True]
        cm = confusion_matrix(preds, labels)
        assert cm.tp == 0
        assert cm.tn == 0
        assert cm.fp == 2
        assert cm.fn == 2

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            confusion_matrix([True], [True, False])


class TestConfusionMatrixFromArrays:
    """Tests for confusion_matrix_from_arrays() vectorised function."""

    def test_matches_scalar_version(self) -> None:
        preds = [True, False, True, False, True]
        labels = [True, False, False, True, True]
        scalar_cm = confusion_matrix(preds, labels)
        arr_cm = confusion_matrix_from_arrays(
            np.array(preds), np.array(labels)
        )
        assert arr_cm.tp == scalar_cm.tp
        assert arr_cm.tn == scalar_cm.tn
        assert arr_cm.fp == scalar_cm.fp
        assert arr_cm.fn == scalar_cm.fn

    def test_large_array(self) -> None:
        """Test with larger arrays for correctness."""
        rng = np.random.RandomState(42)
        preds = rng.randint(0, 2, size=1000).astype(bool)
        labels = rng.randint(0, 2, size=1000).astype(bool)
        cm = confusion_matrix_from_arrays(preds, labels)
        assert cm.total == 1000
        assert cm.tp + cm.tn + cm.fp + cm.fn == 1000


class TestCalibrationError:
    """Tests for calibration_error (ECE) function."""

    def test_perfect_calibration(self) -> None:
        """If predicted prob matches actual accuracy perfectly, ECE ~0."""
        # 100 positive samples with prob=1.0 → perfect calibration
        probs = [1.0] * 100
        labels = [True] * 100
        ece = calibration_error(probs, labels, num_bins=10)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_worst_calibration(self) -> None:
        """All prob=1.0 but all labels=False → maximal miscalibration."""
        probs = [1.0] * 100
        labels = [False] * 100
        ece = calibration_error(probs, labels, num_bins=10)
        assert ece == pytest.approx(1.0, abs=0.01)

    def test_empty_probs(self) -> None:
        assert calibration_error([], [], num_bins=10) == 0.0

    def test_uniform_probs(self) -> None:
        """ECE for uniformly distributed probs."""
        rng = np.random.RandomState(42)
        probs = rng.uniform(0, 1, size=500).tolist()
        labels = (rng.uniform(0, 1, size=500) > 0.5).tolist()
        ece = calibration_error(probs, labels, num_bins=10)
        assert 0.0 <= ece <= 1.0


class TestComputeMetricsAtThreshold:
    """Tests for compute_metrics_at_threshold function."""

    def test_threshold_0_5(self) -> None:
        probs = np.array([0.1, 0.4, 0.6, 0.9])
        labels = np.array([False, False, True, True])
        result = compute_metrics_at_threshold(probs, labels, threshold=0.5)
        assert result.threshold == 0.5
        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_threshold_too_high(self) -> None:
        probs = np.array([0.1, 0.4, 0.6, 0.9])
        labels = np.array([False, False, True, True])
        result = compute_metrics_at_threshold(probs, labels, threshold=0.8)
        # Only 0.9 >= 0.8 → pred = [F, F, F, T]
        assert result.confusion.tp == 1
        assert result.confusion.fn == 1

    def test_threshold_0_gives_all_positive(self) -> None:
        probs = np.array([0.1, 0.5, 0.9])
        labels = np.array([True, False, True])
        result = compute_metrics_at_threshold(probs, labels, threshold=0.0)
        assert result.confusion.tp == 2
        assert result.confusion.fp == 1


class TestRouterEvaluator:
    """Tests for RouterEvaluator class."""

    def test_default_threshold(self) -> None:
        evaluator = RouterEvaluator()
        assert evaluator.threshold == 0.5

    def test_evaluate_from_predictions(self) -> None:
        evaluator = RouterEvaluator(
            threshold=0.5,
            threshold_sweep_values=[0.3, 0.5, 0.7],
        )
        probs = [0.1, 0.3, 0.6, 0.8, 0.2, 0.9, 0.4, 0.7]
        labels = [False, False, True, True, False, True, False, True]
        results = evaluator.evaluate_from_predictions(probs, labels)
        assert results.num_samples == 8
        assert results.threshold == 0.5
        assert 0.0 <= results.accuracy <= 1.0
        assert 0.0 <= results.f1 <= 1.0
        assert 0.0 <= results.expected_calibration_error <= 1.0
        # Threshold sweep should have 3 entries
        assert len(results.threshold_sweep) == 3

    def test_evaluate_from_predictions_perfect(self) -> None:
        evaluator = RouterEvaluator(threshold=0.5)
        probs = [0.0, 0.0, 1.0, 1.0]
        labels = [False, False, True, True]
        results = evaluator.evaluate_from_predictions(probs, labels)
        assert results.accuracy == 1.0
        assert results.f1 == 1.0

    def test_best_threshold_selection(self) -> None:
        evaluator = RouterEvaluator(threshold_sweep_values=[0.3, 0.5, 0.7])
        probs = [0.4, 0.6, 0.8]
        labels = [False, True, True]
        results = evaluator.evaluate_from_predictions(probs, labels)
        # best_threshold should be chosen based on highest F1
        assert results.best_threshold in [0.3, 0.5, 0.7]
        assert results.best_f1 >= 0.0

    def test_error_analysis(self) -> None:
        evaluator = RouterEvaluator(threshold=0.5)
        # Create some FP and FN cases
        probs = [0.9, 0.1, 0.9, 0.1]
        labels = [False, True, True, False]  # FP=1, FN=1
        results = evaluator.evaluate_from_predictions(probs, labels)
        assert len(results.error_analysis.false_positive_examples) >= 1
        assert len(results.error_analysis.false_negative_examples) >= 1

    def test_router_results_to_dict(self) -> None:
        results = RouterResults(
            threshold=0.5,
            accuracy=0.85,
            precision=0.9,
            recall=0.8,
            f1=0.85,
            num_samples=100,
        )
        d = results.to_dict()
        assert d["threshold"] == 0.5
        assert d["accuracy"] == 0.85
        assert d["num_samples"] == 100


class TestThresholdResult:
    """Tests for ThresholdResult dataclass."""

    def test_to_dict(self) -> None:
        tr = ThresholdResult(threshold=0.6, accuracy=0.9, f1=0.88)
        d = tr.to_dict()
        assert d["threshold"] == 0.6
        assert d["accuracy"] == 0.9


class TestErrorAnalysis:
    """Tests for ErrorAnalysis dataclass."""

    def test_to_dict(self) -> None:
        ea = ErrorAnalysis(
            false_positive_examples=[{"index": 0, "prob": 0.8}],
            false_negative_examples=[],
            error_by_position={"first_quarter": 0.1},
        )
        d = ea.to_dict()
        assert d["false_positive_count"] == 1
        assert d["false_negative_count"] == 0
        assert d["error_by_position"]["first_quarter"] == 0.1


# ====================================================================
# SECTION 4 — End-to-End Evaluation
# ====================================================================


class TestComputeFactualAccuracy:
    """Tests for compute_factual_accuracy function."""

    def test_all_facts_present(self) -> None:
        texts = ["Aspirin treats Headache and reduces pain."]
        gt_facts = [[
            {"subject_label": "Aspirin", "object_label": "Headache", "relation_type": "TREATS"}
        ]]
        result = compute_factual_accuracy(texts, [[]], gt_facts)
        assert result.overall == 1.0
        assert result.by_relation_type["TREATS"] == 1.0

    def test_no_facts_present(self) -> None:
        texts = ["Something completely unrelated."]
        gt_facts = [[
            {"subject_label": "Aspirin", "object_label": "Headache", "relation_type": "TREATS"}
        ]]
        result = compute_factual_accuracy(texts, [[]], gt_facts)
        assert result.overall == 0.0

    def test_partial_facts(self) -> None:
        texts = ["Aspirin inhibits COX-2. Unrelated text about weather."]
        gt_facts = [[
            {"subject_label": "Aspirin", "object_label": "COX-2", "relation_type": "INHIBITS"},
            {"subject_label": "Drug-X", "object_label": "Target-Y", "relation_type": "ACTIVATES"},
        ]]
        result = compute_factual_accuracy(texts, [[]], gt_facts)
        assert result.overall == pytest.approx(0.5)

    def test_case_insensitive(self) -> None:
        texts = ["aspirin treats headache"]
        gt_facts = [[
            {"subject_label": "Aspirin", "object_label": "Headache", "relation_type": "TREATS"}
        ]]
        result = compute_factual_accuracy(texts, [[]], gt_facts)
        assert result.overall == 1.0

    def test_empty_inputs(self) -> None:
        result = compute_factual_accuracy([], [], [])
        assert result.overall == 0.0
        assert result.num_samples == 0

    def test_multiple_samples(self) -> None:
        texts = [
            "Aspirin treats Headache",
            "No match here",
        ]
        gt_facts = [
            [{"subject_label": "Aspirin", "object_label": "Headache", "relation_type": "TREATS"}],
            [{"subject_label": "Drug-X", "object_label": "Target-Y", "relation_type": "INHIBITS"}],
        ]
        result = compute_factual_accuracy(texts, [[], []], gt_facts)
        assert result.overall == 0.5
        assert result.num_samples == 2


class TestComputeTemporalConsistency:
    """Tests for compute_temporal_consistency function."""

    def test_current_mode_with_current_keywords(self) -> None:
        texts = ["Aspirin is currently used to treat headaches."]
        modes = ["CURRENT"]
        timestamps = [None]
        result = compute_temporal_consistency(texts, modes, timestamps)
        assert result.overall == 1.0

    def test_history_mode_with_past_tense(self) -> None:
        texts = ["This drug was previously used for treatment."]
        modes = ["HISTORY"]
        timestamps = [None]
        result = compute_temporal_consistency(texts, modes, timestamps)
        assert result.overall == 1.0
        assert result.by_mode["HISTORY"] == 1.0

    def test_at_timestamp_mode(self) -> None:
        texts = ["In 2020, the study found results."]
        modes = ["AT_TIMESTAMP"]
        timestamps = ["2020-01-01"]
        result = compute_temporal_consistency(texts, modes, timestamps)
        assert result.overall == 1.0

    def test_empty_inputs(self) -> None:
        result = compute_temporal_consistency([], [], [])
        assert result.overall == 0.0
        assert result.num_samples == 0

    def test_multiple_modes(self) -> None:
        texts = [
            "Currently approved treatment.",
            "Was previously used in 1990.",
            "Study from 2015 shows results.",
        ]
        modes = ["CURRENT", "HISTORY", "AT_TIMESTAMP"]
        timestamps = [None, None, "2015-01-01"]
        result = compute_temporal_consistency(texts, modes, timestamps)
        assert result.num_samples == 3
        assert 0.0 <= result.overall <= 1.0


class TestFactualAccuracyResult:
    """Tests for FactualAccuracyResult dataclass."""

    def test_to_dict(self) -> None:
        result = FactualAccuracyResult(
            overall=0.85,
            by_relation_type={"TREATS": 0.9, "INHIBITS": 0.8},
            num_samples=50,
        )
        d = result.to_dict()
        assert d["overall"] == 0.85
        assert d["num_samples"] == 50
        assert "TREATS" in d["by_relation_type"]


class TestTemporalConsistencyResult:
    """Tests for TemporalConsistencyResult dataclass."""

    def test_to_dict(self) -> None:
        result = TemporalConsistencyResult(
            overall=0.9,
            by_mode={"CURRENT": 1.0, "HISTORY": 0.8},
            num_samples=20,
        )
        d = result.to_dict()
        assert d["overall"] == 0.9
        assert "CURRENT" in d["by_mode"]


class TestEndToEndComparison:
    """Tests for EndToEndComparison dataclass."""

    def test_to_dict(self) -> None:
        comp = EndToEndComparison(
            frlm_perplexity=50.0,
            baseline_perplexity=80.0,
            improvement_perplexity=0.375,
        )
        d = comp.to_dict()
        assert d["frlm_perplexity"] == 50.0
        assert "improvement_perplexity_pct" in d


class TestEndToEndResults:
    """Tests for EndToEndResults dataclass."""

    def test_to_dict_minimal(self) -> None:
        results = EndToEndResults(overall_score=0.75)
        d = results.to_dict()
        assert d["overall_score"] == 0.75
        assert "factual_accuracy" in d
        assert "temporal_consistency" in d

    def test_to_dict_with_components(self) -> None:
        results = EndToEndResults(
            retrieval=RetrievalResults(mrr=0.8, num_samples=10),
            generation=GenerationResults(
                perplexity=PerplexityResult(overall=20.0),
                num_tokens=100,
            ),
            router=RouterResults(f1=0.85, num_samples=50),
            overall_score=0.78,
        )
        d = results.to_dict()
        assert "retrieval" in d
        assert "generation" in d
        assert "router" in d


class TestEndToEndEvaluator:
    """Tests for EndToEndEvaluator class."""

    def test_init_defaults(self) -> None:
        evaluator = EndToEndEvaluator()
        assert evaluator.compute_factual is True
        assert evaluator.compute_temporal is True

    def test_evaluate_from_predictions_retrieval_only(self) -> None:
        evaluator = EndToEndEvaluator()
        predictions = [
            {
                "predicted_fact_ids": ["a", "b"],
                "ground_truth_fact_ids": ["a"],
            }
        ]
        results = evaluator.evaluate_from_predictions(
            retrieval_predictions=predictions,
        )
        assert results.retrieval is not None
        assert results.retrieval.num_samples == 1

    def test_evaluate_from_predictions_router_only(self) -> None:
        evaluator = EndToEndEvaluator()
        results = evaluator.evaluate_from_predictions(
            router_probs=[0.1, 0.8, 0.6, 0.3],
            router_labels=[False, True, True, False],
        )
        assert results.router is not None
        assert results.router.num_samples == 4

    def test_evaluate_from_predictions_factual_accuracy(self) -> None:
        evaluator = EndToEndEvaluator()
        results = evaluator.evaluate_from_predictions(
            generated_texts=["Aspirin treats Headache"],
            ground_truth_facts=[[
                {"subject_label": "Aspirin", "object_label": "Headache",
                 "relation_type": "TREATS"}
            ]],
        )
        assert results.factual_accuracy.overall == 1.0

    def test_evaluate_from_predictions_all_components(self) -> None:
        evaluator = EndToEndEvaluator()
        results = evaluator.evaluate_from_predictions(
            retrieval_predictions=[
                {"predicted_fact_ids": ["a"], "ground_truth_fact_ids": ["a"]},
            ],
            router_probs=[0.1, 0.9],
            router_labels=[False, True],
            generated_texts=["Aspirin treats Headache"],
            ground_truth_facts=[[
                {"subject_label": "Aspirin", "object_label": "Headache",
                 "relation_type": "TREATS"}
            ]],
            temporal_modes=["CURRENT"],
            timestamps=[None],
        )
        assert results.retrieval is not None
        assert results.router is not None
        assert results.factual_accuracy.overall == 1.0
        assert results.overall_score > 0.0

    def test_evaluate_from_predictions_generation(self) -> None:
        evaluator = EndToEndEvaluator()
        losses = [1.0, 2.0, 3.0]
        mask = [False, True, False]
        results = evaluator.evaluate_from_predictions(
            generation_losses=losses,
            generation_router_mask=mask,
        )
        assert results.generation is not None
        assert results.generation.num_tokens == 3

    def test_export_results(self) -> None:
        results = EndToEndResults(overall_score=0.85)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results" / "eval.json"
            written = EndToEndEvaluator.export_results(results, output_path)
            assert written.exists()
            with open(written) as f:
                data = json.load(f)
            assert data["overall_score"] == 0.85

    def test_export_results_nested_dir(self) -> None:
        """Export creates parent directories."""
        results = EndToEndResults(overall_score=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "a" / "b" / "c" / "results.json"
            written = EndToEndEvaluator.export_results(results, output_path)
            assert written.exists()


# ====================================================================
# SECTION 5 — Integration tests: evaluation __init__ exports
# ====================================================================


class TestEvaluationModuleExports:
    """Ensure the evaluation __init__.py re-exports everything."""

    def test_retrieval_exports(self) -> None:
        from src.evaluation import (
            precision_at_k,
            mean_reciprocal_rank,
            temporal_accuracy,
            granularity_accuracy,
            RetrievalEvaluator,
            RetrievalResults,
        )

    def test_generation_exports(self) -> None:
        from src.evaluation import (
            compute_perplexity,
            compute_token_level_loss,
            GenerationEvaluator,
            GenerationResults,
            BaselineComparison,
        )

    def test_router_exports(self) -> None:
        from src.evaluation import (
            confusion_matrix,
            confusion_matrix_from_arrays,
            calibration_error,
            RouterEvaluator,
            RouterResults,
            ConfusionMatrix,
        )

    def test_end_to_end_exports(self) -> None:
        from src.evaluation import (
            compute_factual_accuracy,
            compute_temporal_consistency,
            EndToEndEvaluator,
            EndToEndResults,
        )
