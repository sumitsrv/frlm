"""
Router Evaluator — metrics for the FRLM binary router head.

Computes:
- Accuracy, precision, recall, F1 at a given threshold
- Threshold sweep to find optimal operating point
- Confusion matrix (TP, TN, FP, FN)
- Expected Calibration Error (ECE)
- Error analysis by span type and position

Public API
----------
- ``evaluate(model, dataloader, threshold)`` → ``RouterResults``
- ``threshold_sweep(model, dataloader, thresholds)`` → ``list[ThresholdResult]``
- ``confusion_matrix(predictions, labels)`` → ``ConfusionMatrix``
- ``calibration_error(probs, labels, num_bins)`` → ``float``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ===========================================================================
# Result containers
# ===========================================================================


@dataclass
class ConfusionMatrix:
    """Standard 2×2 confusion matrix for binary classification."""

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    def to_matrix(self) -> List[List[int]]:
        """Return as [[TN, FP], [FN, TP]]."""
        return [[self.tn, self.fp], [self.fn, self.tp]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "specificity": round(self.specificity, 4),
            "matrix": self.to_matrix(),
        }


@dataclass
class ThresholdResult:
    """Metrics at a specific threshold."""

    threshold: float = 0.5
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)

    def to_dict(self) -> Dict[str, float]:
        return {
            "threshold": self.threshold,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "specificity": round(self.specificity, 4),
        }


@dataclass
class ErrorAnalysis:
    """Error analysis breakdown."""

    false_positive_examples: List[Dict[str, Any]] = field(default_factory=list)
    false_negative_examples: List[Dict[str, Any]] = field(default_factory=list)
    error_by_position: Dict[str, float] = field(default_factory=dict)
    error_by_context_length: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "false_positive_count": len(self.false_positive_examples),
            "false_negative_count": len(self.false_negative_examples),
            "error_by_position": dict(self.error_by_position),
            "error_by_context_length": dict(self.error_by_context_length),
        }


@dataclass
class RouterResults:
    """Complete router evaluation results."""

    threshold: float = 0.5
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    expected_calibration_error: float = 0.0
    threshold_sweep: List[ThresholdResult] = field(default_factory=list)
    best_threshold: float = 0.5
    best_f1: float = 0.0
    error_analysis: ErrorAnalysis = field(default_factory=ErrorAnalysis)
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "threshold": self.threshold,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "specificity": round(self.specificity, 4),
            "confusion_matrix": self.confusion.to_matrix(),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "best_threshold": self.best_threshold,
            "best_f1": round(self.best_f1, 4),
            "threshold_sweep": [t.to_dict() for t in self.threshold_sweep],
            "error_analysis": self.error_analysis.to_dict(),
            "num_samples": self.num_samples,
        }
        return result


# ===========================================================================
# Core metric functions
# ===========================================================================


def confusion_matrix(
    predictions: Sequence[bool],
    labels: Sequence[bool],
) -> ConfusionMatrix:
    """Compute confusion matrix from binary predictions and labels.

    Parameters
    ----------
    predictions : sequence of bool
        Model predictions (True = retrieval).
    labels : sequence of bool
        Ground truth (True = retrieval).

    Returns
    -------
    ConfusionMatrix
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, labels={len(labels)}"
        )

    tp = tn = fp = fn = 0
    for pred, label in zip(predictions, labels):
        if pred and label:
            tp += 1
        elif not pred and not label:
            tn += 1
        elif pred and not label:
            fp += 1
        else:
            fn += 1

    return ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)


def confusion_matrix_from_arrays(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> ConfusionMatrix:
    """Vectorised confusion matrix computation.

    Parameters
    ----------
    predictions : np.ndarray
        Boolean array of predictions.
    labels : np.ndarray
        Boolean array of ground truth.

    Returns
    -------
    ConfusionMatrix
    """
    tp = int(np.sum(predictions & labels))
    tn = int(np.sum(~predictions & ~labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    return ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)


def calibration_error(
    probs: Sequence[float],
    labels: Sequence[bool],
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Partitions predictions into bins by predicted probability,
    then measures the gap between mean predicted confidence and
    actual accuracy in each bin, weighted by bin size.

    Parameters
    ----------
    probs : sequence of float
        Predicted probabilities in [0, 1].
    labels : sequence of bool
        Ground truth labels.
    num_bins : int
        Number of bins for calibration.

    Returns
    -------
    float
        ECE in [0, 1].
    """
    if not probs:
        return 0.0

    prob_arr = np.array(probs, dtype=np.float64)
    label_arr = np.array(labels, dtype=np.float64)
    n = len(prob_arr)

    bin_edges = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        in_bin = (prob_arr >= bin_edges[i]) & (prob_arr < bin_edges[i + 1])
        # Include right edge for last bin
        if i == num_bins - 1:
            in_bin = in_bin | (prob_arr == bin_edges[i + 1])

        bin_count = int(in_bin.sum())
        if bin_count == 0:
            continue

        avg_confidence = float(prob_arr[in_bin].mean())
        avg_accuracy = float(label_arr[in_bin].mean())
        ece += (bin_count / n) * abs(avg_confidence - avg_accuracy)

    return ece


def compute_metrics_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> ThresholdResult:
    """Compute all classification metrics at a given threshold.

    Parameters
    ----------
    probs : np.ndarray
        Predicted probabilities.
    labels : np.ndarray
        Binary ground truth (bool or int).
    threshold : float
        Decision boundary.

    Returns
    -------
    ThresholdResult
    """
    preds = probs >= threshold
    cm = confusion_matrix_from_arrays(preds, labels.astype(bool))
    return ThresholdResult(
        threshold=threshold,
        accuracy=cm.accuracy,
        precision=cm.precision,
        recall=cm.recall,
        f1=cm.f1,
        specificity=cm.specificity,
        confusion=cm,
    )


# ===========================================================================
# Evaluator class
# ===========================================================================


class RouterEvaluator:
    """Full router evaluation pipeline.

    Parameters
    ----------
    threshold : float
        Default decision threshold.
    threshold_sweep_values : list of float
        Thresholds to sweep for finding the optimal operating point.
    compute_calibration : bool
        Whether to compute ECE.
    calibration_bins : int
        Number of bins for calibration.
    max_error_examples : int
        Maximum number of FP/FN examples to store for error analysis.
    device : str
        Device for model inference.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        threshold_sweep_values: Optional[List[float]] = None,
        compute_calibration: bool = True,
        calibration_bins: int = 10,
        max_error_examples: int = 50,
        device: str = "cpu",
    ) -> None:
        self.threshold = threshold
        self.threshold_sweep_values = threshold_sweep_values or [
            0.3, 0.4, 0.5, 0.6, 0.7
        ]
        self.compute_calibration = compute_calibration
        self.calibration_bins = calibration_bins
        self.max_error_examples = max_error_examples
        self.device = device

    @classmethod
    def from_config(cls, eval_config: Any) -> "RouterEvaluator":
        """Create from a :class:`RouterEvalConfig`."""
        return cls(
            threshold_sweep_values=list(eval_config.threshold_sweep),
            compute_calibration=eval_config.compute_calibration,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: Any,
        dataloader: DataLoader,
        threshold: Optional[float] = None,
        max_samples: Optional[int] = None,
    ) -> RouterResults:
        """Run router evaluation over the dataloader.

        Each batch should yield dicts with keys:
            - ``input_ids``: ``(batch, seq_len)``
            - ``attention_mask``: ``(batch, seq_len)``
            - ``router_labels``: ``(batch, seq_len)`` binary — 1=retrieval

        Parameters
        ----------
        model : FRLMModel
            FRLM model in eval mode.
        dataloader : DataLoader
            Test set.
        threshold : float, optional
            Override default threshold.
        max_samples : int, optional
            Cap on samples.

        Returns
        -------
        RouterResults
        """
        t = threshold if threshold is not None else self.threshold
        model.eval()
        model.to(self.device)

        all_probs: List[float] = []
        all_labels: List[bool] = []
        sample_count = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            router_labels = batch["router_labels"].to(self.device)

            bsz = input_ids.size(0)
            if max_samples and sample_count + bsz > max_samples:
                remaining = max_samples - sample_count
                input_ids = input_ids[:remaining]
                attention_mask = attention_mask[:remaining]
                router_labels = router_labels[:remaining]

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = output.router_probs.squeeze(-1)  # (batch, seq_len)

            # Flatten, filtering by attention mask
            mask = attention_mask.bool()
            if mask.shape != probs.shape:
                # Handle dimension mismatch
                min_len = min(mask.size(-1), probs.size(-1))
                mask = mask[:, :min_len]
                probs = probs[:, :min_len]
                router_labels = router_labels[:, :min_len]

            flat_probs = probs[mask].cpu().numpy().tolist()
            flat_labels = router_labels[mask].cpu().bool().numpy().tolist()

            all_probs.extend(flat_probs)
            all_labels.extend(flat_labels)

            sample_count += input_ids.size(0)
            if max_samples and sample_count >= max_samples:
                break

        if not all_probs:
            return RouterResults(threshold=t, num_samples=0)

        return self._compute_results(
            probs=np.array(all_probs),
            labels=np.array(all_labels),
            threshold=t,
        )

    def evaluate_from_predictions(
        self,
        probs: Sequence[float],
        labels: Sequence[bool],
        threshold: Optional[float] = None,
    ) -> RouterResults:
        """Evaluate from pre-computed probabilities and labels.

        Parameters
        ----------
        probs : sequence of float
            Router probabilities in [0, 1].
        labels : sequence of bool
            Ground truth (True = retrieval).
        threshold : float, optional
            Override default threshold.

        Returns
        -------
        RouterResults
        """
        t = threshold if threshold is not None else self.threshold
        return self._compute_results(
            probs=np.array(probs, dtype=np.float64),
            labels=np.array(labels, dtype=bool),
            threshold=t,
        )

    def _compute_results(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> RouterResults:
        """Compute all router metrics from collected probs and labels."""
        labels_bool = labels.astype(bool)

        # Metrics at default threshold
        primary = compute_metrics_at_threshold(probs, labels_bool, threshold)

        # Threshold sweep
        sweep_results: List[ThresholdResult] = []
        for t in self.threshold_sweep_values:
            sweep_results.append(compute_metrics_at_threshold(probs, labels_bool, t))

        best_sweep = max(sweep_results, key=lambda r: r.f1) if sweep_results else primary

        # ECE
        ece = 0.0
        if self.compute_calibration:
            ece = calibration_error(
                probs.tolist(),
                labels_bool.tolist(),
                num_bins=self.calibration_bins,
            )

        # Error analysis
        error_analysis = self._error_analysis(probs, labels_bool, threshold)

        return RouterResults(
            threshold=threshold,
            accuracy=primary.accuracy,
            precision=primary.precision,
            recall=primary.recall,
            f1=primary.f1,
            specificity=primary.specificity,
            confusion=primary.confusion,
            expected_calibration_error=ece,
            threshold_sweep=sweep_results,
            best_threshold=best_sweep.threshold,
            best_f1=best_sweep.f1,
            error_analysis=error_analysis,
            num_samples=len(probs),
        )

    def _error_analysis(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> ErrorAnalysis:
        """Compute error analysis breakdown."""
        preds = probs >= threshold
        analysis = ErrorAnalysis()

        # Collect FP / FN examples (up to max)
        fp_mask = preds & ~labels
        fn_mask = ~preds & labels

        fp_indices = np.where(fp_mask)[0]
        fn_indices = np.where(fn_mask)[0]

        for idx in fp_indices[: self.max_error_examples]:
            analysis.false_positive_examples.append(
                {"index": int(idx), "prob": float(probs[idx])}
            )
        for idx in fn_indices[: self.max_error_examples]:
            analysis.false_negative_examples.append(
                {"index": int(idx), "prob": float(probs[idx])}
            )

        # Error by position (quartiles of sequence)
        n = len(probs)
        if n > 0:
            positions = np.arange(n) / n
            error_mask = preds != labels
            for quartile_name, lo, hi in [
                ("first_quarter", 0.0, 0.25),
                ("second_quarter", 0.25, 0.5),
                ("third_quarter", 0.5, 0.75),
                ("fourth_quarter", 0.75, 1.0),
            ]:
                in_range = (positions >= lo) & (positions < hi)
                if hi == 1.0:
                    in_range = in_range | (positions == 1.0)
                count_in = int(in_range.sum())
                if count_in > 0:
                    analysis.error_by_position[quartile_name] = float(
                        error_mask[in_range].mean()
                    )

        return analysis


# ===========================================================================
# Visualization helpers
# ===========================================================================


def plot_confusion_matrix(
    cm: ConfusionMatrix,
    title: str = "Router Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> Any:
    """Plot a 2×2 confusion matrix heatmap with matplotlib/seaborn.

    Parameters
    ----------
    cm : ConfusionMatrix
        Computed confusion matrix.
    title : str
        Plot title.
    save_path : str, optional
        If given, save the figure to this path (PNG/PDF).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object (for further customisation).
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning(
            "matplotlib/seaborn not installed — skipping confusion matrix plot"
        )
        return None

    matrix = np.array(cm.to_matrix())  # [[TN, FP], [FN, TP]]
    labels = np.array([
        [f"TN\n{matrix[0, 0]}", f"FP\n{matrix[0, 1]}"],
        [f"FN\n{matrix[1, 0]}", f"TP\n{matrix[1, 1]}"],
    ])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["Predicted\nGeneration", "Predicted\nRetrieval"],
        yticklabels=["Actual\nGeneration", "Actual\nRetrieval"],
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    # Add summary metrics as text below the plot
    summary = (
        f"Accuracy={cm.accuracy:.3f}  Precision={cm.precision:.3f}  "
        f"Recall={cm.recall:.3f}  F1={cm.f1:.3f}"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


def plot_threshold_sweep(
    results: Sequence[ThresholdResult],
    title: str = "Router Threshold Sweep",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> Any:
    """Plot precision, recall, F1, and accuracy across router thresholds.

    Parameters
    ----------
    results : list[ThresholdResult]
        Output from :meth:`RouterEvaluator.threshold_sweep`.
    title : str
        Plot title.
    save_path : str, optional
        If given, save the figure to this path.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping threshold sweep plot")
        return None

    thresholds = [r.threshold for r in results]
    accuracies = [r.accuracy for r in results]
    precisions = [r.precision for r in results]
    recalls = [r.recall for r in results]
    f1s = [r.f1 for r in results]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, accuracies, label="Accuracy", marker=".", linewidth=1.5)
    ax.plot(thresholds, precisions, label="Precision", marker=".", linewidth=1.5)
    ax.plot(thresholds, recalls, label="Recall", marker=".", linewidth=1.5)
    ax.plot(thresholds, f1s, label="F1", marker=".", linewidth=2)

    # Mark the best F1 point
    best_idx = int(np.argmax(f1s))
    ax.axvline(
        thresholds[best_idx],
        color="red",
        linestyle="--",
        alpha=0.6,
        label=f"Best F1={f1s[best_idx]:.3f} @ τ={thresholds[best_idx]:.2f}",
    )

    ax.set_xlabel("Router Threshold (τ)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Threshold sweep plot saved to %s", save_path)

    return fig
