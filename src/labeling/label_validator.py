"""
Label quality validator for FRLM router labels.

Provides statistics, confidence filtering, inter-annotator agreement
(Cohen's kappa), and export to Label Studio format for human review.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.labeling.llm_labeler import SpanLabel

logger = logging.getLogger(__name__)


class LabelValidator:
    """Validate and analyse router span labels.

    All methods are stateless — pass label data in, get results out.
    Configuration thresholds (retrieval ratio bounds, etc.) are passed
    where needed, typically from ``LabelingConfig.validation``.
    """

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_statistics(
        labels: List[SpanLabel],
    ) -> Dict[str, Any]:
        """Compute descriptive statistics over a set of span labels.

        Parameters
        ----------
        labels : List[SpanLabel]
            Span-level labels to analyse.

        Returns
        -------
        Dict[str, Any]
            Keys:

            * ``total_spans`` – number of spans.
            * ``class_distribution`` – counts per label.
            * ``class_ratio`` – ratio per label.
            * ``avg_confidence`` – mean confidence across all spans.
            * ``avg_confidence_per_class`` – mean confidence by label.
            * ``span_length_stats`` – min / max / mean / median char lengths.
            * ``span_length_per_class`` – same, broken down by label.
        """
        if not labels:
            return {
                "total_spans": 0,
                "class_distribution": {},
                "class_ratio": {},
                "avg_confidence": 0.0,
                "avg_confidence_per_class": {},
                "span_length_stats": {},
                "span_length_per_class": {},
            }

        total = len(labels)

        # Class distribution
        class_counts: Counter[str] = Counter(s.label for s in labels)
        class_ratio = {k: round(v / total, 4) for k, v in class_counts.items()}

        # Confidence
        avg_conf = sum(s.confidence for s in labels) / total
        conf_by_class: Dict[str, List[float]] = {}
        for s in labels:
            conf_by_class.setdefault(s.label, []).append(s.confidence)
        avg_conf_per_class = {
            k: round(sum(v) / len(v), 4) for k, v in conf_by_class.items()
        }

        # Span lengths
        lengths = [s.end_char - s.start_char for s in labels]
        length_stats = _length_stats(lengths)

        lengths_by_class: Dict[str, List[int]] = {}
        for s in labels:
            lengths_by_class.setdefault(s.label, []).append(
                s.end_char - s.start_char
            )
        span_length_per_class = {
            k: _length_stats(v) for k, v in lengths_by_class.items()
        }

        return {
            "total_spans": total,
            "class_distribution": dict(class_counts),
            "class_ratio": class_ratio,
            "avg_confidence": round(avg_conf, 4),
            "avg_confidence_per_class": avg_conf_per_class,
            "span_length_stats": length_stats,
            "span_length_per_class": span_length_per_class,
        }

    # ------------------------------------------------------------------
    # Low-confidence filtering
    # ------------------------------------------------------------------

    @staticmethod
    def find_low_confidence(
        labels: List[SpanLabel],
        threshold: float = 0.7,
    ) -> List[SpanLabel]:
        """Return spans whose confidence is below *threshold*.

        Parameters
        ----------
        labels : List[SpanLabel]
            Span labels to filter.
        threshold : float
            Confidence cut-off (exclusive).

        Returns
        -------
        List[SpanLabel]
            Low-confidence spans sorted by ascending confidence.
        """
        low = [s for s in labels if s.confidence < threshold]
        low.sort(key=lambda s: s.confidence)
        return low

    # ------------------------------------------------------------------
    # Inter-annotator agreement
    # ------------------------------------------------------------------

    @staticmethod
    def inter_annotator_agreement(
        llm_labels: List[SpanLabel],
        human_labels: List[SpanLabel],
        text_length: int,
    ) -> Dict[str, Any]:
        """Compute Cohen's kappa between two label sets at character level.

        Both label sets must refer to the same text.

        Parameters
        ----------
        llm_labels : List[SpanLabel]
            Machine-generated labels.
        human_labels : List[SpanLabel]
            Human-annotated labels.
        text_length : int
            Length (in characters) of the original text.

        Returns
        -------
        Dict[str, Any]
            ``kappa``, ``observed_agreement``, ``expected_agreement``,
            ``confusion_matrix`` (TP/FP/FN/TN at char level where
            positive = factual).
        """
        if text_length == 0:
            return {
                "kappa": 0.0,
                "observed_agreement": 0.0,
                "expected_agreement": 0.0,
                "confusion_matrix": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
            }

        arr_a = _spans_to_char_array(llm_labels, text_length)
        arr_b = _spans_to_char_array(human_labels, text_length)

        n = text_length
        tp = fn = fp = tn = 0
        for a, b in zip(arr_a, arr_b):
            if a == 1 and b == 1:
                tp += 1
            elif a == 1 and b == 0:
                fp += 1
            elif a == 0 and b == 1:
                fn += 1
            else:
                tn += 1

        agreement = tp + tn
        p_o = agreement / n

        p_a1 = sum(arr_a) / n
        p_b1 = sum(arr_b) / n
        p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)

        if abs(1 - p_e) < 1e-10:
            kappa = 1.0 if abs(p_o - 1.0) < 1e-10 else 0.0
        else:
            kappa = (p_o - p_e) / (1 - p_e)

        return {
            "kappa": round(kappa, 4),
            "observed_agreement": round(p_o, 4),
            "expected_agreement": round(p_e, 4),
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }

    # ------------------------------------------------------------------
    # Validation against configured thresholds
    # ------------------------------------------------------------------

    @staticmethod
    def validate_labels(
        labels: List[SpanLabel],
        text_length: int,
        *,
        min_retrieval_ratio: float = 0.15,
        max_retrieval_ratio: float = 0.85,
        min_spans_per_chunk: int = 1,
        max_spans_per_chunk: int = 10000,
    ) -> Tuple[bool, List[str]]:
        """Validate labels against quality thresholds.

        Parameters
        ----------
        labels : List[SpanLabel]
            Span labels to validate.
        text_length : int
            Character length of the source text.
        min_retrieval_ratio, max_retrieval_ratio : float
            Acceptable range for proportion of characters labelled factual.
        min_spans_per_chunk, max_spans_per_chunk : int
            Acceptable span count range.

        Returns
        -------
        Tuple[bool, List[str]]
            ``(is_valid, issues)`` where *issues* is empty on success.
        """
        issues: List[str] = []

        if not labels:
            issues.append("No labels produced")
            return False, issues

        n_spans = len(labels)
        if n_spans < min_spans_per_chunk:
            issues.append(
                f"Too few spans: {n_spans} < {min_spans_per_chunk}"
            )
        if n_spans > max_spans_per_chunk:
            issues.append(
                f"Too many spans: {n_spans} > {max_spans_per_chunk}"
            )

        if text_length > 0:
            factual_chars = sum(
                (s.end_char - s.start_char) for s in labels if s.label == "factual"
            )
            ratio = factual_chars / text_length
            if ratio < min_retrieval_ratio:
                issues.append(
                    f"Retrieval ratio too low: {ratio:.3f} < {min_retrieval_ratio}"
                )
            if ratio > max_retrieval_ratio:
                issues.append(
                    f"Retrieval ratio too high: {ratio:.3f} > {max_retrieval_ratio}"
                )

        return (len(issues) == 0), issues

    # ------------------------------------------------------------------
    # Export for human review (Label Studio JSON format)
    # ------------------------------------------------------------------

    @staticmethod
    def export_for_review(
        text: str,
        labels: List[SpanLabel],
        output_path: str | Path,
        *,
        source_id: str = "",
    ) -> Path:
        """Export labelled text in Label Studio JSON format.

        Label Studio expects an array of *task* objects.  Each task has:
        ``data.text`` and ``annotations[0].result[]`` with NER-style
        spans.

        Parameters
        ----------
        text : str
            Original text.
        labels : List[SpanLabel]
            Span labels.
        output_path : str | Path
            Destination file path.
        source_id : str
            Optional identifier for the text.

        Returns
        -------
        Path
            The written file path.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        results = []
        for idx, span in enumerate(labels):
            results.append(
                {
                    "id": f"span_{idx}",
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": span.start_char,
                        "end": span.end_char,
                        "text": span.text,
                        "labels": [span.label],
                    },
                    "score": span.confidence,
                }
            )

        task = {
            "data": {"text": text, "source_id": source_id},
            "annotations": [{"result": results}],
        }

        with open(out, "w", encoding="utf-8") as fh:
            json.dump([task], fh, indent=2, ensure_ascii=False)

        logger.info("Exported %d spans to %s", len(labels), out)
        return out

    # ------------------------------------------------------------------
    # Batch export: many texts → one LS file
    # ------------------------------------------------------------------

    @staticmethod
    def export_corpus_for_review(
        records: List[Dict[str, Any]],
        output_path: str | Path,
    ) -> Path:
        """Export multiple labelled texts into a single Label Studio file.

        Parameters
        ----------
        records : List[Dict]
            Each dict must have keys ``"text"`` and ``"spans"``
            (list of ``SpanLabel`` dicts or ``SpanLabel`` objects).
        output_path : str | Path
            Destination file.

        Returns
        -------
        Path
            The written file path.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        tasks = []
        for rec_idx, rec in enumerate(records):
            text = rec.get("text", "")
            raw_spans = rec.get("spans", [])
            results = []
            for s_idx, s in enumerate(raw_spans):
                if isinstance(s, SpanLabel):
                    span = s
                elif isinstance(s, dict):
                    span = SpanLabel(**s)
                else:
                    continue
                results.append(
                    {
                        "id": f"span_{rec_idx}_{s_idx}",
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": span.start_char,
                            "end": span.end_char,
                            "text": span.text,
                            "labels": [span.label],
                        },
                        "score": span.confidence,
                    }
                )
            tasks.append(
                {
                    "data": {
                        "text": text,
                        "source_id": rec.get("source_id", str(rec_idx)),
                    },
                    "annotations": [{"result": results}],
                }
            )

        with open(out, "w", encoding="utf-8") as fh:
            json.dump(tasks, fh, indent=2, ensure_ascii=False)

        logger.info(
            "Exported %d tasks (%d total spans) to %s",
            len(tasks),
            sum(len(t["annotations"][0]["result"]) for t in tasks),
            out,
        )
        return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _spans_to_char_array(spans: List[SpanLabel], length: int) -> List[int]:
    """Convert spans to a character-level binary array (1 = factual)."""
    arr = [0] * length
    for s in spans:
        if s.label == "factual":
            for i in range(s.start_char, min(s.end_char, length)):
                arr[i] = 1
    return arr


def _length_stats(lengths: List[int]) -> Dict[str, float]:
    """Compute min / max / mean / median for a list of integers."""
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    sorted_l = sorted(lengths)
    n = len(sorted_l)
    median = (
        sorted_l[n // 2]
        if n % 2 == 1
        else (sorted_l[n // 2 - 1] + sorted_l[n // 2]) / 2.0
    )
    return {
        "min": sorted_l[0],
        "max": sorted_l[-1],
        "mean": round(sum(sorted_l) / n, 2),
        "median": round(float(median), 2),
    }
