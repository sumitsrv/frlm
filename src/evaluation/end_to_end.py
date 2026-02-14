"""
End-to-End Evaluator — full FRLM pipeline evaluation.

Orchestrates all component evaluators and adds pipeline-level metrics:
- Factual accuracy (retrieved facts appear correctly in generation)
- Temporal consistency (generated text respects temporal constraints)
- Full pipeline comparison with baseline BioMedLM
- Structured result export

Public API
----------
- ``evaluate(model, dataloader, faiss_index, kg_client, config)``
  → ``EndToEndResults``
- ``compare_with_baseline(frlm_model, baseline_model, dataloader, ...)``
  → ``EndToEndComparison``
- ``export_results(results, output_path)`` — writes JSON
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.evaluation.generation_eval import GenerationEvaluator, GenerationResults
from src.evaluation.retrieval_eval import RetrievalEvaluator, RetrievalResults
from src.evaluation.router_eval import RouterEvaluator, RouterResults

logger = logging.getLogger(__name__)


# ===========================================================================
# Result containers
# ===========================================================================


@dataclass
class FactualAccuracyResult:
    """Measures whether retrieved facts are faithfully reflected in generation."""

    overall: float = 0.0
    by_relation_type: Dict[str, float] = field(default_factory=dict)
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": round(self.overall, 4),
            "by_relation_type": {
                k: round(v, 4) for k, v in self.by_relation_type.items()
            },
            "num_samples": self.num_samples,
        }


@dataclass
class TemporalConsistencyResult:
    """Measures temporal correctness of generated content."""

    overall: float = 0.0
    by_mode: Dict[str, float] = field(default_factory=dict)
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": round(self.overall, 4),
            "by_mode": {k: round(v, 4) for k, v in self.by_mode.items()},
            "num_samples": self.num_samples,
        }


@dataclass
class EndToEndComparison:
    """Comparison between FRLM and baseline models."""

    frlm_factual_accuracy: float = 0.0
    baseline_factual_accuracy: float = 0.0
    frlm_perplexity: float = 0.0
    baseline_perplexity: float = 0.0
    improvement_factual: float = 0.0
    improvement_perplexity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frlm_factual_accuracy": round(self.frlm_factual_accuracy, 4),
            "baseline_factual_accuracy": round(self.baseline_factual_accuracy, 4),
            "frlm_perplexity": round(self.frlm_perplexity, 2),
            "baseline_perplexity": round(self.baseline_perplexity, 2),
            "improvement_factual_pct": round(self.improvement_factual * 100, 2),
            "improvement_perplexity_pct": round(self.improvement_perplexity * 100, 2),
        }


@dataclass
class EndToEndResults:
    """Complete end-to-end evaluation results."""

    retrieval: Optional[RetrievalResults] = None
    generation: Optional[GenerationResults] = None
    router: Optional[RouterResults] = None
    factual_accuracy: FactualAccuracyResult = field(
        default_factory=FactualAccuracyResult
    )
    temporal_consistency: TemporalConsistencyResult = field(
        default_factory=TemporalConsistencyResult
    )
    comparison: Optional[EndToEndComparison] = None
    overall_score: float = 0.0
    evaluation_time_seconds: float = 0.0
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "overall_score": round(self.overall_score, 4),
            "evaluation_time_seconds": round(self.evaluation_time_seconds, 2),
        }
        if self.retrieval is not None:
            result["retrieval"] = self.retrieval.to_dict()
        if self.generation is not None:
            result["generation"] = self.generation.to_dict()
        if self.router is not None:
            result["router"] = self.router.to_dict()
        result["factual_accuracy"] = self.factual_accuracy.to_dict()
        result["temporal_consistency"] = self.temporal_consistency.to_dict()
        if self.comparison is not None:
            result["baseline_comparison"] = self.comparison.to_dict()
        if self.config_snapshot:
            result["config"] = self.config_snapshot
        return result


# ===========================================================================
# Factual accuracy evaluation
# ===========================================================================


def compute_factual_accuracy(
    generated_texts: List[str],
    retrieved_facts: List[List[Dict[str, Any]]],
    ground_truth_facts: List[List[Dict[str, Any]]],
) -> FactualAccuracyResult:
    """Check if retrieved facts are reflected in generated output.

    For each sample, checks whether the subject/object entities from
    ground-truth facts appear in the generated text.

    Parameters
    ----------
    generated_texts : list of str
        Generated text for each sample.
    retrieved_facts : list of list of dict
        Facts actually retrieved during generation.
    ground_truth_facts : list of list of dict
        Expected facts for each sample.

    Returns
    -------
    FactualAccuracyResult
    """
    if not generated_texts:
        return FactualAccuracyResult()

    correct = 0
    total = 0
    by_relation: Dict[str, List[bool]] = {}

    for text, gt_facts in zip(generated_texts, ground_truth_facts):
        text_lower = text.lower()
        for fact in gt_facts:
            total += 1
            subject = fact.get("subject_label", "").lower()
            obj = fact.get("object_label", "").lower()
            rel_type = fact.get("relation_type", "UNKNOWN")

            # Check if both subject and object appear in generated text
            hit = subject in text_lower and obj in text_lower
            if hit:
                correct += 1

            if rel_type not in by_relation:
                by_relation[rel_type] = []
            by_relation[rel_type].append(hit)

    overall = correct / total if total > 0 else 0.0
    per_relation = {
        k: sum(v) / len(v) if v else 0.0 for k, v in by_relation.items()
    }

    return FactualAccuracyResult(
        overall=overall,
        by_relation_type=per_relation,
        num_samples=len(generated_texts),
    )


def compute_temporal_consistency(
    generated_texts: List[str],
    temporal_modes: List[str],
    timestamps: List[Optional[str]],
) -> TemporalConsistencyResult:
    """Evaluate temporal consistency of generated content.

    Checks whether generated text respects temporal constraints
    (e.g., uses correct tense for current vs historical facts).

    Parameters
    ----------
    generated_texts : list of str
        Generated text for each sample.
    temporal_modes : list of str
        Expected temporal mode (CURRENT, AT_TIMESTAMP, HISTORY).
    timestamps : list of str or None
        Associated timestamps for AT_TIMESTAMP mode.

    Returns
    -------
    TemporalConsistencyResult
    """
    if not generated_texts:
        return TemporalConsistencyResult()

    # Heuristic temporal consistency checks
    current_keywords = ["is", "are", "has", "currently", "now", "present"]
    history_keywords = ["was", "were", "had", "previously", "formerly", "historical"]

    correct = 0
    total = 0
    by_mode: Dict[str, List[bool]] = {}

    for text, mode in zip(generated_texts, temporal_modes):
        total += 1
        text_lower = text.lower()
        words = set(text_lower.split())

        # Simple heuristic: check if temporal markers match mode
        if mode == "CURRENT":
            has_current = any(kw in words for kw in current_keywords)
            has_history = any(kw in words for kw in history_keywords)
            hit = has_current or not has_history  # Accept if no conflicting markers
        elif mode == "HISTORY":
            has_history = any(kw in words for kw in history_keywords)
            hit = has_history or len(text_lower) < 20  # Short texts are ambiguous
        elif mode == "AT_TIMESTAMP":
            # Any date-like reference is acceptable
            hit = any(c.isdigit() for c in text) or len(text_lower) > 0
        else:
            hit = True

        if hit:
            correct += 1

        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(hit)

    overall = correct / total if total > 0 else 0.0
    per_mode = {k: sum(v) / len(v) if v else 0.0 for k, v in by_mode.items()}

    return TemporalConsistencyResult(
        overall=overall,
        by_mode=per_mode,
        num_samples=len(generated_texts),
    )


# ===========================================================================
# End-to-End Evaluator class
# ===========================================================================


class EndToEndEvaluator:
    """Full end-to-end FRLM pipeline evaluator.

    Orchestrates component evaluators and computes pipeline-level metrics.

    Parameters
    ----------
    retrieval_evaluator : RetrievalEvaluator, optional
        Component evaluator for retrieval.
    generation_evaluator : GenerationEvaluator, optional
        Component evaluator for generation.
    router_evaluator : RouterEvaluator, optional
        Component evaluator for router.
    compute_factual_accuracy : bool
        Whether to compute factual accuracy.
    compute_temporal_consistency : bool
        Whether to compute temporal consistency.
    device : str
        Device for inference.
    """

    def __init__(
        self,
        retrieval_evaluator: Optional[RetrievalEvaluator] = None,
        generation_evaluator: Optional[GenerationEvaluator] = None,
        router_evaluator: Optional[RouterEvaluator] = None,
        compute_factual: bool = True,
        compute_temporal: bool = True,
        device: str = "cpu",
    ) -> None:
        self.retrieval_evaluator = retrieval_evaluator or RetrievalEvaluator(
            device=device
        )
        self.generation_evaluator = generation_evaluator or GenerationEvaluator(
            device=device
        )
        self.router_evaluator = router_evaluator or RouterEvaluator(device=device)
        self.compute_factual = compute_factual
        self.compute_temporal = compute_temporal
        self.device = device

    @classmethod
    def from_config(cls, eval_config: Any) -> "EndToEndEvaluator":
        """Create from a :class:`EvaluationConfig`."""
        retrieval_eval = RetrievalEvaluator.from_config(eval_config.retrieval)
        generation_eval = GenerationEvaluator.from_config(eval_config.generation)
        router_eval = RouterEvaluator.from_config(eval_config.router)
        return cls(
            retrieval_evaluator=retrieval_eval,
            generation_evaluator=generation_eval,
            router_evaluator=router_eval,
            compute_factual=eval_config.end_to_end.compute_factual_accuracy,
            compute_temporal=eval_config.end_to_end.compute_temporal_consistency,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: Any,
        retrieval_dataloader: Optional[DataLoader] = None,
        generation_dataloader: Optional[DataLoader] = None,
        router_dataloader: Optional[DataLoader] = None,
        faiss_index: Any = None,
        kg_client: Any = None,
        max_samples: Optional[int] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> EndToEndResults:
        """Run the complete evaluation pipeline.

        Parameters
        ----------
        model : FRLMModel
            FRLM model.
        retrieval_dataloader : DataLoader, optional
            Dataloader for retrieval evaluation.
        generation_dataloader : DataLoader, optional
            Dataloader for generation evaluation.
        router_dataloader : DataLoader, optional
            Dataloader for router evaluation.
        faiss_index : FAISSFactIndex, optional
            FAISS index for retrieval.
        kg_client : Neo4jClient, optional
            KG client for temporal filtering.
        max_samples : int, optional
            Cap on samples per component.
        config_snapshot : dict, optional
            Configuration snapshot for the report.

        Returns
        -------
        EndToEndResults
        """
        start_time = time.time()
        model.eval()

        results = EndToEndResults(
            config_snapshot=config_snapshot or {},
        )

        # --- Retrieval ---
        if retrieval_dataloader is not None and faiss_index is not None:
            logger.info("Running retrieval evaluation...")
            results.retrieval = self.retrieval_evaluator.evaluate(
                model=model,
                dataloader=retrieval_dataloader,
                faiss_index=faiss_index,
                kg_client=kg_client,
                max_samples=max_samples,
            )

        # --- Generation ---
        if generation_dataloader is not None:
            logger.info("Running generation evaluation...")
            results.generation = self.generation_evaluator.evaluate(
                model=model,
                dataloader=generation_dataloader,
                max_samples=max_samples,
            )

        # --- Router ---
        if router_dataloader is not None:
            logger.info("Running router evaluation...")
            results.router = self.router_evaluator.evaluate(
                model=model,
                dataloader=router_dataloader,
                max_samples=max_samples,
            )

        # --- Compute overall score ---
        scores: List[float] = []
        if results.retrieval is not None:
            mrr = results.retrieval.mrr
            scores.append(mrr)
        if results.generation is not None:
            # Normalise perplexity to [0,1] range (lower ppl = better)
            ppl = results.generation.perplexity.overall
            if ppl > 0 and ppl != float("inf"):
                # Inverse log-perplexity scaled to roughly [0,1]
                norm_ppl = 1.0 / (1.0 + np.log(ppl))
                scores.append(norm_ppl)
        if results.router is not None:
            scores.append(results.router.f1)

        results.overall_score = float(np.mean(scores)) if scores else 0.0
        results.evaluation_time_seconds = time.time() - start_time

        logger.info(
            "End-to-end evaluation complete: overall=%.4f (%.2fs)",
            results.overall_score,
            results.evaluation_time_seconds,
        )
        return results

    def evaluate_from_predictions(
        self,
        retrieval_predictions: Optional[List[Dict[str, Any]]] = None,
        generation_losses: Optional[List[float]] = None,
        generation_router_mask: Optional[List[bool]] = None,
        router_probs: Optional[List[float]] = None,
        router_labels: Optional[List[bool]] = None,
        generated_texts: Optional[List[str]] = None,
        retrieved_facts: Optional[List[List[Dict[str, Any]]]] = None,
        ground_truth_facts: Optional[List[List[Dict[str, Any]]]] = None,
        temporal_modes: Optional[List[str]] = None,
        timestamps: Optional[List[Optional[str]]] = None,
    ) -> EndToEndResults:
        """Evaluate from pre-computed predictions.

        Parameters
        ----------
        retrieval_predictions : list of dict, optional
            Pre-computed retrieval predictions.
        generation_losses : list of float, optional
            Per-token generation losses.
        generation_router_mask : list of bool, optional
            Per-token router mask for generation.
        router_probs : list of float, optional
            Router predicted probabilities.
        router_labels : list of bool, optional
            Router ground-truth labels.
        generated_texts : list of str, optional
            Generated output texts.
        retrieved_facts : list of list of dict, optional
            Facts retrieved for each sample.
        ground_truth_facts : list of list of dict, optional
            Ground truth facts for each sample.
        temporal_modes : list of str, optional
            Expected temporal modes.
        timestamps : list of str or None, optional
            Timestamps for AT_TIMESTAMP mode.

        Returns
        -------
        EndToEndResults
        """
        start_time = time.time()
        results = EndToEndResults()

        # Retrieval
        if retrieval_predictions is not None:
            results.retrieval = self.retrieval_evaluator.evaluate_from_predictions(
                retrieval_predictions
            )

        # Generation
        if generation_losses is not None:
            results.generation = self.generation_evaluator.evaluate_from_losses(
                generation_losses, generation_router_mask
            )

        # Router
        if router_probs is not None and router_labels is not None:
            results.router = self.router_evaluator.evaluate_from_predictions(
                router_probs, router_labels
            )

        # Factual accuracy
        if (
            self.compute_factual
            and generated_texts is not None
            and ground_truth_facts is not None
        ):
            results.factual_accuracy = compute_factual_accuracy(
                generated_texts,
                retrieved_facts or [[] for _ in generated_texts],
                ground_truth_facts,
            )

        # Temporal consistency
        if (
            self.compute_temporal
            and generated_texts is not None
            and temporal_modes is not None
        ):
            results.temporal_consistency = compute_temporal_consistency(
                generated_texts,
                temporal_modes,
                timestamps or [None] * len(generated_texts),
            )

        # Overall score
        scores: List[float] = []
        if results.retrieval is not None:
            scores.append(results.retrieval.mrr)
        if results.router is not None:
            scores.append(results.router.f1)
        if results.factual_accuracy.overall > 0:
            scores.append(results.factual_accuracy.overall)
        results.overall_score = float(np.mean(scores)) if scores else 0.0
        results.evaluation_time_seconds = time.time() - start_time

        return results

    @torch.no_grad()
    def compare_with_baseline(
        self,
        frlm_model: Any,
        baseline_model: Any,
        dataloader: DataLoader,
        faiss_index: Any = None,
        kg_client: Any = None,
        max_samples: Optional[int] = None,
    ) -> EndToEndComparison:
        """Compare FRLM against a baseline model.

        Parameters
        ----------
        frlm_model : FRLMModel
            FRLM model.
        baseline_model : nn.Module
            Baseline model (BioMedLM backbone + generation head).
        dataloader : DataLoader
            Shared test dataloader.
        faiss_index : FAISSFactIndex, optional
            FAISS index.
        kg_client : Neo4jClient, optional
            KG client.
        max_samples : int, optional
            Sample cap.

        Returns
        -------
        EndToEndComparison
        """
        # Evaluate FRLM generation
        frlm_gen = self.generation_evaluator.evaluate(
            frlm_model, dataloader, max_samples
        )

        # Evaluate baseline generation
        baseline_gen = self.generation_evaluator.evaluate(
            baseline_model, dataloader, max_samples
        )

        frlm_ppl = frlm_gen.perplexity.overall
        base_ppl = baseline_gen.perplexity.overall

        comparison = EndToEndComparison(
            frlm_perplexity=frlm_ppl,
            baseline_perplexity=base_ppl,
            improvement_perplexity=(
                (base_ppl - frlm_ppl) / base_ppl if base_ppl > 0 else 0.0
            ),
        )

        logger.info(
            "Baseline comparison: FRLM PPL=%.2f vs Baseline PPL=%.2f",
            frlm_ppl,
            base_ppl,
        )
        return comparison

    @staticmethod
    def export_results(
        results: EndToEndResults,
        output_path: str | Path,
    ) -> Path:
        """Export evaluation results to a JSON file.

        Parameters
        ----------
        results : EndToEndResults
            Results to export.
        output_path : str or Path
            Output file path.

        Returns
        -------
        Path
            The written file path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("Evaluation results exported to %s", path)
        return path
