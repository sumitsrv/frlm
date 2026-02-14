"""
Evaluation module.

Metrics for retrieval (P@1, P@5, temporal accuracy), generation
(perplexity, linguistic quality), router (accuracy, confusion matrix),
and end-to-end pipeline evaluation.
"""

from src.evaluation.end_to_end import (
    EndToEndComparison,
    EndToEndEvaluator,
    EndToEndResults,
    FactualAccuracyResult,
    TemporalConsistencyResult,
    compute_factual_accuracy,
    compute_temporal_consistency,
)
from src.evaluation.generation_eval import (
    BaselineComparison,
    GenerationEvaluator,
    GenerationResults,
    PerplexityResult,
    compute_perplexity,
    compute_token_level_loss,
)
from src.evaluation.retrieval_eval import (
    GranularityAccuracyResult,
    PrecisionAtKResult,
    RetrievalEvaluator,
    RetrievalResults,
    TemporalAccuracyResult,
    granularity_accuracy,
    mean_reciprocal_rank,
    precision_at_k,
    temporal_accuracy,
)
from src.evaluation.router_eval import (
    ConfusionMatrix,
    ErrorAnalysis,
    RouterEvaluator,
    RouterResults,
    ThresholdResult,
    calibration_error,
    confusion_matrix,
    confusion_matrix_from_arrays,
    compute_metrics_at_threshold,
)

__all__ = [
    # Retrieval
    "RetrievalEvaluator",
    "RetrievalResults",
    "PrecisionAtKResult",
    "TemporalAccuracyResult",
    "GranularityAccuracyResult",
    "precision_at_k",
    "mean_reciprocal_rank",
    "temporal_accuracy",
    "granularity_accuracy",
    # Generation
    "GenerationEvaluator",
    "GenerationResults",
    "PerplexityResult",
    "BaselineComparison",
    "compute_perplexity",
    "compute_token_level_loss",
    # Router
    "RouterEvaluator",
    "RouterResults",
    "ConfusionMatrix",
    "ThresholdResult",
    "ErrorAnalysis",
    "confusion_matrix",
    "confusion_matrix_from_arrays",
    "calibration_error",
    "compute_metrics_at_threshold",
    # End-to-end
    "EndToEndEvaluator",
    "EndToEndResults",
    "EndToEndComparison",
    "FactualAccuracyResult",
    "TemporalConsistencyResult",
    "compute_factual_accuracy",
    "compute_temporal_consistency",
]