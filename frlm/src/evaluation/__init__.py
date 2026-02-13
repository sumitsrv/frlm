"""
Evaluation module.

Metrics for retrieval (P@1, P@5, temporal accuracy), generation
(perplexity, linguistic quality), router (accuracy, confusion matrix),
and end-to-end pipeline evaluation.
"""

from src.evaluation.retrieval_eval import RetrievalEvaluator
from src.evaluation.generation_eval import GenerationEvaluator
from src.evaluation.router_eval import RouterEvaluator
from src.evaluation.end_to_end import EndToEndEvaluator

__all__ = [
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "RouterEvaluator",
    "EndToEndEvaluator",
]