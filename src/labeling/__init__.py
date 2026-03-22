"""
Labeling module.

LLM-based span labeling using Claude API to classify text spans
as factual-retrieval vs. generation, with quality validation.
Includes a fast heuristic pre-labeler for obvious cases.
"""

from src.labeling.heuristic_labeler import HeuristicLabeler
from src.labeling.llm_labeler import CostTracker, LLMLabeler, SpanLabel
from src.labeling.label_validator import LabelValidator

# Alias per the Phase 5 spec naming convention
RouterLabelGenerator = LLMLabeler

__all__ = [
    "CostTracker",
    "HeuristicLabeler",
    "LLMLabeler",
    "LabelValidator",
    "RouterLabelGenerator",
    "SpanLabel",
]