"""
Extraction module.

Entity extraction via SciSpacy NER with UMLS linking,
relation extraction via Claude API, and PMC XML corpus loading.
"""

from src.extraction.entity_extractor import EntityExtractor
from src.extraction.relation_extractor import RelationExtractor
from src.extraction.corpus_loader import CorpusLoader

__all__ = [
    "EntityExtractor",
    "RelationExtractor",
    "CorpusLoader",
]