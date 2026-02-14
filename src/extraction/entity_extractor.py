"""
Entity Extractor — SciSpacy NER with UMLS linking for FRLM.

Provides:
    - EntityExtractor: Extract biomedical entities from text using SciSpacy
    - UMLS concept linking with confidence thresholds
    - Entity type classification (drug, gene, disease, protein, etc.)
    - Batch processing and deduplication

Handles entities that don't link to UMLS by generating content-hash fallback IDs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ===========================================================================
# Entity type mapping
# ===========================================================================

# Map UMLS semantic types to our simplified entity types
UMLS_SEMANTIC_TYPE_MAP = {
    # Drugs / Chemicals
    "T116": "protein",           # Amino Acid, Peptide, or Protein
    "T118": "drug",              # Carbohydrate
    "T119": "drug",              # Lipid
    "T120": "drug",              # Chemical Viewed Functionally
    "T121": "drug",              # Pharmacologic Substance
    "T122": "drug",              # Biomedical or Dental Material
    "T123": "drug",              # Biologically Active Substance
    "T124": "drug",              # Neuroreactive Substance or Biogenic Amine
    "T125": "drug",              # Hormone
    "T126": "drug",              # Enzyme
    "T127": "drug",              # Vitamin
    "T129": "drug",              # Immunologic Factor
    "T130": "drug",              # Indicator, Reagent, or Diagnostic Aid
    "T131": "drug",              # Hazardous or Poisonous Substance
    "T195": "drug",              # Antibiotic
    "T196": "drug",              # Element, Ion, or Isotope
    "T197": "drug",              # Inorganic Chemical
    "T200": "drug",              # Clinical Drug

    # Genes / Molecular Biology
    "T028": "gene",              # Gene or Genome
    "T085": "gene",              # Molecular Sequence
    "T086": "gene",              # Nucleotide Sequence
    "T087": "gene",              # Amino Acid Sequence
    "T088": "gene",              # Carbohydrate Sequence
    "T114": "gene",              # Nucleic Acid, Nucleoside, or Nucleotide

    # Proteins
    "T087": "protein",           # Amino Acid Sequence (also gene-related)
    "T116": "protein",           # Amino Acid, Peptide, or Protein

    # Diseases / Disorders
    "T019": "disease",           # Congenital Abnormality
    "T020": "disease",           # Acquired Abnormality
    "T033": "disease",           # Finding
    "T037": "disease",           # Injury or Poisoning
    "T046": "disease",           # Pathologic Function
    "T047": "disease",           # Disease or Syndrome
    "T048": "disease",           # Mental or Behavioral Dysfunction
    "T049": "disease",           # Cell or Molecular Dysfunction
    "T184": "disease",           # Sign or Symptom
    "T190": "disease",           # Anatomical Abnormality
    "T191": "disease",           # Neoplastic Process

    # Anatomical structures
    "T017": "anatomical_structure",   # Anatomical Structure
    "T018": "anatomical_structure",   # Embryonic Structure
    "T021": "anatomical_structure",   # Fully Formed Anatomical Structure
    "T022": "anatomical_structure",   # Body System
    "T023": "anatomical_structure",   # Body Part, Organ, or Organ Component
    "T024": "anatomical_structure",   # Tissue
    "T025": "cell_type",              # Cell
    "T026": "cell_type",              # Cell Component
    "T029": "anatomical_structure",   # Body Location or Region
    "T030": "anatomical_structure",   # Body Space or Junction

    # Cell types
    "T025": "cell_type",              # Cell

    # Organisms
    "T001": "organism",          # Organism
    "T002": "organism",          # Plant
    "T004": "organism",          # Fungus
    "T005": "organism",          # Virus
    "T007": "organism",          # Bacterium
    "T008": "organism",          # Animal
    "T010": "organism",          # Vertebrate
    "T011": "organism",          # Amphibian
    "T012": "organism",          # Bird
    "T013": "organism",          # Fish
    "T014": "organism",          # Reptile
    "T015": "organism",          # Mammal
    "T016": "organism",          # Human

    # Pathways / Biological processes
    "T038": "pathway",           # Biologic Function
    "T039": "pathway",           # Physiologic Function
    "T040": "pathway",           # Organism Function
    "T041": "pathway",           # Mental Process
    "T042": "pathway",           # Organ or Tissue Function
    "T043": "pathway",           # Cell Function
    "T044": "pathway",           # Molecular Function
    "T045": "pathway",           # Genetic Function
}

# Default entity type when semantic type not found
DEFAULT_ENTITY_TYPE = "biomedical_entity"

# Valid entity types for FRLM
VALID_ENTITY_TYPES = {
    "drug",
    "gene",
    "disease",
    "protein",
    "pathway",
    "cell_type",
    "organism",
    "anatomical_structure",
    "biomedical_entity",
}


# ===========================================================================
# Import SciSpacy lazily to avoid startup overhead
# ===========================================================================


def _get_spacy_model(model_name: str = "en_core_sci_lg"):
    """Load SciSpacy model lazily."""
    try:
        import spacy
        return spacy.load(model_name)
    except OSError:
        logger.error(
            "SciSpacy model '%s' not found. Install with: "
            "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz",
            model_name,
        )
        raise


def _get_umls_linker():
    """Load UMLS entity linker lazily."""
    try:
        from scispacy.linking import EntityLinker
        return EntityLinker
    except ImportError:
        logger.error(
            "scispacy not properly installed. Install with: pip install scispacy"
        )
        raise


# ===========================================================================
# Entity data class
# ===========================================================================


@dataclass
class ExtractedEntity:
    """A biomedical entity extracted from text.

    Attributes
    ----------
    text : str
        Original text span.
    start_char : int
        Start character offset in source text.
    end_char : int
        End character offset in source text.
    label : str
        Canonical label (preferred name).
    entity_type : str
        Semantic type (drug, gene, disease, etc.).
    canonical_id : str
        Ontology identifier (UMLS CUI or content hash).
    source_ontology : str
        Source of the canonical_id ("UMLS" or "CONTENT_HASH").
    confidence : float
        UMLS linking confidence score.
    umls_semantic_types : List[str]
        UMLS semantic type identifiers.
    aliases : List[str]
        Alternative names for the entity.
    definition : str
        UMLS definition if available.
    """

    text: str
    start_char: int
    end_char: int
    label: str
    entity_type: str
    canonical_id: str
    source_ontology: str = "UMLS"
    confidence: float = 1.0
    umls_semantic_types: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    definition: str = ""

    @property
    def is_umls_linked(self) -> bool:
        """Return True if entity was linked to UMLS."""
        return self.source_ontology == "UMLS"

    def to_biomedical_entity(self):
        """Convert to BiomedicalEntity schema object."""
        from src.kg.schema import BiomedicalEntity
        return BiomedicalEntity(
            id=self.canonical_id,
            label=self.label,
            entity_type=self.entity_type,
            canonical_id=self.canonical_id,
            source_ontology=self.source_ontology,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "label": self.label,
            "entity_type": self.entity_type,
            "canonical_id": self.canonical_id,
            "source_ontology": self.source_ontology,
            "confidence": self.confidence,
            "umls_semantic_types": self.umls_semantic_types,
            "aliases": self.aliases,
            "definition": self.definition,
        }


# ===========================================================================
# Entity Extractor
# ===========================================================================


class EntityExtractor:
    """Extract biomedical entities from text using SciSpacy + UMLS linking.

    Parameters
    ----------
    spacy_model : str
        SciSpacy model name (default: "en_core_sci_lg").
    linker : str
        Entity linker type (default: "umls").
    resolve_abbreviations : bool
        Whether to resolve abbreviations.
    min_entity_length : int
        Minimum character length for entities.
    max_entity_length : int
        Maximum character length for entities.
    confidence_threshold : float
        Minimum UMLS linking confidence to accept.
    batch_size : int
        Batch size for processing.
    n_process : int
        Number of parallel processes for batch processing.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_sci_lg",
        linker: str = "umls",
        resolve_abbreviations: bool = True,
        min_entity_length: int = 2,
        max_entity_length: int = 100,
        confidence_threshold: float = 0.7,
        batch_size: int = 128,
        n_process: int = 4,
    ):
        self.spacy_model_name = spacy_model
        self.linker_type = linker
        self.resolve_abbreviations = resolve_abbreviations
        self.min_entity_length = min_entity_length
        self.max_entity_length = max_entity_length
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.n_process = n_process

        # Lazy initialization
        self._nlp = None
        self._linker = None

        # Statistics
        self._stats: Dict[str, int] = {
            "texts_processed": 0,
            "entities_extracted": 0,
            "umls_linked": 0,
            "fallback_ids": 0,
            "filtered_low_confidence": 0,
            "filtered_length": 0,
        }

        logger.info(
            "EntityExtractor initialized: model=%s, linker=%s, "
            "confidence_threshold=%.2f",
            spacy_model,
            linker,
            confidence_threshold,
        )

    def _ensure_model_loaded(self) -> None:
        """Load SciSpacy model and UMLS linker if not already loaded."""
        if self._nlp is not None:
            return

        import spacy

        logger.info("Loading SciSpacy model: %s", self.spacy_model_name)
        self._nlp = _get_spacy_model(self.spacy_model_name)

        # Add UMLS linker
        if self.linker_type == "umls":
            if "scispacy_linker" not in self._nlp.pipe_names:
                logger.info("Adding UMLS entity linker...")
                self._nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": self.resolve_abbreviations,
                        "linker_name": "umls",
                    },
                )
            self._linker = self._nlp.get_pipe("scispacy_linker")

        # Add abbreviation detector if needed
        if self.resolve_abbreviations and "abbreviation_detector" not in self._nlp.pipe_names:
            try:
                self._nlp.add_pipe("abbreviation_detector")
            except Exception as e:
                logger.warning("Could not add abbreviation detector: %s", e)

        logger.info("SciSpacy model loaded with %d pipes", len(self._nlp.pipe_names))

    @property
    def stats(self) -> Dict[str, int]:
        """Return extraction statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset extraction statistics."""
        for key in self._stats:
            self._stats[key] = 0

    def _compute_content_hash_id(self, text: str) -> str:
        """Generate a content-based hash ID for entities without UMLS links.

        Parameters
        ----------
        text : str
            Entity text to hash.

        Returns
        -------
        str
            Content hash ID prefixed with "HASH:".
        """
        # Normalize text: lowercase, strip whitespace, remove punctuation
        normalized = re.sub(r"[^\w\s]", "", text.lower().strip())
        normalized = re.sub(r"\s+", "_", normalized)

        hash_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"HASH:{hash_digest}"

    def _infer_entity_type(
        self,
        umls_semantic_types: List[str],
        text: str,
    ) -> str:
        """Infer entity type from UMLS semantic types or text heuristics.

        Parameters
        ----------
        umls_semantic_types : List[str]
            List of UMLS semantic type identifiers (e.g., "T047").
        text : str
            Entity text for heuristic fallback.

        Returns
        -------
        str
            Inferred entity type.
        """
        # Check semantic types
        for st in umls_semantic_types:
            if st in UMLS_SEMANTIC_TYPE_MAP:
                return UMLS_SEMANTIC_TYPE_MAP[st]

        # Heuristic fallback based on text patterns
        text_lower = text.lower()

        # Drug patterns
        if any(suffix in text_lower for suffix in ["mab", "nib", "vir", "ine", "cin"]):
            return "drug"

        # Gene patterns (all caps, numbers)
        if re.match(r"^[A-Z0-9]{2,10}$", text):
            return "gene"

        # Disease patterns
        if any(word in text_lower for word in ["cancer", "tumor", "syndrome", "disease", "disorder"]):
            return "disease"

        return DEFAULT_ENTITY_TYPE

    def _process_spacy_entity(
        self,
        ent,
        kb_ents: Optional[List] = None,
    ) -> Optional[ExtractedEntity]:
        """Process a single SpaCy entity into an ExtractedEntity.

        Parameters
        ----------
        ent : spacy.tokens.Span
            SpaCy entity span.
        kb_ents : List, optional
            UMLS KB entity matches from the linker.

        Returns
        -------
        ExtractedEntity or None
            Processed entity or None if filtered out.
        """
        text = ent.text.strip()

        # Length filter
        if len(text) < self.min_entity_length or len(text) > self.max_entity_length:
            self._stats["filtered_length"] += 1
            return None

        # Get UMLS linking info
        canonical_id = ""
        label = text
        source_ontology = "CONTENT_HASH"
        confidence = 0.0
        umls_semantic_types = []
        aliases = []
        definition = ""

        if kb_ents:
            # Get best UMLS match
            best_match = kb_ents[0] if kb_ents else None
            if best_match:
                cui, score = best_match

                if score >= self.confidence_threshold:
                    canonical_id = cui
                    confidence = score
                    source_ontology = "UMLS"

                    # Get entity info from UMLS KB
                    if self._linker is not None:
                        kb = self._linker.kb
                        if cui in kb:
                            entity_info = kb[cui]
                            label = entity_info.canonical_name or text
                            umls_semantic_types = list(entity_info.types or [])
                            aliases = list(entity_info.aliases or [])[:10]
                            definition = entity_info.definition or ""

                    self._stats["umls_linked"] += 1
                else:
                    self._stats["filtered_low_confidence"] += 1

        # Fallback to content hash ID
        if not canonical_id:
            canonical_id = self._compute_content_hash_id(text)
            source_ontology = "CONTENT_HASH"
            confidence = 1.0  # High confidence in the hash itself
            self._stats["fallback_ids"] += 1
            logger.debug("Entity not linked to UMLS, using content hash: %s -> %s", text, canonical_id)

        # Infer entity type
        entity_type = self._infer_entity_type(umls_semantic_types, text)

        return ExtractedEntity(
            text=text,
            start_char=ent.start_char,
            end_char=ent.end_char,
            label=label,
            entity_type=entity_type,
            canonical_id=canonical_id,
            source_ontology=source_ontology,
            confidence=confidence,
            umls_semantic_types=umls_semantic_types,
            aliases=aliases,
            definition=definition,
        )

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract biomedical entities from text.

        Parameters
        ----------
        text : str
            Input text to process.

        Returns
        -------
        List[ExtractedEntity]
            List of extracted entities.
        """
        self._ensure_model_loaded()

        if not text or not text.strip():
            return []

        self._stats["texts_processed"] += 1

        # Process with SpaCy
        doc = self._nlp(text)

        entities = []
        for ent in doc.ents:
            # Get KB entities if available
            kb_ents = getattr(ent._, "kb_ents", None)

            entity = self._process_spacy_entity(ent, kb_ents)
            if entity is not None:
                entities.append(entity)
                self._stats["entities_extracted"] += 1

        return entities

    def extract_entities_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[List[ExtractedEntity]]:
        """Extract entities from multiple texts in batch.

        Parameters
        ----------
        texts : List[str]
            List of texts to process.
        show_progress : bool
            Whether to log progress.

        Returns
        -------
        List[List[ExtractedEntity]]
            List of entity lists, one per input text.
        """
        self._ensure_model_loaded()

        results = []

        for idx, text in enumerate(texts):
            entities = self.extract_entities(text)
            results.append(entities)

            if show_progress and (idx + 1) % 100 == 0:
                logger.info("Processed %d/%d texts", idx + 1, len(texts))

        return results

    def extract_from_paper(
        self,
        parsed_paper: Any,
        sections: Optional[List[str]] = None,
        deduplicate: bool = True,
    ) -> List[ExtractedEntity]:
        """Extract entities from a parsed paper with deduplication.

        Parameters
        ----------
        parsed_paper : ParsedPaper
            Parsed paper object from corpus_loader.
        sections : List[str], optional
            Sections to extract from. If None, uses all text.
        deduplicate : bool
            Whether to deduplicate entities by canonical_id.

        Returns
        -------
        List[ExtractedEntity]
            List of unique extracted entities.
        """
        # Gather text from sections
        texts_to_process = []

        if sections:
            # Abstract
            if "abstract" in [s.lower() for s in sections] and parsed_paper.abstract:
                texts_to_process.append(("abstract", parsed_paper.abstract))

            # Body sections
            for sec_name in sections:
                sec_lower = sec_name.lower()
                for key, text in parsed_paper.body_sections.items():
                    if sec_lower in key.lower():
                        texts_to_process.append((key, text))
        else:
            # Use all available text
            if parsed_paper.abstract:
                texts_to_process.append(("abstract", parsed_paper.abstract))
            if parsed_paper.full_text:
                texts_to_process.append(("full_text", parsed_paper.full_text))

        # Extract from all texts
        all_entities: List[ExtractedEntity] = []
        for section_name, text in texts_to_process:
            entities = self.extract_entities(text)
            for ent in entities:
                # Add source section to metadata
                all_entities.append(ent)

        # Deduplicate by canonical_id
        if deduplicate:
            seen_ids: Set[str] = set()
            unique_entities = []
            for ent in all_entities:
                if ent.canonical_id not in seen_ids:
                    seen_ids.add(ent.canonical_id)
                    unique_entities.append(ent)
            return unique_entities

        return all_entities

    def get_entity_statistics(self, entities: List[ExtractedEntity]) -> Dict[str, Any]:
        """Compute statistics about extracted entities.

        Parameters
        ----------
        entities : List[ExtractedEntity]
            List of extracted entities.

        Returns
        -------
        Dict[str, Any]
            Statistics dictionary.
        """
        if not entities:
            return {
                "total": 0,
                "umls_linked": 0,
                "fallback_ids": 0,
                "by_type": {},
                "avg_confidence": 0.0,
            }

        umls_linked = sum(1 for e in entities if e.is_umls_linked)
        by_type: Dict[str, int] = {}
        for ent in entities:
            by_type[ent.entity_type] = by_type.get(ent.entity_type, 0) + 1

        avg_confidence = sum(e.confidence for e in entities) / len(entities)

        return {
            "total": len(entities),
            "umls_linked": umls_linked,
            "fallback_ids": len(entities) - umls_linked,
            "umls_ratio": umls_linked / len(entities),
            "by_type": by_type,
            "avg_confidence": avg_confidence,
        }


# ===========================================================================
# Factory function
# ===========================================================================


def create_extractor_from_config(config: Any) -> EntityExtractor:
    """Create an EntityExtractor from FRLM config.

    Parameters
    ----------
    config : FRLMConfig
        FRLM configuration object.

    Returns
    -------
    EntityExtractor
        Configured entity extractor.
    """
    entity_cfg = config.extraction.entity

    return EntityExtractor(
        spacy_model=entity_cfg.spacy_model,
        linker=entity_cfg.linker,
        resolve_abbreviations=entity_cfg.resolve_abbreviations,
        min_entity_length=entity_cfg.min_entity_length,
        max_entity_length=entity_cfg.max_entity_length,
        confidence_threshold=entity_cfg.confidence_threshold,
        batch_size=entity_cfg.batch_size,
        n_process=entity_cfg.n_process,
    )

