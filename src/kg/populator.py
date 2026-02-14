"""
KG Populator — orchestrates the full pipeline to populate the Neo4j knowledge graph.

Steps:
    1. Iterate papers from :class:`PMCCorpusLoader`
    2. Extract entities via :class:`EntityExtractor`
    3. Extract relations via :class:`RelationExtractor`
    4. Convert to :class:`Fact` objects with SHA-256 content-addressable IDs
    5. Bulk-import into Neo4j via :class:`Neo4jClient`

Supports checkpoint/resume (writes progress to a JSON file so that a
crashed run can continue where it left off), deduplication via fact_id,
and configurable batch sizes.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.extraction.corpus_loader import PMCCorpusLoader
from src.extraction.entity_extractor import EntityExtractor
from src.extraction.relation_extractor import RelationExtractor
from src.kg.neo4j_client import Neo4jClient
from src.kg.schema import (
    BiomedicalEntity,
    Fact,
    Relation,
    RelationType,
    TemporalEnvelope,
    compute_fact_id,
)

logger = logging.getLogger(__name__)


class KGPopulator:
    """Orchestrate corpus → NER → RE → KG import.

    Parameters
    ----------
    corpus_loader : PMCCorpusLoader
        Corpus iterator over PubMed Central papers.
    entity_extractor : EntityExtractor
        Named entity recognition pipeline.
    relation_extractor : RelationExtractor
        Relation extraction pipeline (Claude API-backed).
    neo4j_client : Neo4jClient
        Neo4j connection client.
    config : object
        ``FRLMConfig`` (or compatible) carrying ``paths``, ``neo4j``,
        and extraction sub-configs.
    batch_size : int
        Number of facts to accumulate before flushing to Neo4j.
    checkpoint_dir : Path or str, optional
        Directory for checkpoint files.  Defaults to ``config.paths.processed_dir``.
    """

    def __init__(
        self,
        corpus_loader: PMCCorpusLoader,
        entity_extractor: EntityExtractor,
        relation_extractor: RelationExtractor,
        neo4j_client: Neo4jClient,
        config: Any,
        batch_size: int = 500,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self._corpus = corpus_loader
        self._ner = entity_extractor
        self._re = relation_extractor
        self._neo4j = neo4j_client
        self._cfg = config
        self._batch_size = batch_size

        self._checkpoint_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else Path(config.paths.resolve("processed_dir"))
        )
        self._checkpoint_path = self._checkpoint_dir / "kg_populator_checkpoint.json"

        # Deduplication: track fact_ids already imported
        self._seen_fact_ids: Set[str] = set()

        # Statistics
        self._stats: Dict[str, int] = {
            "papers_processed": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "facts_created": 0,
            "facts_deduplicated": 0,
            "facts_imported": 0,
        }

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> Set[str]:
        """Load set of already-processed paper IDs from checkpoint file."""
        if not self._checkpoint_path.exists():
            return set()
        try:
            with open(self._checkpoint_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            processed = set(data.get("processed_paper_ids", []))
            self._seen_fact_ids = set(data.get("seen_fact_ids", []))
            self._stats = data.get("stats", self._stats)
            logger.info(
                "Loaded checkpoint: %d papers already processed, %d facts seen",
                len(processed),
                len(self._seen_fact_ids),
            )
            return processed
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load checkpoint: %s — starting fresh", exc)
            return set()

    def _save_checkpoint(self, processed_ids: Set[str]) -> None:
        """Persist progress to disk."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "processed_paper_ids": sorted(processed_ids),
            "seen_fact_ids": sorted(self._seen_fact_ids),
            "stats": self._stats,
        }
        with open(self._checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.debug("Checkpoint saved (%d papers)", len(processed_ids))

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _relation_to_fact(
        subject: BiomedicalEntity,
        relation_type: str,
        obj: BiomedicalEntity,
        confidence: float = 1.0,
        evidence_span: str = "",
        valid_from: Optional[str] = None,
    ) -> Optional[Fact]:
        """Convert extracted relation components into a :class:`Fact`.

        Returns *None* if the relation type is not in the ontology.
        """
        try:
            rel_type = RelationType(relation_type)
        except ValueError:
            logger.debug("Unknown relation type '%s' — skipping", relation_type)
            return None

        temporal = TemporalEnvelope(
            valid_from=valid_from or str(__import__("datetime").date.today()),
        )

        return Fact(
            subject=subject,
            relation=Relation(type=rel_type),
            object=obj,
            temporal=temporal,
            confidence=confidence,
            evidence_span=evidence_span,
        )

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _process_paper(
        self, paper: Dict[str, Any]
    ) -> Tuple[List[BiomedicalEntity], List[Fact]]:
        """Extract entities and relations from a single paper.

        Returns
        -------
        entities : list[BiomedicalEntity]
        facts : list[Fact]
        """
        text = paper.get("text", "")
        if not text:
            return [], []

        # --- NER ---
        raw_entities = self._ner.extract_entities(text)
        entities: List[BiomedicalEntity] = []
        entity_map: Dict[str, BiomedicalEntity] = {}

        for ent in raw_entities:
            # EntityExtractor returns dicts with keys like
            # "text", "label", "cui", "canonical_name", "confidence"
            if isinstance(ent, dict):
                cui = ent.get("cui", ent.get("canonical_id", ""))
                if not cui:
                    continue
                bio_ent = BiomedicalEntity(
                    id=cui,
                    label=ent.get("canonical_name", ent.get("text", "")),
                    entity_type=ent.get("label", "Unknown"),
                    canonical_id=cui,
                    source_ontology=ent.get("source_ontology", "UMLS"),
                )
            elif isinstance(ent, BiomedicalEntity):
                bio_ent = ent
            else:
                continue
            entities.append(bio_ent)
            entity_map[bio_ent.canonical_id] = bio_ent

        self._stats["entities_extracted"] += len(entities)

        # --- Relation Extraction ---
        raw_relations = self._re.extract_relations(text, raw_entities)
        self._stats["relations_extracted"] += len(raw_relations)

        # --- Build Facts ---
        facts: List[Fact] = []
        for rel in raw_relations:
            if isinstance(rel, dict):
                subj_id = rel.get("subject", "")
                obj_id = rel.get("object", "")
                rel_type = rel.get("relation_type", "")
                confidence = rel.get("confidence", 1.0)
                evidence = rel.get("evidence_span", "")
                valid_from = rel.get("valid_from")
            else:
                continue

            subj = entity_map.get(subj_id)
            obj_ent = entity_map.get(obj_id)
            if subj is None or obj_ent is None:
                continue

            fact = self._relation_to_fact(
                subject=subj,
                relation_type=rel_type,
                obj=obj_ent,
                confidence=confidence,
                evidence_span=evidence,
                valid_from=valid_from,
            )
            if fact is None:
                continue

            # Deduplicate by fact_id
            fid = fact.fact_id
            if fid in self._seen_fact_ids:
                self._stats["facts_deduplicated"] += 1
                continue

            self._seen_fact_ids.add(fid)
            facts.append(fact)

        self._stats["facts_created"] += len(facts)
        return entities, facts

    def _flush_facts(self, facts: List[Fact]) -> int:
        """Bulk-import a batch of facts into Neo4j.

        Returns the number of facts successfully imported.
        """
        if not facts:
            return 0
        try:
            imported = self._neo4j.bulk_import_facts(facts)
            self._stats["facts_imported"] += imported
            logger.info("Flushed %d facts to Neo4j", imported)
            return imported
        except Exception as exc:
            logger.error("Failed to flush %d facts: %s", len(facts), exc)
            return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(self, max_papers: Optional[int] = None) -> Dict[str, int]:
        """Run the full population pipeline.

        Parameters
        ----------
        max_papers : int, optional
            Cap on the number of papers to process (useful for testing).

        Returns
        -------
        dict
            Final statistics dictionary.
        """
        t0 = time.time()
        processed_ids = self._load_checkpoint()
        fact_buffer: List[Fact] = []

        logger.info("=== KG Population Pipeline ===")
        logger.info("Batch size: %d, checkpoint: %s", self._batch_size, self._checkpoint_path)

        paper_count = 0
        for paper in self._corpus.iterate_corpus():
            paper_id = paper.get("pmcid", paper.get("id", str(paper_count)))

            # Skip already-processed papers (resume support)
            if paper_id in processed_ids:
                continue

            if max_papers is not None and paper_count >= max_papers:
                logger.info("Reached max_papers=%d — stopping early", max_papers)
                break

            entities, facts = self._process_paper(paper)
            fact_buffer.extend(facts)

            processed_ids.add(paper_id)
            self._stats["papers_processed"] += 1
            paper_count += 1

            # Flush when buffer is full
            if len(fact_buffer) >= self._batch_size:
                self._flush_facts(fact_buffer)
                fact_buffer.clear()
                self._save_checkpoint(processed_ids)

            if paper_count % 100 == 0:
                logger.info(
                    "Progress: %d papers, %d entities, %d relations, %d facts",
                    self._stats["papers_processed"],
                    self._stats["entities_extracted"],
                    self._stats["relations_extracted"],
                    self._stats["facts_created"],
                )

        # Flush remaining
        if fact_buffer:
            self._flush_facts(fact_buffer)
        self._save_checkpoint(processed_ids)

        elapsed = time.time() - t0
        logger.info(
            "=== KG Population Complete in %.1fs ===\n"
            "  Papers processed:    %d\n"
            "  Entities extracted:  %d\n"
            "  Relations extracted: %d\n"
            "  Facts created:       %d\n"
            "  Facts deduplicated:  %d\n"
            "  Facts imported:      %d",
            elapsed,
            self._stats["papers_processed"],
            self._stats["entities_extracted"],
            self._stats["relations_extracted"],
            self._stats["facts_created"],
            self._stats["facts_deduplicated"],
            self._stats["facts_imported"],
        )
        return dict(self._stats)

    @property
    def stats(self) -> Dict[str, int]:
        """Return current statistics."""
        return dict(self._stats)
