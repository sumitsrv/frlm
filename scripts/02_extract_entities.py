#!/usr/bin/env python3
"""
02_extract_entities.py - Extract biomedical entities with SciSpacy NER + UMLS linking.

Processes raw corpus XML files through SciSpacy's en_core_sci_lg model with
UMLS concept linking. Outputs entity annotations as JSON.

Pipeline position: Step 2 of 11
Reads from:  config.paths.corpus_dir (raw XML)
Writes to:   config.paths.processed_dir (entity JSON)
Config used: config.extraction.entity, config.paths
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FRLMConfig, load_config, setup_logging

logger = logging.getLogger(__name__)


def _parse_pmc_xml(xml_path: Path, sections: List[str]) -> List[Dict[str, str]]:
    """Parse a PMC XML file and extract text from configured sections.

    Returns a list of dicts with keys: section, text, pmcid.
    """
    pmcid = xml_path.stem
    results: List[Dict[str, str]] = []

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        # Extract abstract
        if "abstract" in sections:
            for abstract in root.iter("abstract"):
                text = " ".join(abstract.itertext()).strip()
                if text:
                    results.append(
                        {"section": "abstract", "text": text, "pmcid": pmcid}
                    )

        # Extract body sections
        for body in root.iter("body"):
            for sec in body.iter("sec"):
                sec_title_elem = sec.find("title")
                sec_title = (
                    sec_title_elem.text.lower().strip()
                    if sec_title_elem is not None and sec_title_elem.text
                    else ""
                )

                matched_section = None
                for s in sections:
                    if s in sec_title:
                        matched_section = s
                        break

                if matched_section:
                    paragraphs = []
                    for p in sec.iter("p"):
                        p_text = " ".join(p.itertext()).strip()
                        if p_text:
                            paragraphs.append(p_text)
                    if paragraphs:
                        results.append(
                            {
                                "section": matched_section,
                                "text": " ".join(paragraphs),
                                "pmcid": pmcid,
                            }
                        )

    except ET.ParseError as exc:
        logger.warning("Failed to parse XML %s: %s", xml_path.name, exc)
    except Exception as exc:
        logger.error("Unexpected error parsing %s: %s", xml_path.name, exc)

    return results


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks by approximate word count."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - chunk_overlap
    return chunks


def _extract_entities_from_text(
    text: str,
    nlp: Any,
    entity_cfg: Any,
) -> List[Dict[str, Any]]:
    """Run NER + UMLS linking on a text chunk.

    Returns a list of entity dicts with keys:
    text, label, start, end, cui, canonical_name, confidence.
    """
    entities: List[Dict[str, Any]] = []

    doc = nlp(text)
    for ent in doc.ents:
        if len(ent.text) < entity_cfg.min_entity_length:
            continue
        if len(ent.text) > entity_cfg.max_entity_length:
            continue

        entity_dict: Dict[str, Any] = {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "cui": None,
            "canonical_name": None,
            "confidence": 0.0,
        }

        # UMLS linking via linker component
        if hasattr(ent, "_") and hasattr(ent._, "kb_ents") and ent._.kb_ents:
            top_concept = ent._.kb_ents[0]
            cui, score = top_concept
            if score >= entity_cfg.confidence_threshold:
                entity_dict["cui"] = cui
                entity_dict["confidence"] = round(float(score), 4)
                # Resolve canonical name from linker KB
                try:
                    linker = nlp.get_pipe("scispacy_linker")
                    kb = linker.kb
                    # UmlsKnowledgeBase uses cui_to_entity dict, not __contains__
                    cui_map = getattr(kb, "cui_to_entity", None)
                    if cui_map is not None and cui in cui_map:
                        entity_dict["canonical_name"] = cui_map[cui].canonical_name
                    elif hasattr(kb, "__getitem__"):
                        entity_dict["canonical_name"] = kb[cui].canonical_name
                except Exception:
                    pass  # canonical name lookup is best-effort

        entities.append(entity_dict)

    return entities


def _load_spacy_model(entity_cfg: Any) -> Any:
    """Load the SciSpacy model with UMLS linker.

    Returns the loaded spaCy nlp pipeline.
    """
    import spacy

    logger.info("Loading SciSpacy model: %s", entity_cfg.spacy_model)
    nlp = spacy.load(entity_cfg.spacy_model)

    if entity_cfg.resolve_abbreviations:
        try:
            from scispacy.abbreviation import AbbreviationDetector

            nlp.add_pipe("abbreviation_detector")
            logger.info("Added abbreviation detector")
        except ImportError:
            logger.warning("scispacy abbreviation detector not available")

    if entity_cfg.linker == "umls":
        try:
            from scispacy.linking import EntityLinker

            nlp.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": entity_cfg.resolve_abbreviations, "linker_name": "umls"},
            )
            logger.info("Added UMLS entity linker")
        except ImportError:
            logger.warning("scispacy entity linker not available")

    return nlp


def extract_entities(cfg: FRLMConfig) -> None:
    """Orchestrate entity extraction across all corpus files.

    1. Load SciSpacy model.
    2. Discover XML files.
    3. Parse, chunk, and extract entities.
    4. Write entity JSON per document.
    """
    entity_cfg = cfg.extraction.entity
    corpus_cfg = cfg.extraction.corpus
    corpus_dir = cfg.paths.resolve("corpus_dir")
    processed_dir = cfg.paths.resolve("processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Entity Extraction (SciSpacy + UMLS) ===")
    logger.info("Model: %s", entity_cfg.spacy_model)
    logger.info("Linker: %s", entity_cfg.linker)
    logger.info("Confidence threshold: %.2f", entity_cfg.confidence_threshold)
    logger.info("Chunk size: %d, Overlap: %d", corpus_cfg.chunk_size, corpus_cfg.chunk_overlap)

    # Step 1: load model
    nlp = _load_spacy_model(entity_cfg)

    # Step 2: discover files
    xml_files = sorted(corpus_dir.glob("*.xml"))
    logger.info("Found %d XML files in %s", len(xml_files), corpus_dir)

    if not xml_files:
        logger.warning("No XML files found. Run step 01 first.")
        return

    # Step 3 & 4: process
    start_time = time.time()
    total_entities = 0
    total_chunks = 0
    failed_files = 0

    for idx, xml_path in enumerate(xml_files, start=1):
        output_path = processed_dir / f"entities_{xml_path.stem}.json"
        if output_path.exists():
            logger.debug("Skipping already processed: %s", xml_path.stem)
            continue

        try:
            sections = _parse_pmc_xml(xml_path, corpus_cfg.sections)
            doc_entities: List[Dict[str, Any]] = []

            for sec in sections:
                chunks = _chunk_text(
                    sec["text"], corpus_cfg.chunk_size, corpus_cfg.chunk_overlap
                )
                for chunk_idx, chunk in enumerate(chunks):
                    entities = _extract_entities_from_text(chunk, nlp, entity_cfg)
                    for ent in entities:
                        ent["pmcid"] = sec["pmcid"]
                        ent["section"] = sec["section"]
                        ent["chunk_idx"] = chunk_idx
                    doc_entities.extend(entities)
                    total_chunks += 1

            total_entities += len(doc_entities)

            # Atomic write
            tmp_path = output_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(doc_entities, fh, indent=2, ensure_ascii=False)
            tmp_path.replace(output_path)

        except Exception as exc:
            failed_files += 1
            logger.error(
                "Failed to process %s: %s", xml_path.name, exc, exc_info=True
            )
            # Continue with the next file — don't lose all progress

        if idx % 50 == 0 or idx == len(xml_files):
            elapsed = time.time() - start_time
            logger.info(
                "Entity extraction progress: %d/%d files, %d entities, %d chunks, %d failures (%.1fs)",
                idx,
                len(xml_files),
                total_entities,
                total_chunks,
                failed_files,
                elapsed,
            )

    total_time = time.time() - start_time
    logger.info("=== Entity Extraction Summary ===")
    logger.info("Files processed: %d", len(xml_files))
    logger.info("Total entities: %d", total_entities)
    logger.info("Total chunks: %d", total_chunks)
    logger.info("Failed files: %d", failed_files)
    logger.info("Processing time: %.2f seconds", total_time)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, load config, and run entity extraction."""
    parser = argparse.ArgumentParser(
        description="Extract biomedical entities with SciSpacy NER + UMLS linking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of XML files to process.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    logger.info("Starting 02_extract_entities with config: %s", args.config)

    try:
        extract_entities(cfg)
    except KeyboardInterrupt:
        logger.warning("Entity extraction interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Entity extraction failed with an unexpected error.")
        sys.exit(1)

    logger.info("02_extract_entities completed successfully.")


if __name__ == "__main__":
    main()