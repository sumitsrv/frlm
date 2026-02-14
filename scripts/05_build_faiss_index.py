#!/usr/bin/env python3
"""
05_build_faiss_index.py — Build FAISS IVF-PQ index over SapBERT KG embeddings.

Embeds all KG facts with frozen SapBERT, trains a FAISS IVF-PQ index,
and builds hierarchical indices at four granularity levels.

Pipeline position: Step 5 of 11
Reads from:  Neo4j KG (config.neo4j) or exported facts JSON
Writes to:   config.paths.faiss_index_dir
Config used: config.faiss, config.sapbert, config.neo4j
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, load_config, setup_logging
from src.embeddings.sapbert import SapBERTEncoder
from src.embeddings.hierarchical import HierarchicalIndex
from src.kg.schema import (
    BiomedicalEntity,
    Fact,
    FactCluster,
    Relation,
    TemporalEnvelope,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fact loading
# ---------------------------------------------------------------------------


def _load_facts_from_export(cfg: FRLMConfig) -> List[Fact]:
    """Load facts from a JSON export file on disk.

    The expected file is ``<kg_dir>/exported_facts.json`` with an array of
    dicts each containing the keys written by ``Fact.to_neo4j_properties()``.
    """
    kg_dir = cfg.paths.resolve("kg_dir")
    facts_path = kg_dir / "exported_facts.json"

    if not facts_path.exists():
        logger.warning("No exported facts at %s", facts_path)
        return []

    logger.info("Loading facts from %s", facts_path)
    with open(facts_path, "r", encoding="utf-8") as fh:
        raw: List[Dict[str, Any]] = json.load(fh)

    facts: List[Fact] = []
    for entry in raw:
        try:
            from datetime import date as _date

            valid_from = _date.fromisoformat(entry["valid_from"])
            valid_to_raw = entry.get("valid_to")
            valid_to = (
                _date.fromisoformat(valid_to_raw)
                if valid_to_raw is not None
                else None
            )

            fact = Fact(
                subject=BiomedicalEntity(
                    id=entry.get("subject_id", ""),
                    label=entry.get("subject_label", ""),
                    entity_type=entry.get("subject_entity_type", "unknown"),
                    canonical_id=entry.get("subject_id", ""),
                ),
                relation=Relation(type=entry["relation_type"]),
                object=BiomedicalEntity(
                    id=entry.get("object_id", ""),
                    label=entry.get("object_label", ""),
                    entity_type=entry.get("object_entity_type", "unknown"),
                    canonical_id=entry.get("object_id", ""),
                ),
                temporal=TemporalEnvelope(
                    valid_from=valid_from,
                    valid_to=valid_to,
                ),
                source=entry.get("source", ""),
                confidence=entry.get("confidence", 1.0),
                metadata=json.loads(entry.get("metadata", "{}"))
                if isinstance(entry.get("metadata"), str)
                else entry.get("metadata", {}),
            )
            facts.append(fact)
        except Exception as exc:
            logger.warning("Skipping malformed fact entry: %s — %s", entry, exc)

    logger.info("Loaded %d facts from export", len(facts))
    return facts


def _load_facts_from_neo4j(cfg: FRLMConfig) -> List[Fact]:
    """Query all facts from a live Neo4j instance."""
    from src.kg.neo4j_client import Neo4jClient

    client = Neo4jClient.from_config(cfg)
    try:
        client.connect()
        logger.info("Querying facts from Neo4j at %s", cfg.neo4j.uri)
        logger.warning(
            "Live Neo4j bulk export not yet implemented; "
            "use exported_facts.json instead"
        )
        return []
    except Exception as exc:
        logger.warning("Could not connect to Neo4j: %s", exc)
        return []
    finally:
        try:
            client.close()
        except Exception:
            pass


def load_facts(cfg: FRLMConfig) -> List[Fact]:
    """Load facts from the best available source."""
    facts = _load_facts_from_export(cfg)
    if facts:
        return facts
    return _load_facts_from_neo4j(cfg)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_faiss_index(cfg: FRLMConfig) -> None:
    """Orchestrate FAISS index building at all hierarchical levels."""
    index_dir = cfg.paths.resolve("faiss_index_dir")
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== FAISS Index Building ===")
    logger.info("Index type: %s", cfg.faiss.index_type)
    logger.info("Embedding dim: %d", cfg.faiss.embedding_dim)
    logger.info("Output directory: %s", index_dir)

    # 1. Load facts
    t0 = time.time()
    facts = load_facts(cfg)
    if not facts:
        logger.warning(
            "No facts found. Run step 04 (populate KG) first, "
            "or place exported_facts.json in %s",
            cfg.paths.resolve("kg_dir"),
        )
        logger.info("Building empty indices for testing.")

    logger.info("Loaded %d facts in %.2fs", len(facts), time.time() - t0)

    # 2. Encode with SapBERT
    t0 = time.time()
    encoder = SapBERTEncoder.from_config(cfg.sapbert)
    if facts:
        atomic_embeddings = encoder.encode_facts_batch(
            facts, batch_size=cfg.sapbert.batch_size
        )
    else:
        atomic_embeddings = np.empty(
            (0, cfg.faiss.embedding_dim), dtype=np.float32
        )
    logger.info(
        "Encoded %d facts → (%s) in %.2fs",
        len(facts),
        atomic_embeddings.shape,
        time.time() - t0,
    )

    # 3. Build hierarchical index
    t0 = time.time()
    hier = HierarchicalIndex.from_config(cfg.faiss)
    hier.build_all_levels(facts, atomic_embeddings, clusters=None)
    logger.info("Hierarchical index built in %.2fs", time.time() - t0)

    # 4. Save to disk
    hier.save(index_dir)

    # 5. Summary
    stats = hier.stats()
    logger.info("=== Index Statistics ===")
    for level_name, level_stats in stats.items():
        logger.info("  %s: %d vectors", level_name, level_stats["ntotal"])

    # Write machine-readable metadata
    meta_path = index_dir / "index_metadata.json"
    meta = {
        "total_facts": len(facts),
        "embedding_dim": cfg.faiss.embedding_dim,
        "index_type": cfg.faiss.index_type,
        "levels": stats,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info("=== FAISS Index Building Complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, load config, and build FAISS index."""
    parser = argparse.ArgumentParser(
        description="Build FAISS IVF-PQ index over SapBERT KG embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    logger.info("Starting 05_build_faiss_index with config: %s", args.config)

    try:
        build_faiss_index(cfg)
    except KeyboardInterrupt:
        logger.warning("FAISS index building interrupted.")
        sys.exit(130)
    except Exception:
        logger.exception("FAISS index building failed.")
        sys.exit(1)

    logger.info("05_build_faiss_index completed successfully.")


if __name__ == "__main__":
    main()
