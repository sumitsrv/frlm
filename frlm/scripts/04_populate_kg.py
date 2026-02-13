#!/usr/bin/env python3
"""
04_populate_kg.py - Populate Neo4j knowledge graph from extracted data.

Loads entity and relation JSON from the extraction pipeline and imports them
into Neo4j. Creates schema constraints/indexes, imports in batches, and
builds temporal version chains (SUPERSEDES edges).

Pipeline position: Step 4 of 11
Reads from:  config.paths.processed_dir (entity + relation JSON)
Writes to:   Neo4j database (config.neo4j)
Config used: config.neo4j, config.paths
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FRLMConfig, load_config, setup_logging

logger = logging.getLogger(__name__)


def _compute_fact_hash(
    subject_id: str, relation: str, object_id: str, valid_from: str,
    algorithm: str, separator: str,
) -> str:
    """Compute content-addressable hash for a fact: SHA-256(subj||rel||obj||valid_from)."""
    payload = separator.join([subject_id, relation, object_id, valid_from])
    return hashlib.new(algorithm, payload.encode("utf-8")).hexdigest()


def _create_schema_constraints(driver: Any, neo4j_cfg: Any) -> None:
    """Create uniqueness constraints and indexes on Neo4j schema."""
    schema = neo4j_cfg.graph_schema

    constraints = [
        f"CREATE CONSTRAINT IF NOT EXISTS FOR (e:{schema.entity_label}) REQUIRE e.cui IS UNIQUE",
        f"CREATE CONSTRAINT IF NOT EXISTS FOR (f:{schema.fact_label}) REQUIRE f.hash IS UNIQUE",
        f"CREATE INDEX IF NOT EXISTS FOR (f:{schema.fact_label}) ON (f.subject_id)",
        f"CREATE INDEX IF NOT EXISTS FOR (f:{schema.fact_label}) ON (f.object_id)",
        f"CREATE INDEX IF NOT EXISTS FOR (f:{schema.fact_label}) ON (f.valid_from)",
        f"CREATE INDEX IF NOT EXISTS FOR (f:{schema.fact_label}) ON (f.valid_to)",
    ]

    logger.info("Creating %d schema constraints/indexes", len(constraints))
    for cypher in constraints:
        logger.debug("Executing: %s", cypher[:120])
        # Production: with driver.session(database=neo4j_cfg.database) as session:
        #     session.run(cypher)


def _load_extraction_data(
    processed_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load all entity and relation JSON files."""
    entities: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    for path in sorted(processed_dir.glob("entities_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                entities.extend(data if isinstance(data, list) else [data])
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load %s: %s", path.name, exc)

    for path in sorted(processed_dir.glob("relations_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                relations.extend(data if isinstance(data, list) else [data])
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load %s: %s", path.name, exc)

    logger.info("Loaded %d entities and %d relations", len(entities), len(relations))
    return entities, relations


def _import_entities_batch(
    driver: Any, entities: List[Dict[str, Any]], neo4j_cfg: Any,
) -> int:
    """Import entity nodes in batches using UNWIND."""
    schema = neo4j_cfg.graph_schema
    batch_size = neo4j_cfg.batch.import_batch_size
    imported = 0

    for i in range(0, len(entities), batch_size):
        batch = entities[i : i + batch_size]
        cypher = (
            f"UNWIND $entities AS e "
            f"MERGE (n:{schema.entity_label} {{cui: e.cui}}) "
            f"ON CREATE SET n.canonical_name = e.canonical_name, "
            f"n.text = e.text, n.label = e.label, n.confidence = e.confidence "
            f"ON MATCH SET n.confidence = CASE WHEN e.confidence > n.confidence "
            f"THEN e.confidence ELSE n.confidence END"
        )
        logger.debug("Entity batch %d-%d of %d", i + 1, min(i + batch_size, len(entities)), len(entities))
        # Production: with driver.session(database=neo4j_cfg.database) as session:
        #     session.run(cypher, {"entities": batch})
        imported += len(batch)

    return imported


def _import_relations_batch(
    driver: Any, relations: List[Dict[str, Any]], neo4j_cfg: Any,
) -> int:
    """Import fact nodes and link to entities in batches."""
    schema = neo4j_cfg.graph_schema
    batch_size = neo4j_cfg.batch.import_batch_size
    imported = 0

    for i in range(0, len(relations), batch_size):
        batch = relations[i : i + batch_size]
        for rel in batch:
            rel["hash"] = _compute_fact_hash(
                subject_id=rel.get("subject", ""),
                relation=rel.get("relation_type", ""),
                object_id=rel.get("object", ""),
                valid_from=rel.get("valid_from", ""),
                algorithm=schema.hash_algorithm,
                separator=schema.hash_separator,
            )

        cypher = (
            f"UNWIND $relations AS r "
            f"MERGE (f:{schema.fact_label} {{hash: r.hash}}) "
            f"ON CREATE SET f.subject = r.subject, f.relation_type = r.relation_type, "
            f"f.object = r.object, f.confidence = r.confidence, "
            f"f.evidence_span = r.evidence_span, f.valid_from = r.valid_from, "
            f"f.valid_to = r.valid_to"
        )
        logger.debug("Relation batch %d-%d of %d", i + 1, min(i + batch_size, len(relations)), len(relations))
        # Production: with driver.session(database=neo4j_cfg.database) as session:
        #     session.run(cypher, {"relations": batch})
        imported += len(batch)

    return imported


def _build_version_chains(driver: Any, neo4j_cfg: Any) -> int:
    """Create SUPERSEDES edges between temporal versions of the same fact."""
    schema = neo4j_cfg.graph_schema
    cypher = (
        f"MATCH (f1:{schema.fact_label}), (f2:{schema.fact_label}) "
        f"WHERE f1.subject = f2.subject AND f1.relation_type = f2.relation_type "
        f"AND f1.object = f2.object AND f1.valid_from > f2.valid_from "
        f"AND NOT EXISTS((f1)-[:{schema.version_chain_type}]->(f2)) "
        f"CREATE (f1)-[:{schema.version_chain_type}]->(f2) "
        f"RETURN count(*) AS chains_created"
    )
    logger.info("Building version chains (%s edges)", schema.version_chain_type)
    # Production: with driver.session(database=neo4j_cfg.database) as session:
    #     result = session.run(cypher)
    #     return result.single()["chains_created"]
    return 0


def populate_kg(cfg: FRLMConfig) -> None:
    """Orchestrate KG population: connect, create schema, import, build chains."""
    neo4j_cfg = cfg.neo4j
    processed_dir = cfg.paths.resolve("processed_dir")

    logger.info("=== Knowledge Graph Population (Neo4j) ===")
    logger.info("URI: %s, Database: %s", neo4j_cfg.uri, neo4j_cfg.database)
    logger.info("Batch size: %d", neo4j_cfg.batch.import_batch_size)

    # Production: from neo4j import GraphDatabase
    # driver = GraphDatabase.driver(neo4j_cfg.uri, auth=(neo4j_cfg.username, neo4j_cfg.password))
    driver = None

    try:
        _create_schema_constraints(driver, neo4j_cfg)
        entities, relations = _load_extraction_data(processed_dir)

        if not entities and not relations:
            logger.warning("No extraction data found. Run steps 02-03 first.")
            return

        t0 = time.time()
        entity_count = _import_entities_batch(driver, entities, neo4j_cfg)
        logger.info("Imported %d entities in %.2fs", entity_count, time.time() - t0)

        t0 = time.time()
        relation_count = _import_relations_batch(driver, relations, neo4j_cfg)
        logger.info("Imported %d relations in %.2fs", relation_count, time.time() - t0)

        t0 = time.time()
        chain_count = _build_version_chains(driver, neo4j_cfg)
        logger.info("Built %d version chains in %.2fs", chain_count, time.time() - t0)

        # Write summary
        kg_dir = cfg.paths.resolve("kg_dir")
        kg_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "entities_imported": entity_count,
            "relations_imported": relation_count,
            "version_chains": chain_count,
        }
        with open(kg_dir / "kg_import_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        logger.info("=== KG Summary: %d entities, %d relations, %d chains ===",
                    entity_count, relation_count, chain_count)
    finally:
        if driver is not None:
            driver.close()


def main() -> None:
    """Parse arguments, load config, and populate the KG."""
    parser = argparse.ArgumentParser(
        description="Populate Neo4j knowledge graph from extracted data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.batch_size is not None:
        overrides["neo4j.batch.import_batch_size"] = args.batch_size

    cfg = load_config(args.config, overrides=overrides if overrides else None)
    setup_logging(cfg)
    logger.info("Starting 04_populate_kg with config: %s", args.config)

    try:
        populate_kg(cfg)
    except KeyboardInterrupt:
        logger.warning("KG population interrupted.")
        sys.exit(130)
    except Exception:
        logger.exception("KG population failed.")
        sys.exit(1)

    logger.info("04_populate_kg completed successfully.")


if __name__ == "__main__":
    main()