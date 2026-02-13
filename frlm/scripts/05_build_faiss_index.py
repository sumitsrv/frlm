#!/usr/bin/env python3
"""
05_build_faiss_index.py - Build FAISS IVF-PQ index over SapBERT KG embeddings.

Embeds all KG facts with frozen SapBERT, trains a FAISS IVF-PQ index,
and builds hierarchical indices at four granularity levels.

Pipeline position: Step 5 of 11
Reads from:  Neo4j KG (config.neo4j), SapBERT model
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FRLMConfig, load_config, setup_logging

logger = logging.getLogger(__name__)


def _load_facts_from_kg(cfg: FRLMConfig) -> List[Dict[str, Any]]:
    """Load all facts from Neo4j for embedding.

    Returns list of fact dicts with text representation for SapBERT.
    """
    kg_dir = cfg.paths.resolve("kg_dir")
    facts_path = kg_dir / "exported_facts.json"

    if facts_path.exists():
        logger.info("Loading cached facts from %s", facts_path)
        with open(facts_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    logger.info("Querying facts from Neo4j at %s", cfg.neo4j.uri)
    # Production:
    #   from neo4j import GraphDatabase
    #   driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(...))
    #   with driver.session(database=cfg.neo4j.database) as session:
    #       result = session.run(
    #           f"MATCH (f:{cfg.neo4j.graph_schema.fact_label}) RETURN f"
    #       )
    #       facts = [record["f"] for record in result]
    logger.info("Would query Neo4j for all facts (stub mode)")
    return []


def _build_fact_text(fact: Dict[str, Any], level: int) -> str:
    """Build text representation of a fact for SapBERT embedding.

    Level 0 (atomic): "subject relation object"
    Level 1 (relation): "subject [all relations] object"
    Level 2 (entity): "entity: [all relations and objects]"
    Level 3 (cluster): "cluster: [entity1, entity2, ...]"
    """
    subject = fact.get("subject", fact.get("subject_label", ""))
    relation = fact.get("relation_type", fact.get("relation", ""))
    obj = fact.get("object", fact.get("object_label", ""))

    if level == 0:
        return f"{subject} {relation} {obj}"
    elif level == 1:
        return f"{subject} relates to {obj}"
    elif level == 2:
        return f"{subject}"
    elif level == 3:
        return f"{subject} {obj}"
    return f"{subject} {relation} {obj}"


def _embed_facts(
    facts: List[Dict[str, Any]],
    level: int,
    sapbert_cfg: Any,
) -> np.ndarray:
    """Embed facts using frozen SapBERT encoder.

    Returns numpy array of shape (n_facts, embedding_dim).
    """
    texts = [_build_fact_text(f, level) for f in facts]
    n = len(texts)
    dim = sapbert_cfg.embedding_dim

    logger.info("Embedding %d facts at level %d with SapBERT", n, level)
    logger.info("Model: %s, Device: %s, Batch size: %d",
                sapbert_cfg.model_name, sapbert_cfg.device, sapbert_cfg.batch_size)

    # Production:
    #   from transformers import AutoTokenizer, AutoModel
    #   tokenizer = AutoTokenizer.from_pretrained(sapbert_cfg.model_name)
    #   model = AutoModel.from_pretrained(sapbert_cfg.model_name).to(sapbert_cfg.device)
    #   model.eval()
    #   embeddings = []
    #   for i in range(0, n, sapbert_cfg.batch_size):
    #       batch = texts[i:i+sapbert_cfg.batch_size]
    #       inputs = tokenizer(batch, padding=True, truncation=True,
    #                          max_length=sapbert_cfg.max_length, return_tensors="pt")
    #       inputs = {k: v.to(sapbert_cfg.device) for k, v in inputs.items()}
    #       with torch.no_grad():
    #           outputs = model(**inputs)
    #       if sapbert_cfg.pool_strategy == "cls":
    #           emb = outputs.last_hidden_state[:, 0, :]
    #       emb = F.normalize(emb, p=2, dim=1)
    #       embeddings.append(emb.cpu().numpy())
    #   return np.vstack(embeddings)

    logger.info("Would embed %d texts (stub mode, returning random vectors)", n)
    if n == 0:
        return np.empty((0, dim), dtype=np.float32)
    return np.random.randn(n, dim).astype(np.float32)


def _build_faiss_index(
    embeddings: np.ndarray,
    faiss_cfg: Any,
    output_path: Path,
) -> None:
    """Build and save a FAISS IVF-PQ index.

    1. Train on a sample of embeddings.
    2. Add all embeddings.
    3. Save to disk.
    """
    n, dim = embeddings.shape
    logger.info("Building FAISS index: %s (%d vectors, %d dims)", faiss_cfg.index_type, n, dim)

    # Production:
    #   import faiss
    #   quantizer = faiss.IndexFlatL2(dim)
    #   index = faiss.IndexIVFPQ(quantizer, dim, faiss_cfg.nlist, faiss_cfg.pq_m, faiss_cfg.pq_nbits)
    #   train_sample_size = min(faiss_cfg.train_sample_size, n)
    #   train_indices = np.random.choice(n, train_sample_size, replace=False)
    #   train_vectors = embeddings[train_indices]
    #   index.train(train_vectors)
    #   index.add(embeddings)
    #   index.nprobe = faiss_cfg.nprobe
    #   if faiss_cfg.use_gpu:
    #       res = faiss.StandardGpuResources()
    #       index = faiss.index_cpu_to_gpu(res, faiss_cfg.gpu_id, index)
    #   faiss.write_index(faiss.index_gpu_to_cpu(index), str(output_path))

    logger.info("Would build FAISS %s index (stub mode)", faiss_cfg.index_type)
    logger.info("nlist=%d, pq_m=%d, pq_nbits=%d, nprobe=%d",
                faiss_cfg.nlist, faiss_cfg.pq_m, faiss_cfg.pq_nbits, faiss_cfg.nprobe)
    logger.info("GPU: %s (device %d)", faiss_cfg.use_gpu, faiss_cfg.gpu_id)
    logger.info("Index would be saved to %s", output_path)


def build_faiss_index(cfg: FRLMConfig) -> None:
    """Orchestrate FAISS index building at all hierarchical levels."""
    faiss_cfg = cfg.faiss
    sapbert_cfg = cfg.sapbert
    index_dir = cfg.paths.resolve("faiss_index_dir")
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== FAISS Index Building ===")
    logger.info("Index type: %s", faiss_cfg.index_type)
    logger.info("Embedding dim: %d", faiss_cfg.embedding_dim)

    facts = _load_facts_from_kg(cfg)
    if not facts:
        logger.warning("No facts found. Run step 04 first or export facts.")
        logger.info("Building with empty index for testing.")

    # Build index for each hierarchical level
    level_names = {
        0: faiss_cfg.hierarchical.level_0,
        1: faiss_cfg.hierarchical.level_1,
        2: faiss_cfg.hierarchical.level_2,
        3: faiss_cfg.hierarchical.level_3,
    }

    metadata = {"levels": {}, "total_facts": len(facts)}

    for level, name in level_names.items():
        logger.info("--- Level %d: %s ---", level, name)
        t0 = time.time()

        embeddings = _embed_facts(facts, level, sapbert_cfg)
        index_path = index_dir / f"index_level_{level}_{name}.faiss"
        _build_faiss_index(embeddings, faiss_cfg, index_path)

        elapsed = time.time() - t0
        metadata["levels"][str(level)] = {
            "name": name,
            "num_vectors": int(embeddings.shape[0]) if embeddings.size > 0 else 0,
            "index_path": str(index_path),
            "build_time_seconds": round(elapsed, 2),
        }
        logger.info("Level %d (%s): %d vectors in %.2fs", level, name,
                    embeddings.shape[0] if embeddings.size > 0 else 0, elapsed)

    # Save metadata
    with open(index_dir / "index_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("=== FAISS Index Building Complete ===")


def main() -> None:
    """Parse arguments, load config, and build FAISS index."""
    parser = argparse.ArgumentParser(
        description="Build FAISS IVF-PQ index over SapBERT KG embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--levels", type=str, default="0,1,2,3",
                        help="Comma-separated hierarchical levels to build.")
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