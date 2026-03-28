#!/usr/bin/env python3
"""
06b_prepare_training_data.py — Convert span labels → tokenized training data.

**This is the missing bridge between step 06 and steps 07-09.**

Step 06 produces per-entity span labels (factual / linguistic).
Steps 07-09 expect tokenized training examples with:
  - Phase 1 (Router):    {input_ids, attention_mask, router_labels}
  - Phase 2 (Retrieval): {input_ids, attention_mask, span_mask, positive_embedding, negative_embeddings}
  - Phase 3 (Joint):     all above + {token_labels}

This script:
1. Re-extracts text from corpus XML (same chunking as step 02).
2. Loads entity offsets + span labels from steps 02/06.
3. Tokenizes with the backbone tokenizer.
4. Converts character-level labels → token-level binary labels.
5. For retrieval/joint data: embeds factual spans with SapBERT, mines
   positives/negatives from FAISS.
6. Writes JSONL files into data/labels/tokenized/ (Phase 1),
   data/processed/retrieval/ (Phase 2), data/processed/joint/ (Phase 3).

Pipeline position: Step 6b of 11 (inserted between 06 and 07)
Reads from:  data/corpus/*.xml, data/processed/entities_*.json,
             data/labels/labels_*.json, data/faiss_indices/
Writes to:   data/labels/tokenized/, data/processed/retrieval/,
             data/processed/joint/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, load_config, setup_logging

logger = logging.getLogger(__name__)


# ===========================================================================
# Text extraction (mirrors step 02 logic exactly)
# ===========================================================================


def _parse_pmc_xml(xml_path: Path, sections: List[str]) -> List[Dict[str, str]]:
    """Parse a PMC XML file and extract text from configured sections."""
    pmcid = xml_path.stem
    results: List[Dict[str, str]] = []

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        if "abstract" in sections:
            for abstract in root.iter("abstract"):
                text = " ".join(abstract.itertext()).strip()
                if text:
                    results.append(
                        {"section": "abstract", "text": text, "pmcid": pmcid}
                    )

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
    except Exception as exc:
        logger.warning("Failed to parse XML %s: %s", xml_path.name, exc)

    return results


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks by word count (mirrors step 02)."""
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


# ===========================================================================
# Label-to-token conversion
# ===========================================================================


def _build_char_labels(text: str, entities: List[Dict], labels: List[Dict]) -> List[int]:
    """Build character-level binary label array for a text chunk.

    Maps entity offsets to character positions, then looks up labels.
    Returns: list of ints (0 = linguistic, 1 = factual), one per char.
    """
    char_labels = [0] * len(text)

    # Build lookup: (start, end, text) → label
    label_lookup: Dict[str, str] = {}
    for span in labels:
        key = f"{span.get('text', '')}".lower()
        label_lookup[key] = span.get("label", "linguistic")

    # Mark factual entity positions in text
    for ent in entities:
        ent_text = ent.get("text", "").lower()
        ent_label = label_lookup.get(ent_text, "linguistic")

        if ent_label == "factual":
            start = ent.get("start", 0)
            end = ent.get("end", start + len(ent.get("text", "")))
            for i in range(start, min(end, len(text))):
                char_labels[i] = 1

    return char_labels


def _char_to_token_labels(
    char_labels: List[int],
    offsets: List[Tuple[int, int]],
) -> List[int]:
    """Convert character-level labels to token-level via majority vote."""
    token_labels = []
    for tok_start, tok_end in offsets:
        if tok_start == tok_end:
            token_labels.append(0)
            continue
        factual_chars = sum(char_labels[tok_start:tok_end])
        total_chars = tok_end - tok_start
        token_labels.append(1 if factual_chars > total_chars / 2 else 0)
    return token_labels


# ===========================================================================
# FAISS / SapBERT helpers
# ===========================================================================


def _load_sapbert(model_name: str, device: str = "cpu"):
    """Load SapBERT model and tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, tokenizer


def _embed_texts(texts: List[str], model, tokenizer, device: str = "cpu") -> np.ndarray:
    """Embed a batch of texts with SapBERT, returning (N, dim) array."""
    import torch

    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    encodings = tokenizer(
        texts, padding=True, truncation=True, max_length=64,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS pooling

    return embeddings.cpu().numpy().astype(np.float32)


def _load_faiss_index(index_path: Path):
    """Load a FAISS index."""
    import faiss

    if not index_path.exists():
        logger.warning("FAISS index not found: %s", index_path)
        return None
    return faiss.read_index(str(index_path))


def _search_faiss(index, query_emb: np.ndarray, k: int = 20):
    """Search FAISS index, return (distances, indices)."""
    if index is None or query_emb.size == 0:
        return np.array([]), np.array([])
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    distances, indices = index.search(query_emb, k)
    return distances, indices


# ===========================================================================
# Main preparation logic
# ===========================================================================


def prepare_training_data(
    cfg: FRLMConfig,
    *,
    skip_retrieval: bool = False,
    max_files: Optional[int] = None,
) -> Dict[str, int]:
    """Prepare tokenized training data for all three phases.

    Returns dict of counts: {router_examples, retrieval_examples, joint_examples}.
    """
    from transformers import AutoTokenizer

    corpus_dir = cfg.paths.resolve("corpus_dir")
    processed_dir = cfg.paths.resolve("processed_dir")
    labels_dir = cfg.paths.resolve("labels_dir")
    faiss_dir = cfg.paths.resolve("faiss_index_dir")

    sections = cfg.extraction.corpus.sections
    chunk_size = cfg.extraction.corpus.chunk_size
    chunk_overlap = cfg.extraction.corpus.chunk_overlap
    max_seq = cfg.model.backbone.max_seq_length

    # Output directories
    router_out = labels_dir / "tokenized"
    retrieval_out = processed_dir / "retrieval"
    joint_out = processed_dir / "joint"
    router_out.mkdir(parents=True, exist_ok=True)
    retrieval_out.mkdir(parents=True, exist_ok=True)
    joint_out.mkdir(parents=True, exist_ok=True)

    # Load backbone tokenizer
    logger.info("Loading tokenizer: %s", cfg.model.backbone.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SapBERT + FAISS for retrieval data (optional)
    sapbert_model = sapbert_tok = faiss_index = None
    emb_dim = cfg.model.retrieval_head.semantic.output_dim
    num_hard = cfg.faiss.hard_negatives.num_hard_negatives
    num_rand = cfg.faiss.hard_negatives.num_random_negatives
    num_neg = num_hard + num_rand

    if not skip_retrieval:
        try:
            sapbert_name = cfg.sapbert.model_name
            logger.info("Loading SapBERT: %s", sapbert_name)
            sapbert_model, sapbert_tok = _load_sapbert(sapbert_name, device="cpu")

            index_path = faiss_dir / "level_0_atomic.faiss"
            logger.info("Loading FAISS index: %s", index_path)
            faiss_index = _load_faiss_index(index_path)
            if faiss_index:
                logger.info("FAISS index loaded: %d vectors", faiss_index.ntotal)
        except Exception as e:
            logger.warning("Could not load SapBERT/FAISS: %s — skipping retrieval data", e)
            skip_retrieval = True

    # Discover labeled files
    label_files = sorted(labels_dir.glob("labels_PMC*.json"))
    if max_files:
        label_files = label_files[:max_files]

    logger.info("Found %d label files to process", len(label_files))

    counts = {"router_examples": 0, "retrieval_examples": 0, "joint_examples": 0}

    for lf_idx, label_path in enumerate(label_files, start=1):
        pmcid = label_path.stem.replace("labels_", "")
        xml_path = corpus_dir / f"{pmcid}.xml"
        ent_path = processed_dir / f"entities_{pmcid}.json"

        # --- Resume logic: skip already-processed files ---
        router_jsonl_path = router_out / f"router_{pmcid}.jsonl"
        retrieval_jsonl_path = retrieval_out / f"retrieval_{pmcid}.jsonl"
        router_done = router_jsonl_path.exists() and router_jsonl_path.stat().st_size > 0
        retrieval_done = skip_retrieval or (retrieval_jsonl_path.exists() and retrieval_jsonl_path.stat().st_size > 0)

        if router_done and retrieval_done:
            logger.info("[%d/%d] Skipping %s (already tokenized)", lf_idx, len(label_files), pmcid)
            # Count existing examples for accurate totals
            with open(router_jsonl_path) as f:
                counts["router_examples"] += sum(1 for _ in f)
            if retrieval_jsonl_path.exists():
                with open(retrieval_jsonl_path) as f:
                    n = sum(1 for _ in f)
                    counts["retrieval_examples"] += n
                    counts["joint_examples"] += n
            continue

        if not xml_path.exists():
            logger.warning("Corpus XML not found for %s — skipping", pmcid)
            continue
        if not ent_path.exists():
            logger.warning("Entity file not found for %s — skipping", pmcid)
            continue

        logger.info("[%d/%d] Processing %s", lf_idx, len(label_files), pmcid)

        # Load data
        with open(label_path) as f:
            label_data = json.load(f)
        with open(ent_path) as f:
            entities = json.load(f)

        spans = label_data.get("spans", [])

        # Re-extract text from XML (same chunking as step 02)
        parsed_sections = _parse_pmc_xml(xml_path, sections)

        # Group entities by (section, chunk_idx)
        ent_by_chunk: Dict[Tuple[str, int], List[Dict]] = {}
        for ent in entities:
            key = (ent.get("section", "abstract"), ent.get("chunk_idx", 0))
            ent_by_chunk.setdefault(key, []).append(ent)

        # Router JSONL file for this PMC article
        router_jsonl = router_out / f"router_{pmcid}.jsonl"
        retrieval_jsonl = retrieval_out / f"retrieval_{pmcid}.jsonl"
        joint_jsonl = joint_out / f"joint_{pmcid}.jsonl"

        router_lines: List[str] = []
        retrieval_lines: List[str] = []
        joint_lines: List[str] = []

        for sec in parsed_sections:
            section_name = sec["section"]
            chunks = _chunk_text(sec["text"], chunk_size, chunk_overlap)

            for chunk_idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                # Get entities for this chunk
                chunk_entities = ent_by_chunk.get((section_name, chunk_idx), [])

                # Build char-level labels
                char_labels = _build_char_labels(chunk_text, chunk_entities, spans)

                # Tokenize with offset mapping
                encoding = tokenizer(
                    chunk_text,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_seq,
                )
                input_ids = encoding["input_ids"]
                attention_mask = encoding.get("attention_mask", [1] * len(input_ids))
                offsets = encoding["offset_mapping"]

                # Convert to token-level labels
                router_labels = _char_to_token_labels(char_labels, offsets)

                if len(input_ids) == 0:
                    continue

                # --- Phase 1: Router data ---
                router_example = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "router_labels": router_labels,
                }
                router_lines.append(json.dumps(router_example))
                counts["router_examples"] += 1

                # --- Phase 2: Retrieval data ---
                if not skip_retrieval and sapbert_model is not None:
                    span_mask = router_labels[:]  # retrieval positions = factual tokens

                    # Find factual text spans for embedding
                    factual_texts = []
                    for ent in chunk_entities:
                        ent_text = ent.get("text", "").lower()
                        # Check if this entity is labeled factual
                        for sp in spans:
                            if sp.get("text", "").lower() == ent_text and sp.get("label") == "factual":
                                factual_texts.append(ent.get("text", ""))
                                break

                    if factual_texts:
                        # Embed the factual spans → use mean as positive
                        fact_embs = _embed_texts(factual_texts, sapbert_model, sapbert_tok)
                        positive_emb = fact_embs.mean(axis=0)

                        # Normalize
                        norm = np.linalg.norm(positive_emb)
                        if norm > 0:
                            positive_emb = positive_emb / norm

                        # Search FAISS for hard negatives
                        neg_embs = np.zeros((num_neg, emb_dim), dtype=np.float32)
                        if faiss_index is not None:
                            dists, idxs = _search_faiss(
                                faiss_index, positive_emb, k=num_neg + 1
                            )
                            # Skip first (self-match), use rest as negatives
                            if dists.size > 0:
                                neg_idxs = idxs[0, 1: num_neg + 1]
                                for ni, fi in enumerate(neg_idxs):
                                    if fi >= 0 and ni < num_neg:
                                        try:
                                            neg_embs[ni] = faiss_index.reconstruct(int(fi))
                                        except Exception:
                                            # PQ index can't reconstruct — use random
                                            neg_embs[ni] = np.random.randn(emb_dim).astype(np.float32)
                                            neg_embs[ni] /= max(np.linalg.norm(neg_embs[ni]), 1e-8)
                    else:
                        positive_emb = np.zeros(emb_dim, dtype=np.float32)
                        neg_embs = np.zeros((num_neg, emb_dim), dtype=np.float32)

                    retrieval_example = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "span_mask": span_mask,
                        "positive_embedding": positive_emb.tolist(),
                        "negative_embeddings": neg_embs.tolist(),
                    }
                    retrieval_lines.append(json.dumps(retrieval_example))
                    counts["retrieval_examples"] += 1

                    # --- Phase 3: Joint data ---
                    # token_labels = shifted input_ids (next-token prediction)
                    token_labels = input_ids[1:] + [-100]  # shift left, pad last

                    joint_example = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "router_labels": router_labels,
                        "span_mask": span_mask,
                        "positive_embedding": positive_emb.tolist(),
                        "negative_embeddings": neg_embs.tolist(),
                        "token_labels": token_labels,
                    }
                    joint_lines.append(json.dumps(joint_example))
                    counts["joint_examples"] += 1

        # Write JSONL files
        if router_lines:
            with open(router_jsonl, "w") as f:
                f.write("\n".join(router_lines) + "\n")

        if retrieval_lines:
            with open(retrieval_jsonl, "w") as f:
                f.write("\n".join(retrieval_lines) + "\n")

        if joint_lines:
            with open(joint_jsonl, "w") as f:
                f.write("\n".join(joint_lines) + "\n")

        logger.info(
            "  → %d router, %d retrieval, %d joint examples",
            len(router_lines), len(retrieval_lines), len(joint_lines),
        )

    logger.info("=== Training data preparation complete ===")
    for k, v in counts.items():
        logger.info("  %s: %d", k, v)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 6b: Convert span labels to tokenized training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--skip-retrieval", action="store_true",
        help="Skip retrieval/joint data (only prepare router data).",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Process only the first N label files.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    logger.info("Starting 06b_prepare_training_data with config: %s", args.config)

    t0 = time.time()
    try:
        counts = prepare_training_data(
            cfg,
            skip_retrieval=args.skip_retrieval,
            max_files=args.max_files,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Training data preparation failed.")
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info("06b_prepare_training_data completed in %.1fs", elapsed)
    logger.info("Total examples: router=%d, retrieval=%d, joint=%d",
                counts["router_examples"], counts["retrieval_examples"],
                counts["joint_examples"])


if __name__ == "__main__":
    main()


