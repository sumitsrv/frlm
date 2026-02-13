#!/usr/bin/env python3
"""
06_generate_router_labels.py - Generate router training labels using Claude API.

Labels text spans as RETRIEVAL (factual claims) or GENERATION (linguistic glue)
using Claude. Includes label quality validation and inter-annotator agreement.

Pipeline position: Step 6 of 11
Reads from:  config.paths.processed_dir (chunked text)
Writes to:   config.paths.labels_dir (label JSON)
Config used: config.labeling, config.paths
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FRLMConfig, load_config, setup_logging

logger = logging.getLogger(__name__)


def _discover_chunks(processed_dir: Path) -> List[Path]:
    """Find all processed text chunk files ready for labeling."""
    patterns = ["entities_*.json"]
    files = []
    for pattern in patterns:
        files.extend(sorted(processed_dir.glob(pattern)))
    logger.info("Found %d chunk files in %s", len(files), processed_dir)
    return files


def _call_labeling_api(
    chunks: List[str],
    labeling_cfg: Any,
) -> List[Dict[str, Any]]:
    """Call Claude API to label text spans as RETRIEVAL or GENERATION.

    Returns list of label dicts with character offsets.
    """
    for attempt in range(1, labeling_cfg.max_retries + 1):
        try:
            logger.debug(
                "Labeling API call (attempt %d/%d, %d chunks)",
                attempt, labeling_cfg.max_retries, len(chunks),
            )

            # Production:
            #   import anthropic
            #   client = anthropic.Anthropic(api_key=labeling_cfg.api_key)
            #   text_block = "\n---\n".join(chunks)
            #   response = client.messages.create(
            #       model=labeling_cfg.model,
            #       max_tokens=labeling_cfg.max_tokens,
            #       temperature=labeling_cfg.temperature,
            #       system=labeling_cfg.system_prompt,
            #       messages=[{"role": "user", "content": f"Label the following text:\n\n{text_block}"}],
            #   )
            #   return json.loads(response.content[0].text)

            logger.debug("Would label %d chunks via Claude %s", len(chunks), labeling_cfg.model)
            return []

        except Exception as exc:
            logger.warning("Labeling API call failed (attempt %d): %s", attempt, exc)
            if attempt < labeling_cfg.max_retries:
                time.sleep(labeling_cfg.retry_delay * attempt)

    logger.error("All labeling API attempts exhausted")
    return []


def _validate_labels(
    labels: List[Dict[str, Any]],
    text_length: int,
    validation_cfg: Any,
) -> Tuple[bool, List[str]]:
    """Validate label quality against configured thresholds.

    Returns (is_valid, list_of_issues).
    """
    issues: List[str] = []

    if not labels:
        issues.append("No labels produced")
        return False, issues

    num_spans = len(labels)
    if num_spans < validation_cfg.min_spans_per_chunk:
        issues.append(f"Too few spans: {num_spans} < {validation_cfg.min_spans_per_chunk}")
    if num_spans > validation_cfg.max_spans_per_chunk:
        issues.append(f"Too many spans: {num_spans} > {validation_cfg.max_spans_per_chunk}")

    retrieval_chars = sum(
        (l.get("end", 0) - l.get("start", 0))
        for l in labels
        if l.get("label") == "RETRIEVAL"
    )
    if text_length > 0:
        ratio = retrieval_chars / text_length
        if ratio < validation_cfg.min_retrieval_ratio:
            issues.append(f"Retrieval ratio too low: {ratio:.3f} < {validation_cfg.min_retrieval_ratio}")
        if ratio > validation_cfg.max_retrieval_ratio:
            issues.append(f"Retrieval ratio too high: {ratio:.3f} > {validation_cfg.max_retrieval_ratio}")

    is_valid = len(issues) == 0
    return is_valid, issues


def _compute_inter_annotator_agreement(
    labels_a: List[Dict[str, Any]],
    labels_b: List[Dict[str, Any]],
    text_length: int,
) -> float:
    """Compute Cohen's kappa between two label sets at character level."""
    if text_length == 0:
        return 0.0

    # Build character-level arrays
    arr_a = [0] * text_length
    arr_b = [0] * text_length

    for l in labels_a:
        for i in range(l.get("start", 0), min(l.get("end", 0), text_length)):
            if l.get("label") == "RETRIEVAL":
                arr_a[i] = 1

    for l in labels_b:
        for i in range(l.get("start", 0), min(l.get("end", 0), text_length)):
            if l.get("label") == "RETRIEVAL":
                arr_b[i] = 1

    # Cohen's kappa
    n = text_length
    agreement = sum(1 for a, b in zip(arr_a, arr_b) if a == b)
    p_o = agreement / n

    p_a_1 = sum(arr_a) / n
    p_b_1 = sum(arr_b) / n
    p_e = p_a_1 * p_b_1 + (1 - p_a_1) * (1 - p_b_1)

    if abs(1 - p_e) < 1e-10:
        return 1.0 if abs(p_o - 1.0) < 1e-10 else 0.0

    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def generate_labels(cfg: FRLMConfig) -> None:
    """Orchestrate label generation across all processed chunks."""
    labeling_cfg = cfg.labeling
    processed_dir = cfg.paths.resolve("processed_dir")
    labels_dir = cfg.paths.resolve("labels_dir")
    labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Router Label Generation (Claude API) ===")
    logger.info("Model: %s", labeling_cfg.model)
    logger.info("Batch size: %d chunks per call", labeling_cfg.batch_size)
    logger.info("IAA samples: %d, Min kappa: %.2f",
                labeling_cfg.inter_annotator_samples, labeling_cfg.min_agreement_threshold)

    chunk_files = _discover_chunks(processed_dir)
    if not chunk_files:
        logger.warning("No chunk files found. Run step 02 first.")
        return

    start_time = time.time()
    total_labeled = 0
    total_valid = 0
    total_invalid = 0

    for idx, chunk_path in enumerate(chunk_files, start=1):
        label_filename = chunk_path.name.replace("entities_", "labels_")
        label_path = labels_dir / label_filename

        if label_path.exists():
            logger.debug("Skipping already labeled: %s", chunk_path.name)
            continue

        try:
            with open(chunk_path, "r", encoding="utf-8") as fh:
                entities = json.load(fh)

            chunks = [e.get("text", "") for e in (entities if isinstance(entities, list) else [entities])]
            batch_size = labeling_cfg.batch_size

            all_labels: List[Dict[str, Any]] = []
            for i in range(0, max(len(chunks), 1), batch_size):
                batch = chunks[i : i + batch_size]
                labels = _call_labeling_api(batch, labeling_cfg)
                all_labels.extend(labels)

            # Validate
            text_length = sum(len(c) for c in chunks)
            is_valid, issues = _validate_labels(all_labels, text_length, labeling_cfg.validation)

            if not is_valid:
                logger.warning("Label validation failed for %s: %s", chunk_path.name, issues)
                total_invalid += 1
            else:
                total_valid += 1

            total_labeled += len(all_labels)

            # Save labels
            output = {
                "source_file": chunk_path.name,
                "labels": all_labels,
                "valid": is_valid,
                "issues": issues,
            }
            tmp_path = label_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(output, fh, indent=2, ensure_ascii=False)
            tmp_path.replace(label_path)

        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to process %s: %s", chunk_path.name, exc)

        if idx % 10 == 0 or idx == len(chunk_files):
            logger.info("Labeling progress: %d/%d files, %d labels, %d valid, %d invalid",
                        idx, len(chunk_files), total_labeled, total_valid, total_invalid)

    total_time = time.time() - start_time
    logger.info("=== Label Generation Summary ===")
    logger.info("Files: %d, Labels: %d, Valid: %d, Invalid: %d, Time: %.2fs",
                len(chunk_files), total_labeled, total_valid, total_invalid, total_time)


def main() -> None:
    """Parse arguments, load config, and generate router labels."""
    parser = argparse.ArgumentParser(
        description="Generate router training labels using Claude API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing labels without generating new ones.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    logger.info("Starting 06_generate_router_labels with config: %s", args.config)

    try:
        generate_labels(cfg)
    except KeyboardInterrupt:
        logger.warning("Label generation interrupted.")
        sys.exit(130)
    except Exception:
        logger.exception("Label generation failed.")
        sys.exit(1)

    logger.info("06_generate_router_labels completed successfully.")


if __name__ == "__main__":
    main()