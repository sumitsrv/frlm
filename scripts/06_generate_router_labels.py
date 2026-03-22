#!/usr/bin/env python3
"""
06_generate_router_labels.py — Generate router training labels using Claude API.

Labels text spans as **factual** (retrieval) or **linguistic** (generation)
using the :class:`LLMLabeler`.  Validates quality with :class:`LabelValidator`
and exports statistics + Label-Studio-formatted review files.

Pipeline position: Step 6 of 11
Reads from:  config.paths.processed_dir  (chunked text JSON)
Writes to:   config.paths.labels_dir     (label JSON, statistics, LS export)
Config used: config.labeling, config.paths
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, load_config, setup_logging
from src.labeling.llm_labeler import LLMLabeler, SpanLabel
from src.labeling.heuristic_labeler import HeuristicLabeler
from src.labeling.label_validator import LabelValidator
from src.status import PipelineStatusTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus loading helpers
# ---------------------------------------------------------------------------


def _discover_chunks(processed_dir: Path) -> List[Path]:
    """Find all processed text-chunk files ready for labeling.

    Looks for ``chunks_*.json`` and ``entities_*.json`` files produced by
    earlier pipeline stages.
    """
    patterns = ["chunks_*.json", "entities_*.json"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(processed_dir.glob(pattern)))
    logger.info("Found %d chunk files in %s", len(files), processed_dir)
    return files


def _load_texts_from_file(path: Path) -> List[str]:
    """Extract text strings from a processed-chunk JSON file.

    Supports two layouts:
    * A JSON list of objects each containing a ``"text"`` key.
    * A single JSON object with a ``"text"`` key.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        return [item.get("text", "") for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data.get("text", "")]
    return []


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost(
    total_texts: int,
    avg_text_length: int,
    cfg_labeling: Any,
) -> Dict[str, Any]:
    """Estimate total Claude API cost for a labeling run.

    Rough heuristic: ~1 token ≈ 4 chars for input;
    output ≈ 25 % of input tokens.
    Accounts for heuristic pre-labeling and multi-text batching.
    """
    # Model-aware pricing
    model_lower = cfg_labeling.model.lower()
    if "haiku" in model_lower and ("4-5" in model_lower or "4.5" in model_lower):
        input_price, output_price = 1.00, 5.00
    elif "haiku" in model_lower:
        input_price, output_price = 0.25, 1.25
    else:
        input_price, output_price = 3.00, 15.00

    # Account for heuristic pre-labeling
    use_heuristic = getattr(cfg_labeling, "use_heuristic", False)
    api_batch_size = getattr(cfg_labeling, "api_batch_size", 1)
    heuristic_ratio = 0.55 if use_heuristic else 0.0  # ~55% handled locally
    api_texts = int(total_texts * (1 - heuristic_ratio))

    # With batching, prompt overhead is shared across batch
    num_api_calls = max(1, api_texts // max(api_batch_size, 1))
    prompt_overhead_tokens = 1250  # system + few-shot per call
    text_tokens_per_text = avg_text_length / 4

    total_input = int(
        num_api_calls * prompt_overhead_tokens
        + api_texts * text_tokens_per_text
    )
    total_output = int(api_texts * 8)  # ~8 tokens per short text response

    input_cost = total_input * input_price / 1_000_000
    output_cost = total_output * output_price / 1_000_000
    total_cost = input_cost + output_cost

    return {
        "total_texts": total_texts,
        "texts_heuristic": total_texts - api_texts,
        "texts_api": api_texts,
        "api_calls": num_api_calls,
        "api_batch_size": api_batch_size,
        "avg_text_length_chars": avg_text_length,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_cost_usd": round(total_cost, 2),
        "model": cfg_labeling.model,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def generate_labels(cfg: FRLMConfig, *, max_files: Optional[int] = None) -> None:
    """Orchestrate label generation across all processed chunks.

    Steps:
    1. Discover chunk files in ``processed_dir``.
    2. For each file, extract texts and label via :class:`LLMLabeler`.
    3. Validate each labelled set with :class:`LabelValidator`.
    4. Save per-file labels + per-corpus statistics.
    5. Export low-confidence spans in Label-Studio format for human review.
    """
    labeling_cfg = cfg.labeling
    processed_dir = cfg.paths.resolve("processed_dir")
    labels_dir = cfg.paths.resolve("labels_dir")
    labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Router Label Generation (Claude API) ===")
    logger.info("Model: %s", labeling_cfg.model)
    logger.info("Batch size: %d chunks per call", labeling_cfg.batch_size)
    logger.info(
        "IAA samples: %d, Min kappa: %.2f",
        labeling_cfg.inter_annotator_samples,
        labeling_cfg.min_agreement_threshold,
    )

    # -- 1. Discover chunk files -----------------------------------------
    chunk_files = _discover_chunks(processed_dir)
    if not chunk_files:
        logger.warning("No chunk files found in %s. Run step 02 first.", processed_dir)
        return
    if max_files is not None:
        chunk_files = chunk_files[:max_files]
        logger.info("Limiting to %d files (--max-files)", max_files)

    # -- 2. Cost estimate ------------------------------------------------
    sample_texts: List[str] = []
    for cf in chunk_files[:10]:
        sample_texts.extend(_load_texts_from_file(cf))
    if sample_texts:
        avg_len = sum(len(t) for t in sample_texts) // max(len(sample_texts), 1)
        total_count = sum(
            len(_load_texts_from_file(cf)) for cf in chunk_files
        )
        cost_est = estimate_cost(total_count, avg_len, labeling_cfg)
        logger.info("Cost estimate: %s", json.dumps(cost_est, indent=2))
        est_path = labels_dir / "cost_estimate.json"
        with open(est_path, "w") as fh:
            json.dump(cost_est, fh, indent=2)

    # -- 3. Label each file (hybrid: heuristic → batch-LLM) ---------------
    use_heuristic = getattr(labeling_cfg, "use_heuristic", True)
    api_batch_size = getattr(labeling_cfg, "api_batch_size", 50)

    labeler = LLMLabeler.from_config(labeling_cfg)
    heuristic = HeuristicLabeler() if use_heuristic else None
    validator = LabelValidator()
    tracker = PipelineStatusTracker()
    tracker.mark_running(6, total_items=len(chunk_files))

    if use_heuristic:
        logger.info("Heuristic pre-labeling ENABLED — obvious spans handled locally")
    logger.info("API batch size: %d texts per call", api_batch_size)

    start_time = time.time()
    total_labeled = 0
    total_valid = 0
    total_invalid = 0
    completed_files = 0
    skipped_files = 0
    heuristic_count = 0
    api_count = 0
    all_spans: List[SpanLabel] = []
    review_records: List[Dict[str, Any]] = []

    for file_idx, chunk_path in enumerate(chunk_files, start=1):
        label_filename = chunk_path.stem.replace("entities_", "labels_").replace(
            "chunks_", "labels_"
        ) + ".json"
        label_path = labels_dir / label_filename

        # Resume: skip already-labelled files
        if label_path.exists():
            logger.debug("Skipping already labelled: %s", chunk_path.name)
            skipped_files += 1
            completed_files += 1
            continue

        try:
            texts = _load_texts_from_file(chunk_path)
            file_spans: List[SpanLabel] = []

            # --- Phase A: heuristic pre-labeling --------------------------
            needs_llm_indices: List[int] = []   # indices into `texts`
            needs_llm_texts: List[str] = []

            for i, text in enumerate(texts):
                if not text.strip():
                    continue

                if heuristic is not None:
                    span = heuristic.try_label(text)
                    if span is not None:
                        file_spans.append(span)
                        heuristic_count += 1
                        continue

                # Deferred to LLM
                needs_llm_indices.append(i)
                needs_llm_texts.append(text)

            # --- Phase B: batch LLM labeling for remaining texts ----------
            if needs_llm_texts:
                try:
                    batch_results = labeler.label_texts_batch(
                        needs_llm_texts, api_batch_size=api_batch_size,
                    )
                    for j, spans in enumerate(batch_results):
                        file_spans.extend(spans)
                    api_count += len(needs_llm_texts)
                except Exception as exc:
                    logger.error(
                        "Batch labeling failed for %s: %s",
                        chunk_path.name, exc,
                    )
                    # Fallback: label remaining one-by-one
                    for text in needs_llm_texts:
                        try:
                            spans = labeler.label_text(text)
                        except Exception as inner_exc:
                            logger.error(
                                "Fallback labeling failed in %s: %s",
                                chunk_path.name, inner_exc,
                            )
                            spans = []
                        file_spans.extend(spans)
                        api_count += 1

            all_spans.extend(file_spans)

            # Validate
            text_length = sum(len(t) for t in texts)
            vcfg = labeling_cfg.validation
            is_valid, issues = validator.validate_labels(
                file_spans,
                text_length,
                min_retrieval_ratio=vcfg.min_retrieval_ratio,
                max_retrieval_ratio=vcfg.max_retrieval_ratio,
                min_spans_per_chunk=vcfg.min_spans_per_chunk,
                max_spans_per_chunk=vcfg.max_spans_per_chunk,
            )

            if not is_valid:
                logger.warning(
                    "Validation failed for %s: %s", chunk_path.name, issues
                )
                total_invalid += 1
            else:
                total_valid += 1

            total_labeled += len(file_spans)

            # Collect low-confidence for review
            low_conf = validator.find_low_confidence(file_spans, threshold=0.7)
            if low_conf:
                for t in texts:
                    review_records.append({"text": t, "spans": low_conf})

            # Save per-file labels atomically
            output = {
                "source_file": chunk_path.name,
                "spans": [s.model_dump() for s in file_spans],
                "num_spans": len(file_spans),
                "valid": is_valid,
                "issues": issues,
            }
            tmp_path = label_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(output, fh, indent=2, ensure_ascii=False)
            tmp_path.replace(label_path)
            completed_files += 1

        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to process %s: %s", chunk_path.name, exc)

        if file_idx % 10 == 0 or file_idx == len(chunk_files):
            logger.info(
                "Progress: %d/%d files | %d spans | heuristic %d, api %d | "
                "valid %d, invalid %d | cost $%.4f",
                file_idx,
                len(chunk_files),
                total_labeled,
                heuristic_count,
                api_count,
                total_valid,
                total_invalid,
                labeler.cost_tracker.estimated_cost_usd,
            )
            tracker.update_progress(
                6,
                completed_items=completed_files,
                skipped_items=skipped_files,
                failed_items=total_invalid,
                cost_usd=labeler.cost_tracker.estimated_cost_usd,
            )
            tracker.save()

    elapsed = time.time() - start_time

    # -- Update tracker with final state ----------------------------------
    tracker.update_progress(
        6,
        completed_items=completed_files,
        skipped_items=skipped_files,
        failed_items=total_invalid,
        cost_usd=labeler.cost_tracker.estimated_cost_usd,
    )
    if completed_files >= len(chunk_files):
        tracker.mark_completed(6)
    else:
        tracker.mark_partial(6)

    # -- 4. Corpus-level statistics --------------------------------------
    stats = validator.compute_statistics(all_spans)
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["files_total"] = len(chunk_files)
    stats["files_valid"] = total_valid
    stats["files_invalid"] = total_invalid
    stats["heuristic_labeled"] = heuristic_count
    stats["api_labeled"] = api_count
    stats["cost"] = labeler.cost_tracker.summary()
    if heuristic is not None:
        stats["heuristic_stats"] = heuristic.stats

    stats_path = labels_dir / "label_statistics.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Statistics written to %s", stats_path)

    # -- 5. Export low-confidence spans for human review -----------------
    if review_records:
        review_path = labels_dir / "review_export.json"
        validator.export_corpus_for_review(review_records, review_path)
        logger.info("Exported %d review records to %s", len(review_records), review_path)

    # -- Summary ---------------------------------------------------------
    logger.info("=== Label Generation Summary ===")
    logger.info(
        "Files: %d | Spans: %d | Valid: %d | Invalid: %d | Time: %.1fs | Cost: $%.4f",
        len(chunk_files),
        total_labeled,
        total_valid,
        total_invalid,
        elapsed,
        labeler.cost_tracker.estimated_cost_usd,
    )


# ---------------------------------------------------------------------------
# Validate-only mode
# ---------------------------------------------------------------------------


def validate_existing(cfg: FRLMConfig) -> None:
    """Re-validate existing labels without calling the API."""
    labels_dir = cfg.paths.resolve("labels_dir")
    if not labels_dir.exists():
        logger.error("Labels directory does not exist: %s", labels_dir)
        return

    validator = LabelValidator()
    label_files = sorted(labels_dir.glob("labels_*.json"))
    logger.info("Re-validating %d label files in %s", len(label_files), labels_dir)

    all_spans: List[SpanLabel] = []
    for lf in label_files:
        with open(lf) as fh:
            data = json.load(fh)
        spans = [SpanLabel(**s) for s in data.get("spans", [])]
        all_spans.extend(spans)

    stats = validator.compute_statistics(all_spans)
    low = validator.find_low_confidence(all_spans, threshold=0.7)
    stats["low_confidence_count"] = len(low)

    stats_path = labels_dir / "label_statistics.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    logger.info("Validation complete — %d spans, %d low-confidence", len(all_spans), len(low))
    logger.info("Statistics: %s", json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, load config, and run."""
    parser = argparse.ArgumentParser(
        description="Generate router training labels using Claude API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Limit the number of chunk files to process.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing labels without generating new ones.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    logger.info("Starting 06_generate_router_labels with config: %s", args.config)

    try:
        if args.validate_only:
            validate_existing(cfg)
        else:
            generate_labels(cfg, max_files=args.max_files)
    except KeyboardInterrupt:
        logger.warning("Label generation interrupted.")
        tracker = PipelineStatusTracker()
        tracker.mark_partial(6)
        sys.exit(130)
    except Exception:
        logger.exception("Label generation failed.")
        tracker = PipelineStatusTracker()
        tracker.mark_failed(6, "Unhandled exception")
        sys.exit(1)

    logger.info("06_generate_router_labels completed successfully.")


if __name__ == "__main__":
    main()