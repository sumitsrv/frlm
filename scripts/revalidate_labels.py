#!/usr/bin/env python3
"""
revalidate_labels.py — Re-validate label files with updated thresholds.

Reads all labels_PMC*.json in data/labels/, re-checks them against the
current config thresholds, and updates the 'valid' flag in-place.
Use after widening min_retrieval_ratio / max_retrieval_ratio in config.

Usage:
    python scripts/revalidate_labels.py --config config/default.yaml
    python scripts/revalidate_labels.py --config config/default.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, load_config, setup_logging

logger = logging.getLogger(__name__)


def revalidate_labels(cfg: FRLMConfig, *, dry_run: bool = False) -> dict:
    """Re-validate all label files against current config thresholds.

    Returns dict with counts: {total, previously_valid, previously_invalid,
    now_valid, now_invalid, flipped}.
    """
    labels_dir = cfg.paths.resolve("labels_dir")
    min_ratio = cfg.labeling.validation.min_retrieval_ratio
    max_ratio = cfg.labeling.validation.max_retrieval_ratio
    min_spans = cfg.labeling.validation.min_spans_per_chunk
    max_spans = cfg.labeling.validation.max_spans_per_chunk

    logger.info("Re-validating labels with thresholds:")
    logger.info("  min_retrieval_ratio: %.3f", min_ratio)
    logger.info("  max_retrieval_ratio: %.3f", max_ratio)
    logger.info("  min_spans: %d, max_spans: %d", min_spans, max_spans)

    label_files = sorted(labels_dir.glob("labels_PMC*.json"))
    counts = {
        "total": len(label_files),
        "previously_valid": 0,
        "previously_invalid": 0,
        "now_valid": 0,
        "now_invalid": 0,
        "flipped": 0,
    }

    for lf in label_files:
        with open(lf) as f:
            data = json.load(f)

        was_valid = data.get("valid", True)
        if was_valid:
            counts["previously_valid"] += 1
        else:
            counts["previously_invalid"] += 1

        # Re-validate
        spans = data.get("spans", [])
        num_spans = len(spans)
        issues = []

        if num_spans < min_spans:
            issues.append(f"Too few spans: {num_spans} < {min_spans}")
        if num_spans > max_spans:
            issues.append(f"Too many spans: {num_spans} > {max_spans}")

        # Compute retrieval ratio from spans
        factual = sum(1 for s in spans if s.get("label") == "factual")
        total = len(spans) if spans else 1
        ratio = factual / total

        if ratio < min_ratio:
            issues.append(f"Retrieval ratio too low: {ratio:.3f} < {min_ratio}")
        if ratio > max_ratio:
            issues.append(f"Retrieval ratio too high: {ratio:.3f} > {max_ratio}")

        is_valid = len(issues) == 0

        if is_valid:
            counts["now_valid"] += 1
        else:
            counts["now_invalid"] += 1

        if is_valid != was_valid:
            counts["flipped"] += 1
            pmcid = lf.stem.replace("labels_", "")
            direction = "invalid→valid" if is_valid else "valid→invalid"
            logger.info("  %s: %s %s", pmcid, direction,
                        f"(was: {data.get('issues', [])})" if is_valid else f"(issues: {issues})")

        if not dry_run:
            data["valid"] = is_valid
            data["issues"] = issues
            with open(lf, "w") as f:
                json.dump(data, f, indent=2)

    logger.info("=== Re-validation complete ===")
    logger.info("  Total: %d", counts["total"])
    logger.info("  Previously valid: %d, invalid: %d",
                counts["previously_valid"], counts["previously_invalid"])
    logger.info("  Now valid: %d, invalid: %d",
                counts["now_valid"], counts["now_invalid"])
    logger.info("  Flipped: %d", counts["flipped"])

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-validate label files with updated thresholds.",
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying files.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    revalidate_labels(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

