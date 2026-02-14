#!/usr/bin/env python3
"""
01_download_corpus.py - Download PMC Open Access subset XML files.

Downloads biomedical papers from PubMed Central's Open Access subset,
filters by configured journals and year range, and stores raw XML files
in the corpus directory.

Pipeline position: Step 1 of 11
Reads from:  PMC OA FTP / S3 (remote)
Writes to:   config.paths.corpus_dir
Config used: config.extraction.corpus, config.paths
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure the parent package is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FRLMConfig, load_config, setup_logging

logger = logging.getLogger(__name__)

PMC_OA_FILE_LIST_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv"
)
PMC_OA_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"


def _download_file_list(cache_dir: Path) -> Path:
    """Download the PMC OA file list CSV and cache it locally.

    Returns the path to the cached file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / "oa_file_list.csv"

    if cached_path.exists():
        age_hours = (time.time() - cached_path.stat().st_mtime) / 3600
        if age_hours < 24:
            logger.info(
                "Using cached OA file list (%.1f hours old): %s",
                age_hours,
                cached_path,
            )
            return cached_path

    logger.info("Downloading PMC OA file list from %s", PMC_OA_FILE_LIST_URL)
    tmp_path = cached_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(PMC_OA_FILE_LIST_URL, str(tmp_path))
        tmp_path.replace(cached_path)
        logger.info("File list downloaded: %s", cached_path)
    except Exception as exc:
        logger.error("Failed to download file list: %s", exc)
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return cached_path


def _parse_file_list(
    csv_path: Path,
    corpus_cfg: object,
) -> List[Dict[str, str]]:
    """Parse the OA file list CSV and apply configured filters.

    Filters by journal, year range, and max document count.

    Returns a list of dicts with keys: path, journal, pmcid, year.
    """
    import csv

    entries: List[Dict[str, str]] = []
    filter_journals = set(getattr(corpus_cfg, "filter_journals", []))
    min_year = getattr(corpus_cfg, "min_year", 2000)
    max_year = getattr(corpus_cfg, "max_year", None)
    max_docs = getattr(corpus_cfg, "max_documents", None)

    logger.info(
        "Parsing file list: journals=%s, year_range=[%s, %s], max_docs=%s",
        filter_journals or "all",
        min_year,
        max_year or "now",
        max_docs or "unlimited",
    )

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            year_str = row.get("Year", row.get("year", "0"))
            try:
                year = int(year_str)
            except ValueError:
                continue

            if year < min_year:
                continue
            if max_year is not None and year > max_year:
                continue

            journal = row.get("Journal Title", row.get("journal", ""))
            if filter_journals and journal not in filter_journals:
                continue

            entries.append(
                {
                    "path": row.get("File", row.get("file", "")),
                    "journal": journal,
                    "pmcid": row.get("Accession ID", row.get("pmcid", "")),
                    "year": str(year),
                }
            )

            if max_docs is not None and len(entries) >= max_docs:
                break

    logger.info("Found %d papers matching filters", len(entries))
    return entries


def _download_paper(
    entry: Dict[str, str],
    corpus_dir: Path,
) -> Optional[Path]:
    """Download a single paper XML from PMC.

    Uses atomic write (download to .tmp then rename) to handle interruptions.

    Returns the final path on success, None on failure.
    """
    relative_path = entry["path"]
    pmcid = entry["pmcid"]
    url = PMC_OA_BASE_URL + relative_path

    output_path = corpus_dir / f"{pmcid}.xml"
    if output_path.exists():
        logger.debug("Already downloaded: %s", pmcid)
        return output_path

    tmp_path = output_path.with_suffix(".xml.tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        tmp_path.replace(output_path)
        return output_path
    except Exception as exc:
        logger.warning("Failed to download %s: %s", pmcid, exc)
        if tmp_path.exists():
            tmp_path.unlink()
        return None


def _verify_downloads(corpus_dir: Path, expected_count: int) -> Dict[str, int]:
    """Verify downloaded files and return statistics."""
    xml_files = list(corpus_dir.glob("*.xml"))
    total_bytes = sum(f.stat().st_size for f in xml_files)

    stats = {
        "expected": expected_count,
        "downloaded": len(xml_files),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }

    logger.info(
        "Verification: %d/%d files downloaded (%.2f MB total)",
        stats["downloaded"],
        stats["expected"],
        stats["total_mb"],
    )
    return stats


def download_corpus(cfg: FRLMConfig) -> None:
    """Orchestrate the corpus download pipeline.

    1. Download/cache OA file list.
    2. Parse and filter entries.
    3. Download papers with progress logging.
    4. Verify downloads.
    """
    corpus_cfg = cfg.extraction.corpus
    corpus_dir = cfg.paths.resolve("corpus_dir")
    cache_dir = cfg.paths.resolve("cache_dir")

    corpus_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Corpus Download (PMC Open Access) ===")
    logger.info("Source: %s", corpus_cfg.source)
    logger.info("Output directory: %s", corpus_dir)
    logger.info(
        "Year range: [%d, %s]",
        corpus_cfg.min_year,
        corpus_cfg.max_year or "present",
    )
    logger.info(
        "Max documents: %s",
        corpus_cfg.max_documents or "unlimited",
    )

    # Step 1: file list
    csv_path = _download_file_list(cache_dir)

    # Step 2: filter
    entries = _parse_file_list(csv_path, corpus_cfg)
    if not entries:
        logger.warning("No papers found matching the configured filters.")
        return

    # Step 3: download
    start_time = time.time()
    success_count = 0
    fail_count = 0

    for idx, entry in enumerate(entries, start=1):
        result = _download_paper(entry, corpus_dir)
        if result is not None:
            success_count += 1
        else:
            fail_count += 1

        if idx % 100 == 0 or idx == len(entries):
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            logger.info(
                "Download progress: %d/%d (%.1f papers/sec, %d failures)",
                idx,
                len(entries),
                rate,
                fail_count,
            )

    download_time = time.time() - start_time

    # Step 4: verify
    stats = _verify_downloads(corpus_dir, len(entries))
    stats["download_time_seconds"] = round(download_time, 2)
    stats["failures"] = fail_count

    logger.info("=== Download Summary ===")
    logger.info("Total time: %.2f seconds", download_time)
    logger.info("Successful: %d, Failed: %d", success_count, fail_count)
    logger.info("Total size: %.2f MB", stats["total_mb"])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, load config, and download the corpus."""
    parser = argparse.ArgumentParser(
        description="Download PMC Open Access corpus XML files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Override config.extraction.corpus.max_documents.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List papers to download without actually downloading.",
    )
    args = parser.parse_args()

    overrides = {}
    if args.max_docs is not None:
        overrides["extraction.corpus.max_documents"] = args.max_docs

    cfg = load_config(args.config, overrides=overrides if overrides else None)
    setup_logging(cfg)

    logger.info("Starting 01_download_corpus with config: %s", args.config)

    try:
        download_corpus(cfg)
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Corpus download failed with an unexpected error.")
        sys.exit(1)

    logger.info("01_download_corpus completed successfully.")


if __name__ == "__main__":
    main()