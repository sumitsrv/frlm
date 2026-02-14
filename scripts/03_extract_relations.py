#!/usr/bin/env python3
"""
03_extract_relations.py - Extract relations using Claude API.

Reads entity-annotated text and uses the Claude API to extract structured
biomedical relations (TREATS, CAUSES, INHIBITS, etc.) with evidence spans.

Pipeline position: Step 3 of 11
Reads from:  config.paths.processed_dir (entity JSON)
Writes to:   config.paths.processed_dir (relation JSON)
Config used: config.extraction.relation, config.paths
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

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._request_times: List[float] = []
        self._token_counts: List[tuple] = []  # (timestamp, token_count)

    def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Block until a request can be made within rate limits."""
        now = time.time()
        cutoff = now - 60.0

        self._request_times = [t for t in self._request_times if t > cutoff]
        self._token_counts = [(t, c) for t, c in self._token_counts if t > cutoff]

        while len(self._request_times) >= self._rpm:
            sleep_time = self._request_times[0] - cutoff + 0.1
            logger.debug("RPM limit reached, sleeping %.1fs", sleep_time)
            time.sleep(max(sleep_time, 0.1))
            now = time.time()
            cutoff = now - 60.0
            self._request_times = [t for t in self._request_times if t > cutoff]

        current_tokens = sum(c for _, c in self._token_counts)
        while current_tokens + estimated_tokens > self._tpm:
            sleep_time = self._token_counts[0][0] - cutoff + 0.1
            logger.debug("TPM limit reached, sleeping %.1fs", sleep_time)
            time.sleep(max(sleep_time, 0.1))
            now = time.time()
            cutoff = now - 60.0
            self._token_counts = [(t, c) for t, c in self._token_counts if t > cutoff]
            current_tokens = sum(c for _, c in self._token_counts)

        self._request_times.append(time.time())
        self._token_counts.append((time.time(), estimated_tokens))


def _build_relation_prompt(
    text: str,
    entities: List[Dict[str, Any]],
    system_prompt: str,
) -> Dict[str, str]:
    """Build the prompt for Claude API relation extraction."""
    entity_summary = "\n".join(
        f"  - {e.get('text', 'N/A')} ({e.get('cui', 'unknown')}, {e.get('label', 'unknown')})"
        for e in entities[:50]
    )

    user_msg = (
        f"Extract biomedical relations from this text.\n\n"
        f"Text:\n{text}\n\n"
        f"Identified entities:\n{entity_summary}\n\n"
        f"Return a JSON array where each element has: "
        f"subject, relation_type, object, confidence (0-1), evidence_span.\n"
        f"Valid relation types: TREATS, CAUSES, INHIBITS, ACTIVATES, "
        f"ASSOCIATED_WITH, METABOLIZED_BY, CONTRAINDICATED_WITH, "
        f"DOSAGE_OF, BIOMARKER_FOR, RESISTANCE_TO."
    )

    return {"system": system_prompt, "user": user_msg}


def _call_claude_api(
    prompt: Dict[str, str],
    relation_cfg: Any,
    rate_limiter: RateLimiter,
) -> List[Dict[str, Any]]:
    """Call the Claude API for relation extraction with retry logic."""
    rate_limiter.wait_if_needed(estimated_tokens=relation_cfg.max_tokens)

    for attempt in range(1, relation_cfg.max_retries + 1):
        try:
            logger.debug(
                "Claude API call (attempt %d/%d, model=%s)",
                attempt,
                relation_cfg.max_retries,
                relation_cfg.model,
            )

            import anthropic
            client = anthropic.Anthropic(api_key=relation_cfg.api_key)
            response = client.messages.create(
                model=relation_cfg.model,
                max_tokens=relation_cfg.max_tokens,
                temperature=relation_cfg.temperature,
                system=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )
            return json.loads(response.content[0].text)

        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse API response (attempt %d/%d)",
                attempt,
                relation_cfg.max_retries,
            )
        except Exception as exc:
            logger.warning(
                "API call failed (attempt %d/%d): %s",
                attempt,
                relation_cfg.max_retries,
                exc,
            )

        if attempt < relation_cfg.max_retries:
            delay = relation_cfg.retry_delay * attempt
            logger.debug("Retrying in %.1fs", delay)
            time.sleep(delay)

    logger.error("All %d API attempts exhausted", relation_cfg.max_retries)
    return []


def extract_relations(cfg: FRLMConfig) -> None:
    """Orchestrate relation extraction across all entity files.

    1. Discover entity files.
    2. Initialize rate limiter.
    3. For each file, load entities, call Claude API, save relations.
    4. Write summary.
    """
    relation_cfg = cfg.extraction.relation
    processed_dir = cfg.paths.resolve("processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Relation Extraction (Claude API) ===")
    logger.info("Model: %s", relation_cfg.model)
    logger.info("Max tokens: %d, Temperature: %.1f", relation_cfg.max_tokens, relation_cfg.temperature)
    logger.info("Batch size: %d, Rate limits: %d RPM / %d TPM",
                relation_cfg.batch_size, relation_cfg.rate_limit_rpm, relation_cfg.rate_limit_tpm)

    entity_files = sorted(processed_dir.glob("entities_*.json"))
    if not entity_files:
        logger.warning("No entity files found. Run step 02 first.")
        return

    logger.info("Found %d entity files", len(entity_files))

    rate_limiter = RateLimiter(
        requests_per_minute=relation_cfg.rate_limit_rpm,
        tokens_per_minute=relation_cfg.rate_limit_tpm,
    )

    start_time = time.time()
    total_relations = 0
    total_api_calls = 0
    errors = 0

    for idx, entity_path in enumerate(entity_files, start=1):
        relation_filename = entity_path.name.replace("entities_", "relations_")
        relation_path = processed_dir / relation_filename

        if relation_path.exists():
            logger.debug("Skipping already processed: %s", entity_path.name)
            continue

        try:
            with open(entity_path, "r", encoding="utf-8") as fh:
                entities = json.load(fh)

            all_relations: List[Dict[str, Any]] = []
            batch_size = relation_cfg.batch_size

            for i in range(0, max(len(entities), 1), batch_size):
                batch = entities[i : i + batch_size]
                prompt = _build_relation_prompt(
                    text="",
                    entities=batch,
                    system_prompt=relation_cfg.system_prompt,
                )
                relations = _call_claude_api(prompt, relation_cfg, rate_limiter)
                all_relations.extend(relations)
                total_api_calls += 1

            total_relations += len(all_relations)

            tmp_path = relation_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(all_relations, fh, indent=2, ensure_ascii=False)
            tmp_path.replace(relation_path)

        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to process %s: %s", entity_path.name, exc)
            errors += 1

        if idx % 10 == 0 or idx == len(entity_files):
            logger.info(
                "Progress: %d/%d files, %d relations, %d API calls",
                idx, len(entity_files), total_relations, total_api_calls,
            )

    total_time = time.time() - start_time
    logger.info("=== Relation Extraction Summary ===")
    logger.info("Files: %d, Relations: %d, API calls: %d, Errors: %d",
                len(entity_files), total_relations, total_api_calls, errors)
    logger.info("Time: %.2f seconds", total_time)


def main() -> None:
    """Parse arguments, load config, and run relation extraction."""
    parser = argparse.ArgumentParser(
        description="Extract biomedical relations using Claude API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML configuration file.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit entity files to process.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    logger.info("Starting 03_extract_relations with config: %s", args.config)

    try:
        extract_relations(cfg)
    except KeyboardInterrupt:
        logger.warning("Relation extraction interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Relation extraction failed.")
        sys.exit(1)

    logger.info("03_extract_relations completed successfully.")


if __name__ == "__main__":
    main()