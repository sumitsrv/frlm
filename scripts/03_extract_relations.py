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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FRLMConfig, load_config, setup_logging
from src.status import PipelineStatusTracker

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
        f"Return ONLY a raw JSON array (no markdown fences, no explanation) where each element has: "
        f"\"subject\", \"relation_type\", \"object\", \"confidence\" (0-1), \"evidence_span\".\n"
        f"Valid relation types: TREATS, CAUSES, INHIBITS, ACTIVATES, "
        f"ASSOCIATED_WITH, METABOLIZED_BY, CONTRAINDICATED_WITH, "
        f"DOSAGE_OF, BIOMARKER_FOR, RESISTANCE_TO.\n"
        f"If no relations are found, return an empty array: []"
    )

    return {"system": system_prompt, "user": user_msg}


def _extract_json_from_response(text: str) -> List[Dict[str, Any]]:
    """Extract a JSON array from a Claude response that may contain markdown fences or prose.

    Tries, in order:
    1. Direct json.loads on the full text.
    2. Extract content between ```json ... ``` or ``` ... ``` fences.
    3. Find the first '[' and last ']' and parse that substring.
    """
    import re

    # 1. Try direct parse
    text = text.strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    # 2. Try extracting from markdown code fences
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError:
            pass

    # 3. Find the outermost JSON array brackets
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        try:
            result = json.loads(text[first_bracket : last_bracket + 1])
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError:
            pass

    # Nothing worked
    raise json.JSONDecodeError("No valid JSON array found in response", text, 0)


def _response_cache_path(cache_dir: Path, prompt: Dict[str, str]) -> Path:
    """Compute a deterministic cache path for a Claude API prompt."""
    import hashlib
    key = hashlib.sha256(prompt["user"].encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"claude_response_{key}.json"


def _save_raw_response(cache_dir: Path, prompt: Dict[str, str], response_text: str) -> Path:
    """Save a raw Claude API response to the cache directory.

    Every API call is persisted so no paid response is ever lost.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _response_cache_path(cache_dir, prompt)
    payload = {
        "prompt_user": prompt["user"][:500],  # truncated for readability
        "response_text": response_text,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.debug("Saved raw API response to %s", cache_path)
    return cache_path


def _load_cached_response(cache_dir: Path, prompt: Dict[str, str]) -> Optional[str]:
    """Load a previously cached raw response, if it exists."""
    cache_path = _response_cache_path(cache_dir, prompt)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            logger.debug("Using cached API response from %s", cache_path)
            return data.get("response_text")
        except Exception:
            pass
    return None


def _call_claude_api(
    prompt: Dict[str, str],
    relation_cfg: Any,
    rate_limiter: RateLimiter,
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Call the Claude API for relation extraction with retry logic.

    Every raw response is saved to ``cache_dir`` so no paid API call is lost.
    On re-run, cached responses are reused without hitting the API.
    """
    # Check cache first
    if cache_dir:
        cached_text = _load_cached_response(cache_dir, prompt)
        if cached_text is not None:
            try:
                return _extract_json_from_response(cached_text)
            except json.JSONDecodeError:
                logger.warning("Cached response exists but still fails to parse — re-calling API")

    rate_limiter.wait_if_needed(estimated_tokens=relation_cfg.max_tokens)
    response_text = ""

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
            response_text = response.content[0].text

            # Always save the raw response before attempting to parse
            if cache_dir:
                _save_raw_response(cache_dir, prompt, response_text)

            return _extract_json_from_response(response_text)

        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse API response as JSON (attempt %d/%d). Response text: %.200s",
                attempt,
                relation_cfg.max_retries,
                response_text,
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


# ---------------------------------------------------------------------------
# Helpers — XML parsing, chunking, atomic writes
# ---------------------------------------------------------------------------


def _parse_pmc_xml_sections(
    xml_path: Path, sections: List[str],
) -> List[Dict[str, str]]:
    """Parse a PMC XML and extract text from configured sections.

    Returns a list of dicts with keys: section, text, pmcid.
    (Same logic as 02_extract_entities._parse_pmc_xml.)
    """
    pmcid = xml_path.stem
    results: List[Dict[str, str]] = []

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        if "abstract" in sections:
            for abstract in root.iter("abstract"):
                text = " ".join(abstract.itertext()).strip()
                if text:
                    results.append({"section": "abstract", "text": text, "pmcid": pmcid})

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
                        results.append({
                            "section": matched_section,
                            "text": " ".join(paragraphs),
                            "pmcid": pmcid,
                        })

    except Exception as exc:
        logger.warning("Failed to parse XML %s: %s", xml_path.name, exc)

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


def _load_text_chunks(
    xml_path: Path, corpus_cfg: Any,
) -> Dict[Tuple[str, int], str]:
    """Load and chunk text from a corpus XML, keyed by (section, chunk_idx).

    This recreates the same chunking that step 02 used, so chunk indices
    align with the entity JSON.
    """
    sections_list = getattr(corpus_cfg, "sections", ["abstract"])
    chunk_size = getattr(corpus_cfg, "chunk_size", 512)
    chunk_overlap = getattr(corpus_cfg, "chunk_overlap", 64)

    result: Dict[Tuple[str, int], str] = {}
    parsed = _parse_pmc_xml_sections(xml_path, sections_list)

    for sec in parsed:
        chunks = _chunk_text(sec["text"], chunk_size, chunk_overlap)
        for ci, chunk_text in enumerate(chunks):
            result[(sec["section"], ci)] = chunk_text

    return result


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (write to .tmp then rename)."""
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def extract_relations(cfg: FRLMConfig) -> None:
    """Orchestrate relation extraction across all entity files.

    1. Discover entity files.
    2. Initialize rate limiter.
    3. For each file, load entities AND the original text from the corpus XML,
       group entities by (section, chunk_idx), then call Claude with
       each text chunk + its entities.
    4. Write relation JSON per document.
    """
    relation_cfg = cfg.extraction.relation
    corpus_cfg = cfg.extraction.corpus
    processed_dir = cfg.paths.resolve("processed_dir")
    corpus_dir = cfg.paths.resolve("corpus_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Cache directory for raw Claude API responses — never lose a paid response
    api_cache_dir = cfg.paths.resolve("cache_dir") / "claude_relation_responses"
    api_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Raw API responses cached in: %s", api_cache_dir)

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

    tracker = PipelineStatusTracker()
    tracker.mark_running(3, total_items=len(entity_files))

    rate_limiter = RateLimiter(
        requests_per_minute=relation_cfg.rate_limit_rpm,
        tokens_per_minute=relation_cfg.rate_limit_tpm,
    )

    start_time = time.time()
    total_relations = 0
    total_api_calls = 0
    errors = 0
    completed_files = 0
    skipped_files = 0

    for idx, entity_path in enumerate(entity_files, start=1):
        relation_filename = entity_path.name.replace("entities_", "relations_")
        relation_path = processed_dir / relation_filename

        if relation_path.exists():
            logger.debug("Skipping already processed: %s", entity_path.name)
            skipped_files += 1
            completed_files += 1
            continue

        try:
            with open(entity_path, "r", encoding="utf-8") as fh:
                entities = json.load(fh)

            if not entities:
                # Write empty relations file and skip
                _atomic_write_json(relation_path, [])
                continue

            # --- Reconstruct text chunks from the corpus XML ---
            pmcid = entities[0].get("pmcid", entity_path.stem.replace("entities_", ""))
            xml_path = corpus_dir / f"{pmcid}.xml"

            chunk_texts = _load_text_chunks(xml_path, corpus_cfg) if xml_path.exists() else {}

            # Group entities by (section, chunk_idx)
            from collections import defaultdict
            entity_groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
            for ent in entities:
                key = (ent.get("section", ""), ent.get("chunk_idx", 0))
                entity_groups[key].append(ent)

            all_relations: List[Dict[str, Any]] = []

            for (section, chunk_idx), group_entities in entity_groups.items():
                text = chunk_texts.get((section, chunk_idx), "")
                if not text and not group_entities:
                    continue

                # Build and send prompt with real text + entities
                prompt = _build_relation_prompt(
                    text=text,
                    entities=group_entities,
                    system_prompt=relation_cfg.system_prompt,
                )
                relations = _call_claude_api(prompt, relation_cfg, rate_limiter, cache_dir=api_cache_dir)

                # Tag each relation with source info
                for rel in relations:
                    if isinstance(rel, dict):
                        rel["pmcid"] = pmcid
                        rel["section"] = section
                        rel["valid_from"] = ""
                        rel["valid_to"] = None

                all_relations.extend(r for r in relations if isinstance(r, dict))
                total_api_calls += 1

            total_relations += len(all_relations)
            _atomic_write_json(relation_path, all_relations)
            completed_files += 1

        except Exception as exc:
            logger.error("Failed to process %s: %s", entity_path.name, exc, exc_info=True)
            errors += 1

        if idx % 10 == 0 or idx == len(entity_files):
            logger.info(
                "Progress: %d/%d files, %d relations, %d API calls, %d errors",
                idx, len(entity_files), total_relations, total_api_calls, errors,
            )
            tracker.update_progress(
                3,
                completed_items=completed_files,
                skipped_items=skipped_files,
                failed_items=errors,
            )
            tracker.save()

    total_time = time.time() - start_time
    tracker.update_progress(
        3,
        completed_items=completed_files,
        skipped_items=skipped_files,
        failed_items=errors,
    )
    if errors == 0:
        tracker.mark_completed(3)
    else:
        tracker.mark_partial(3)

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