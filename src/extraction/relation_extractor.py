"""
Relation Extractor — Claude API-based relation extraction for FRLM.

Extracts biomedical relations from text using Claude API with carefully
crafted prompts. Supports:
    - Rate limiting and retry logic
    - Checkpoint/resume for batch processing
    - Response validation and error handling
    - Cost estimation and tracking

The prompt is designed to extract structured relations that can be
directly converted to Facts for the knowledge graph.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from config.config import get_secret

logger = logging.getLogger(__name__)


# ===========================================================================
# Relation ontology (must match schema.py)
# ===========================================================================

VALID_RELATION_TYPES = {
    "INHIBITS",
    "ACTIVATES",
    "TREATS",
    "CAUSES",
    "BINDS_TO",
    "METABOLIZED_BY",
    "ASSOCIATED_WITH",
    "UPREGULATES",
    "DOWNREGULATES",
    "CONTRAINDICATED_WITH",
    "SYNERGISTIC_WITH",
    "SUBSTRATE_OF",
    "PRODUCT_OF",
    "BIOMARKER_FOR",
    "EXPRESSED_IN",
    "LOCATED_IN",
    "PART_OF",
    "PRECURSOR_OF",
    "ANALOG_OF",
    "TARGET_OF",
    "INDUCES",
    "PREVENTS",
    "DIAGNOSES",
    "PROGNOSTIC_FOR",
    "RESISTANT_TO",
    "SENSITIVE_TO",
    "INTERACTS_WITH",
    "TRANSPORTS",
    "CATALYZES",
    "ENCODES",
}


# ===========================================================================
# Claude API prompt engineering
# ===========================================================================

RELATION_EXTRACTION_SYSTEM_PROMPT = """You are a biomedical relation extraction system specialized in oncology pharmacology. Your task is to extract structured relations from scientific text.

## Task
Given a text passage and a list of identified biomedical entities, extract all relations between entities that are explicitly stated or strongly implied in the text.

## Relation Types (use ONLY these)
- INHIBITS: X inhibits/blocks/suppresses Y
- ACTIVATES: X activates/stimulates/enhances Y
- TREATS: X treats/is used to treat Y (drug-disease)
- CAUSES: X causes/leads to/results in Y
- BINDS_TO: X binds to/interacts with Y (molecular binding)
- METABOLIZED_BY: X is metabolized by Y (enzyme-drug)
- ASSOCIATED_WITH: X is associated with Y (correlation, not causation)
- UPREGULATES: X upregulates/increases expression of Y
- DOWNREGULATES: X downregulates/decreases expression of Y
- CONTRAINDICATED_WITH: X is contraindicated with Y
- SYNERGISTIC_WITH: X has synergistic effect with Y
- SUBSTRATE_OF: X is a substrate of Y
- PRODUCT_OF: X is a product of Y
- BIOMARKER_FOR: X is a biomarker for Y
- EXPRESSED_IN: X is expressed in Y (gene/protein in tissue/cell)
- LOCATED_IN: X is located in Y
- PART_OF: X is part of Y
- PRECURSOR_OF: X is a precursor of Y
- ANALOG_OF: X is an analog/derivative of Y
- TARGET_OF: X is a target of Y (drug target)
- INDUCES: X induces Y
- PREVENTS: X prevents Y
- DIAGNOSES: X is used to diagnose Y
- PROGNOSTIC_FOR: X is prognostic for Y
- RESISTANT_TO: X is resistant to Y
- SENSITIVE_TO: X is sensitive to Y
- INTERACTS_WITH: X interacts with Y (general interaction)
- TRANSPORTS: X transports Y
- CATALYZES: X catalyzes Y
- ENCODES: X encodes Y (gene-protein)

## Output Format
Return a JSON array of relation objects. Each relation must have:
- subject: The subject entity text (exactly as provided in the entity list)
- subject_id: The canonical_id of the subject entity (from the entity list)
- relation_type: One of the valid relation types above (uppercase)
- object: The object entity text (exactly as provided in the entity list)
- object_id: The canonical_id of the object entity (from the entity list)
- confidence: Float 0.0-1.0 indicating how explicit the relation is (1.0 = directly stated, 0.5 = implied, 0.3 = weak evidence)
- evidence_span: The exact text span that supports this relation
- is_negated: Boolean indicating if this is a negative finding (e.g., "X did NOT inhibit Y")
- temporal_context: Any temporal context mentioned (e.g., "after treatment", "in 2020 guidelines") or null

## Confidence Guidelines
- 1.0: Explicitly stated ("Drug X inhibits enzyme Y")
- 0.8-0.9: Clearly implied from experimental results ("Treatment with X resulted in decreased Y activity")
- 0.5-0.7: Inferred from indirect evidence or review statements
- 0.3-0.4: Weak association or speculative

## Examples

Example 1 - Direct statement:
Text: "Gefitinib inhibits EGFR tyrosine kinase with an IC50 of 33 nM."
Entities: [{"text": "Gefitinib", "canonical_id": "C1122962"}, {"text": "EGFR tyrosine kinase", "canonical_id": "C0034802"}]
Output:
[{
  "subject": "Gefitinib",
  "subject_id": "C1122962",
  "relation_type": "INHIBITS",
  "object": "EGFR tyrosine kinase",
  "object_id": "C0034802",
  "confidence": 1.0,
  "evidence_span": "Gefitinib inhibits EGFR tyrosine kinase with an IC50 of 33 nM",
  "is_negated": false,
  "temporal_context": null
}]

Example 2 - Negative finding:
Text: "Unlike other TKIs, osimertinib did not show significant inhibition of wild-type EGFR."
Entities: [{"text": "osimertinib", "canonical_id": "C3896906"}, {"text": "wild-type EGFR", "canonical_id": "C1512035"}]
Output:
[{
  "subject": "osimertinib",
  "subject_id": "C3896906",
  "relation_type": "INHIBITS",
  "object": "wild-type EGFR",
  "object_id": "C1512035",
  "confidence": 0.9,
  "evidence_span": "osimertinib did not show significant inhibition of wild-type EGFR",
  "is_negated": true,
  "temporal_context": null
}]

Example 3 - Multiple relations with temporal context:
Text: "After 2015 FDA approval, pembrolizumab became first-line treatment for PD-L1 positive NSCLC and has been shown to enhance T-cell activation against tumors."
Entities: [{"text": "pembrolizumab", "canonical_id": "C3657270"}, {"text": "PD-L1 positive NSCLC", "canonical_id": "C4288530"}, {"text": "T-cell", "canonical_id": "C0039194"}]
Output:
[{
  "subject": "pembrolizumab",
  "subject_id": "C3657270",
  "relation_type": "TREATS",
  "object": "PD-L1 positive NSCLC",
  "object_id": "C4288530",
  "confidence": 1.0,
  "evidence_span": "pembrolizumab became first-line treatment for PD-L1 positive NSCLC",
  "is_negated": false,
  "temporal_context": "after 2015 FDA approval"
},
{
  "subject": "pembrolizumab",
  "subject_id": "C3657270",
  "relation_type": "ACTIVATES",
  "object": "T-cell",
  "object_id": "C0039194",
  "confidence": 0.85,
  "evidence_span": "has been shown to enhance T-cell activation against tumors",
  "is_negated": false,
  "temporal_context": null
}]

## Rules
1. ONLY extract relations between entities that are in the provided entity list
2. ONLY use relation types from the valid list above
3. Do NOT invent entities - use the exact text and canonical_id from the entity list
4. If no valid relations exist, return an empty array []
5. Extract ALL valid relations, not just the most obvious ones
6. Be precise with evidence_span - copy the relevant text exactly"""


RELATION_EXTRACTION_USER_TEMPLATE = """Extract all biomedical relations from the following text.

## Text
{text}

## Entities
{entities_json}

## Instructions
Return a JSON array of relations. If no valid relations exist, return [].
Output ONLY valid JSON, no additional text or explanation."""


# ===========================================================================
# Data structures
# ===========================================================================


@dataclass
class ExtractedRelation:
    """A relation extracted from text.

    Attributes
    ----------
    subject : str
        Subject entity text.
    subject_id : str
        Subject canonical ID.
    relation_type : str
        Type of relation (from ontology).
    object : str
        Object entity text.
    object_id : str
        Object canonical ID.
    confidence : float
        Extraction confidence score.
    evidence_span : str
        Text evidence for the relation.
    is_negated : bool
        Whether this is a negative finding.
    temporal_context : str
        Any temporal context mentioned.
    source_text : str
        Full source text.
    """

    subject: str
    subject_id: str
    relation_type: str
    object: str
    object_id: str
    confidence: float = 0.8
    evidence_span: str = ""
    is_negated: bool = False
    temporal_context: Optional[str] = None
    source_text: str = ""

    def to_fact(
        self,
        source: str = "",
        publication_date: Optional[date] = None,
    ):
        """Convert to a Fact schema object.

        Parameters
        ----------
        source : str
            Provenance string (e.g., PMID).
        publication_date : date, optional
            Publication date for temporal envelope.

        Returns
        -------
        Fact
            Knowledge graph fact object.
        """
        from src.kg.schema import (
            BiomedicalEntity,
            Fact,
            Relation,
            RelationType,
            TemporalEnvelope,
        )

        # Create subject entity
        subject_entity = BiomedicalEntity(
            id=self.subject_id,
            label=self.subject,
            entity_type="biomedical_entity",
            canonical_id=self.subject_id,
            source_ontology="UMLS" if not self.subject_id.startswith("HASH:") else "CONTENT_HASH",
        )

        # Create object entity
        object_entity = BiomedicalEntity(
            id=self.object_id,
            label=self.object,
            entity_type="biomedical_entity",
            canonical_id=self.object_id,
            source_ontology="UMLS" if not self.object_id.startswith("HASH:") else "CONTENT_HASH",
        )

        # Create relation
        relation = Relation(type=RelationType(self.relation_type))

        # Create temporal envelope
        valid_from = publication_date or date.today()
        temporal = TemporalEnvelope(valid_from=valid_from, valid_to=None)

        # Create fact with metadata
        metadata = {
            "evidence_span": self.evidence_span,
            "is_negated": self.is_negated,
            "extraction_confidence": self.confidence,
        }
        if self.temporal_context:
            metadata["temporal_context"] = self.temporal_context

        fact = Fact(
            subject=subject_entity,
            relation=relation,
            object=object_entity,
            temporal=temporal,
            source=source,
            confidence=self.confidence,
            metadata=metadata,
        )

        return fact

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject": self.subject,
            "subject_id": self.subject_id,
            "relation_type": self.relation_type,
            "object": self.object,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "evidence_span": self.evidence_span,
            "is_negated": self.is_negated,
            "temporal_context": self.temporal_context,
        }


@dataclass
class ExtractionResult:
    """Result of relation extraction for one text passage."""

    text_id: str
    relations: List[ExtractedRelation] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    duration_seconds: float = 0.0


# ===========================================================================
# Rate limiter
# ===========================================================================


class RateLimiter:
    """Token bucket rate limiter for API requests.

    Parameters
    ----------
    requests_per_minute : int
        Maximum requests per minute.
    tokens_per_minute : int
        Maximum tokens per minute.
    """

    def __init__(
        self,
        requests_per_minute: int = 50,
        tokens_per_minute: int = 100000,
    ):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute

        self._request_times: List[float] = []
        self._token_counts: List[Tuple[float, int]] = []

    def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if necessary to respect rate limits.

        Parameters
        ----------
        estimated_tokens : int
            Estimated tokens for the next request.
        """
        now = time.time()
        window = 60.0

        # Clean old entries
        self._request_times = [t for t in self._request_times if now - t < window]
        self._token_counts = [(t, c) for t, c in self._token_counts if now - t < window]

        # Check RPM
        if len(self._request_times) >= self.rpm:
            sleep_time = self._request_times[0] + window - now
            if sleep_time > 0:
                logger.debug("Rate limit (RPM): sleeping %.2f seconds", sleep_time)
                time.sleep(sleep_time)

        # Check TPM
        total_tokens = sum(c for _, c in self._token_counts)
        if total_tokens + estimated_tokens >= self.tpm:
            oldest_time = min(t for t, _ in self._token_counts) if self._token_counts else now
            sleep_time = oldest_time + window - now
            if sleep_time > 0:
                logger.debug("Rate limit (TPM): sleeping %.2f seconds", sleep_time)
                time.sleep(sleep_time)

    def record_request(self, tokens: int) -> None:
        """Record a completed request.

        Parameters
        ----------
        tokens : int
            Number of tokens used.
        """
        now = time.time()
        self._request_times.append(now)
        self._token_counts.append((now, tokens))


# ===========================================================================
# Checkpoint manager
# ===========================================================================


class CheckpointManager:
    """Manage checkpoints for resumable batch processing.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory to store checkpoint files.
    checkpoint_name : str
        Name prefix for checkpoint files.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str = "relation_extraction",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_name = checkpoint_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}_checkpoint.json"
        self._results_file = self.checkpoint_dir / f"{checkpoint_name}_results.jsonl"

    def get_completed_ids(self) -> Set[str]:
        """Get IDs of already processed items."""
        if not self._checkpoint_file.exists():
            return set()

        try:
            with open(self._checkpoint_file, "r") as f:
                data = json.load(f)
            return set(data.get("completed_ids", []))
        except Exception as e:
            logger.warning("Could not load checkpoint: %s", e)
            return set()

    def save_result(self, result: ExtractionResult) -> None:
        """Save a single extraction result and update checkpoint.

        Parameters
        ----------
        result : ExtractionResult
            Extraction result to save.
        """
        # Append result to JSONL file
        result_dict = {
            "text_id": result.text_id,
            "relations": [r.to_dict() for r in result.relations],
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "error": result.error,
            "duration_seconds": result.duration_seconds,
        }

        with open(self._results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_dict) + "\n")

        # Update checkpoint
        completed_ids = self.get_completed_ids()
        completed_ids.add(result.text_id)

        checkpoint_data = {
            "completed_ids": list(completed_ids),
            "total_processed": len(completed_ids),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Atomic write
        tmp_file = self._checkpoint_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f)
        tmp_file.replace(self._checkpoint_file)

    def load_all_results(self) -> List[ExtractionResult]:
        """Load all saved results."""
        if not self._results_file.exists():
            return []

        results = []
        with open(self._results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    relations = [
                        ExtractedRelation(**r) for r in data.get("relations", [])
                    ]
                    result = ExtractionResult(
                        text_id=data["text_id"],
                        relations=relations,
                        input_tokens=data.get("input_tokens", 0),
                        output_tokens=data.get("output_tokens", 0),
                        cost_usd=data.get("cost_usd", 0.0),
                        error=data.get("error"),
                        duration_seconds=data.get("duration_seconds", 0.0),
                    )
                    results.append(result)

        return results


# ===========================================================================
# Relation Extractor
# ===========================================================================


class RelationExtractor:
    """Extract biomedical relations using Claude API.

    Parameters
    ----------
    model : str
        Claude model name.
    api_key : str
        Anthropic API key.
    max_tokens : int
        Maximum tokens for response.
    temperature : float
        Sampling temperature.
    max_retries : int
        Maximum retry attempts.
    retry_delay : float
        Initial retry delay (seconds).
    rate_limit_rpm : int
        Rate limit: requests per minute.
    rate_limit_tpm : int
        Rate limit: tokens per minute.
    checkpoint_dir : Path, optional
        Directory for checkpoints.
    """

    # Claude Sonnet pricing (per million tokens)
    INPUT_COST_PER_MILLION = 3.0
    OUTPUT_COST_PER_MILLION = 15.0

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        rate_limit_rpm: int = 50,
        rate_limit_tpm: int = 100000,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model
        self.api_key = api_key or get_secret("anthropic.api_key")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set anthropic.api_key in "
                "config/secrets.properties or pass api_key."
            )

        # Initialize client lazily
        self._client = None

        # Rate limiting
        self._rate_limiter = RateLimiter(rate_limit_rpm, rate_limit_tpm)

        # Checkpointing
        self._checkpoint_manager = None
        if checkpoint_dir:
            self._checkpoint_manager = CheckpointManager(checkpoint_dir)

        # Statistics
        self._stats: Dict[str, Any] = {
            "texts_processed": 0,
            "relations_extracted": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "errors": 0,
            "validation_failures": 0,
        }

        logger.info(
            "RelationExtractor initialized: model=%s, rpm=%d, tpm=%d",
            model,
            rate_limit_rpm,
            rate_limit_tpm,
        )

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        return self._client

    @property
    def stats(self) -> Dict[str, Any]:
        """Return extraction statistics."""
        return self._stats.copy()

    def _estimate_tokens(self, text: str, entities: List[Dict]) -> int:
        """Estimate token count for a request."""
        # Rough estimate: ~4 chars per token
        entities_json = json.dumps(entities)
        total_chars = len(RELATION_EXTRACTION_SYSTEM_PROMPT) + len(text) + len(entities_json)
        return total_chars // 4 + 500  # Add buffer for formatting

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost in USD."""
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_MILLION
        return input_cost + output_cost

    def _parse_response(
        self,
        response_text: str,
        valid_entity_ids: Set[str],
    ) -> Tuple[List[ExtractedRelation], List[str]]:
        """Parse Claude response JSON into ExtractedRelation objects.

        Parameters
        ----------
        response_text : str
            Raw response text from Claude.
        valid_entity_ids : Set[str]
            Set of valid entity canonical IDs.

        Returns
        -------
        Tuple[List[ExtractedRelation], List[str]]
            (valid_relations, validation_errors)
        """
        relations = []
        errors = []

        # Clean response (handle markdown code blocks)
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Handle empty response
        if not cleaned or cleaned == "[]":
            return [], []

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("JSON parse error: %s", e)
            errors.append(f"JSON parse error: {e}")
            return [], errors

        if not isinstance(data, list):
            errors.append(f"Expected list, got {type(data)}")
            return [], errors

        for idx, item in enumerate(data):
            try:
                # Validate required fields
                required = ["subject", "subject_id", "relation_type", "object", "object_id"]
                missing = [f for f in required if f not in item]
                if missing:
                    errors.append(f"Relation {idx}: missing fields {missing}")
                    continue

                # Validate relation type
                rel_type = item["relation_type"].upper()
                if rel_type not in VALID_RELATION_TYPES:
                    errors.append(f"Relation {idx}: invalid type '{rel_type}'")
                    continue

                # Validate entity IDs
                subj_id = item["subject_id"]
                obj_id = item["object_id"]
                if subj_id not in valid_entity_ids:
                    errors.append(f"Relation {idx}: unknown subject_id '{subj_id}'")
                    continue
                if obj_id not in valid_entity_ids:
                    errors.append(f"Relation {idx}: unknown object_id '{obj_id}'")
                    continue

                # Create relation
                relation = ExtractedRelation(
                    subject=item["subject"],
                    subject_id=subj_id,
                    relation_type=rel_type,
                    object=item["object"],
                    object_id=obj_id,
                    confidence=float(item.get("confidence", 0.8)),
                    evidence_span=item.get("evidence_span", ""),
                    is_negated=bool(item.get("is_negated", False)),
                    temporal_context=item.get("temporal_context"),
                )
                relations.append(relation)

            except Exception as e:
                errors.append(f"Relation {idx}: {e}")

        return relations, errors

    def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        text_id: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract relations from text using Claude API.

        Parameters
        ----------
        text : str
            Source text passage.
        entities : List[Dict]
            List of entity dicts with 'text' and 'canonical_id' keys.
        text_id : str, optional
            Identifier for this text (for checkpointing).

        Returns
        -------
        ExtractionResult
            Extraction result with relations and metadata.
        """
        text_id = text_id or hashlib.md5(text.encode()).hexdigest()[:12]
        start_time = time.time()

        # Check checkpoint
        if self._checkpoint_manager:
            completed = self._checkpoint_manager.get_completed_ids()
            if text_id in completed:
                logger.debug("Skipping already processed text: %s", text_id)
                return ExtractionResult(text_id=text_id, relations=[])

        # Build entity ID set for validation
        valid_entity_ids = {e["canonical_id"] for e in entities if "canonical_id" in e}

        # Format entities for prompt
        entities_json = json.dumps(
            [{"text": e.get("text", e.get("label", "")), "canonical_id": e["canonical_id"]}
             for e in entities],
            indent=2
        )

        # Build user message
        user_message = RELATION_EXTRACTION_USER_TEMPLATE.format(
            text=text,
            entities_json=entities_json,
        )

        # Estimate tokens and wait for rate limit
        estimated_tokens = self._estimate_tokens(text, entities)
        self._rate_limiter.wait_if_needed(estimated_tokens)

        # Call Claude API with retries
        relations = []
        error_msg = None
        input_tokens = 0
        output_tokens = 0

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=RELATION_EXTRACTION_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )

                # Extract usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                self._rate_limiter.record_request(input_tokens + output_tokens)

                # Parse response
                response_text = response.content[0].text if response.content else ""
                relations, validation_errors = self._parse_response(
                    response_text, valid_entity_ids
                )

                if validation_errors:
                    logger.warning(
                        "Validation errors for %s: %s",
                        text_id,
                        validation_errors[:3],
                    )
                    self._stats["validation_failures"] += len(validation_errors)

                break  # Success

            except Exception as e:
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    "API error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    e,
                    delay,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                else:
                    error_msg = str(e)
                    self._stats["errors"] += 1

        # Calculate cost
        cost_usd = self._calculate_cost(input_tokens, output_tokens)

        # Update stats
        self._stats["texts_processed"] += 1
        self._stats["relations_extracted"] += len(relations)
        self._stats["total_input_tokens"] += input_tokens
        self._stats["total_output_tokens"] += output_tokens
        self._stats["total_cost_usd"] += cost_usd

        duration = time.time() - start_time

        result = ExtractionResult(
            text_id=text_id,
            relations=relations,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            error=error_msg,
            duration_seconds=duration,
        )

        # Save checkpoint
        if self._checkpoint_manager:
            self._checkpoint_manager.save_result(result)

        return result

    def extract_relations_batch(
        self,
        items: List[Tuple[str, List[Dict], str]],
        show_progress: bool = True,
    ) -> List[ExtractionResult]:
        """Extract relations from multiple texts.

        Parameters
        ----------
        items : List[Tuple[str, List[Dict], str]]
            List of (text, entities, text_id) tuples.
        show_progress : bool
            Whether to log progress.

        Returns
        -------
        List[ExtractionResult]
            List of extraction results.
        """
        results = []

        for idx, (text, entities, text_id) in enumerate(items):
            result = self.extract_relations(text, entities, text_id)
            results.append(result)

            if show_progress and (idx + 1) % 10 == 0:
                logger.info(
                    "Progress: %d/%d texts, %d relations, $%.4f cost",
                    idx + 1,
                    len(items),
                    self._stats["relations_extracted"],
                    self._stats["total_cost_usd"],
                )

        return results

    def estimate_cost(
        self,
        texts: List[str],
        avg_entities_per_text: int = 10,
    ) -> Dict[str, float]:
        """Estimate API cost for processing texts.

        Parameters
        ----------
        texts : List[str]
            List of texts to process.
        avg_entities_per_text : int
            Average number of entities per text.

        Returns
        -------
        Dict[str, float]
            Cost estimates.
        """
        # Estimate tokens
        total_chars = sum(len(t) for t in texts)
        system_prompt_chars = len(RELATION_EXTRACTION_SYSTEM_PROMPT)
        entity_chars_per_text = avg_entities_per_text * 100  # ~100 chars per entity

        total_input_chars = (
            total_chars +
            len(texts) * (system_prompt_chars + entity_chars_per_text)
        )

        # ~4 chars per token
        estimated_input_tokens = total_input_chars // 4
        estimated_output_tokens = len(texts) * 500  # ~500 tokens per response

        input_cost = (estimated_input_tokens / 1_000_000) * self.INPUT_COST_PER_MILLION
        output_cost = (estimated_output_tokens / 1_000_000) * self.OUTPUT_COST_PER_MILLION

        return {
            "num_texts": len(texts),
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_input_cost_usd": input_cost,
            "estimated_output_cost_usd": output_cost,
            "estimated_total_cost_usd": input_cost + output_cost,
        }


# ===========================================================================
# Factory function
# ===========================================================================


def create_extractor_from_config(config: Any) -> RelationExtractor:
    """Create a RelationExtractor from FRLM config.

    Parameters
    ----------
    config : FRLMConfig
        FRLM configuration object.

    Returns
    -------
    RelationExtractor
        Configured relation extractor.
    """
    rel_cfg = config.extraction.relation
    checkpoint_dir = config.paths.resolve("processed_dir") / "relation_checkpoints"

    return RelationExtractor(
        model=rel_cfg.model,
        api_key=rel_cfg.api_key if rel_cfg.api_key != "CHANGE_ME" else None,
        max_tokens=rel_cfg.max_tokens,
        temperature=rel_cfg.temperature,
        max_retries=rel_cfg.max_retries,
        retry_delay=rel_cfg.retry_delay,
        rate_limit_rpm=rel_cfg.rate_limit_rpm,
        rate_limit_tpm=rel_cfg.rate_limit_tpm,
        checkpoint_dir=checkpoint_dir,
    )

