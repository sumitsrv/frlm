"""
LLM-based span labeler using Claude API.

Classifies biomedical text spans as factual-retrieval vs. linguistic-generation
for training the FRLM router head.  Each span is annotated with a label
(``"factual"`` or ``"linguistic"``) and a confidence score.

Responsibilities
----------------
* Build the few-shot Claude prompt with clear definitions and examples.
* Call the Anthropic API with rate-limiting, retries, and cost tracking.
* Parse the JSON response into ``SpanLabel`` objects and validate offsets.
* Align character-level labels to BioMedLM token positions.
* Run batch labeling over a corpus with checkpoint / resume support.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from config.config import get_secret

if TYPE_CHECKING:
    from config.config import LabelingConfig  # pragma: no cover

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SpanLabel(BaseModel):
    """A single labelled span within a text.

    Attributes
    ----------
    start_char : int
        Inclusive character offset where the span begins.
    end_char : int
        Exclusive character offset where the span ends.
    text : str
        The surface text of the span.
    label : str
        Either ``"factual"`` or ``"linguistic"``.
    confidence : float
        Model confidence in ``[0, 1]``.
    """

    start_char: int = Field(..., ge=0, description="Inclusive start offset")
    end_char: int = Field(..., ge=0, description="Exclusive end offset")
    text: str = Field(..., min_length=0)
    label: Literal["factual", "linguistic"] = Field(
        ..., description="Span classification"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("end_char")
    @classmethod
    def end_after_start(cls, v: int, info: Any) -> int:
        start = info.data.get("start_char", 0)
        if v < start:
            raise ValueError(f"end_char ({v}) must be >= start_char ({start})")
        return v


class CostTracker:
    """Accumulates API usage for cost estimation.

    Pricing is model-aware (per 1 M tokens, as of 2025-Q1):

    ============================  =========  ==========
    Model                         Input $/M  Output $/M
    ============================  =========  ==========
    claude-sonnet-4-6      3.00       15.00
    claude-haiku-4-5-*            1.00        5.00
    claude-3-haiku-*              0.25        1.25
    ============================  =========  ==========
    """

    _PRICING: Dict[str, Tuple[float, float]] = {
        # (input $/M tokens, output $/M tokens)
        "sonnet": (3.00, 15.00),
        "haiku-4-5": (1.00, 5.00),
        "haiku-3": (0.25, 1.25),
    }

    def __init__(self, model: str = "") -> None:
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_requests: int = 0
        inp, out = self._resolve_pricing(model)
        self.INPUT_COST_PER_TOKEN = inp / 1_000_000
        self.OUTPUT_COST_PER_TOKEN = out / 1_000_000

    @classmethod
    def _resolve_pricing(cls, model: str) -> Tuple[float, float]:
        m = model.lower()
        if "haiku" in m and ("4-5" in m or "4.5" in m):
            return cls._PRICING["haiku-4-5"]
        if "haiku" in m:
            return cls._PRICING["haiku-3"]
        # Default to Sonnet pricing (safe upper bound)
        return cls._PRICING["sonnet"]

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.total_input_tokens * self.INPUT_COST_PER_TOKEN
            + self.total_output_tokens * self.OUTPUT_COST_PER_TOKEN
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert biomedical text annotator for the Factual Retrieval Language \
Model (FRLM) project.  Your task is to classify every span in the provided \
text as either **factual** or **linguistic**.

### Definitions

**factual** — Any claim that references a specific entity, measurement, \
relation, experimental result, or established finding that could be stored \
as a structured fact in a biomedical knowledge base.  This includes:
  • Named entities (drugs, genes, proteins, diseases, pathways)
  • Quantitative measurements (IC50 values, dosages, survival rates)
  • Causal / mechanistic claims ("X inhibits Y", "X causes Y")
  • Experimental outcomes even if hedged ("is thought to involve …")
  • Temporal facts ("approved in 2015", "withdrawn after Phase III")

**linguistic** — Discourse connectives, reasoning chains, hedging language, \
meta-commentary, methodological descriptions that do not state a factual \
result, general statements, syntactic boilerplate.  Examples:
  • "These results suggest that …"
  • "In this study, we …"
  • "Further investigation is warranted."
  • "However," "Moreover," "In contrast,"

### Ambiguity rule
When a span is ambiguous — e.g. it hedges but still references a specific \
mechanism or entity — label it **factual**.  The retrieval pipeline can always \
fall back to generation, but missing a factual span means the fact is lost.

### Output format
Return **only** a JSON array.  Each element must be an object with exactly \
three keys:
  • "span": the exact text of the span (verbatim from the input)
  • "label": "factual" or "linguistic"
  • "confidence": a float in [0, 1]

Spans must cover the entire input text with no gaps and no overlaps.  \
Adjacent spans with the same label should be merged.\
"""

FEW_SHOT_EXAMPLES = """\
### Annotated examples

**Example 1 — clear factual claim**
Input: "Gefitinib inhibits EGFR with an IC50 of 33 nM."
Output:
```json
[{"span": "Gefitinib inhibits EGFR with an IC50 of 33 nM.", "label": "factual", "confidence": 0.98}]
```

**Example 2 — clear linguistic span**
Input: "These results suggest that further investigation is warranted."
Output:
```json
[{"span": "These results suggest that further investigation is warranted.", "label": "linguistic", "confidence": 0.95}]
```

**Example 3 — ambiguous (label as factual)**
Input: "The mechanism is thought to involve oxidative stress leading to \
apoptosis of tumor cells."
Output:
```json
[{"span": "The mechanism is thought to involve oxidative stress leading to apoptosis of tumor cells.", "label": "factual", "confidence": 0.82}]
```

**Example 4 — mixed text**
Input: "In this study, we demonstrated that erlotinib (150 mg/day) \
significantly improved progression-free survival in EGFR-mutant NSCLC \
patients compared to chemotherapy."
Output:
```json
[
  {"span": "In this study, we demonstrated that ", "label": "linguistic", "confidence": 0.90},
  {"span": "erlotinib (150 mg/day) significantly improved progression-free survival in EGFR-mutant NSCLC patients compared to chemotherapy.", "label": "factual", "confidence": 0.95}
]
```\
"""


def _build_user_message(text: str) -> str:
    """Build the user message for the Claude API call."""
    return (
        f"{FEW_SHOT_EXAMPLES}\n\n"
        "---\n\n"
        "Now annotate the following text.  Return **only** the JSON array.\n\n"
        f"```\n{text}\n```"
    )


def _build_batch_user_message(texts: List[str]) -> str:
    """Build a user message that asks Claude to label multiple texts at once.

    Each text is given a numeric ID so the response can be matched back.
    Expected response format:  ``{"results": {"0": [...], "1": [...], ...}}``
    """
    lines = [
        f"{FEW_SHOT_EXAMPLES}\n\n---\n\n"
        "Now annotate **each** of the following texts.  "
        "Return a JSON object with a single key `\"results\"` whose value is "
        "an object mapping each text ID (string) to its annotation array.  "
        "Return **only** the JSON object, nothing else.\n"
    ]
    for idx, text in enumerate(texts):
        lines.append(f'\nText {idx}:\n```\n{text}\n```')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLMLabeler
# ---------------------------------------------------------------------------


class LLMLabeler:
    """Label biomedical text spans using the Claude API.

    Parameters
    ----------
    model : str
        Anthropic model identifier (e.g. ``"claude-sonnet-4-20250514"``).
    api_key : str
        Anthropic API key.
    max_tokens : int
        Maximum output tokens per API call.
    temperature : float
        Sampling temperature (0 = deterministic).
    max_retries : int
        Number of retries on transient errors.
    retry_delay : float
        Base delay between retries (multiplied by attempt number).
    rate_limit_rpm : int
        Maximum requests per minute.
    tokenizer_name : str
        HuggingFace tokenizer to use for token alignment.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        rate_limit_rpm: int = 50,
        tokenizer_name: str = "stanford-crfm/BioMedLM",
    ) -> None:
        self.model = model
        self.api_key = api_key or get_secret("anthropic.api_key", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_rpm = rate_limit_rpm
        self._min_interval = 60.0 / max(rate_limit_rpm, 1)
        self._last_request_time: float = 0.0
        self.tokenizer_name = tokenizer_name
        self.cost_tracker = CostTracker(model=self.model)
        self._client: Any = None  # lazily initialised

    # ------------------------------------------------------------------
    # Anthropic client
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazily initialise the Anthropic client."""
        if self._client is None:
            try:
                import anthropic  # type: ignore[import-untyped]

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required. "
                    "Install it with: pip install anthropic"
                )
        return self._client

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Block until the minimum inter-request interval has elapsed."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            sleep_time = self._min_interval - elapsed
            logger.debug("Rate-limiting: sleeping %.2fs", sleep_time)
            time.sleep(sleep_time)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    def _call_api(self, text: str) -> str:
        """Call the Claude API with retries and rate-limiting.

        Returns the raw response text (should be JSON).
        """
        client = self._get_client()
        user_message = _build_user_message(text)

        for attempt in range(1, self.max_retries + 1):
            self._rate_limit()
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                # Track cost
                usage = getattr(response, "usage", None)
                if usage:
                    self.cost_tracker.record(
                        input_tokens=getattr(usage, "input_tokens", 0),
                        output_tokens=getattr(usage, "output_tokens", 0),
                    )
                return response.content[0].text
            except Exception as exc:
                logger.warning(
                    "API call attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"All {self.max_retries} API attempts failed for text "
            f"({len(text)} chars)"
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(raw: str) -> List[Dict[str, Any]]:
        """Extract JSON array from a response that may contain markdown fences."""
        # Try to strip markdown code fences
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
        text = match.group(1) if match else raw.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Last-resort: find first [ ... ] block
            bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
            if bracket_match:
                parsed = json.loads(bracket_match.group(0))
            else:
                raise
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON array")
        return parsed

    @staticmethod
    def _parse_response(
        raw: str, original_text: str
    ) -> List[SpanLabel]:
        """Parse Claude's JSON response into validated ``SpanLabel`` objects.

        Character offsets are computed by locating each span's verbatim text
        in the original text, scanning left-to-right.
        """
        items = LLMLabeler._extract_json(raw)
        labels: List[SpanLabel] = []
        search_start = 0

        for item in items:
            span_text: str = item.get("span", "")
            label: str = item.get("label", "linguistic")
            confidence: float = float(item.get("confidence", 0.5))

            # Normalise label
            label_lower = label.lower().strip()
            if label_lower in ("factual", "retrieval"):
                label_lower = "factual"
            elif label_lower in ("linguistic", "generation"):
                label_lower = "linguistic"
            else:
                logger.warning("Unknown label '%s', defaulting to 'linguistic'", label)
                label_lower = "linguistic"

            # Locate span in original text
            idx = original_text.find(span_text, search_start)
            if idx == -1:
                # Fallback: try case-insensitive
                idx = original_text.lower().find(span_text.lower(), search_start)
            if idx == -1:
                # Still not found — try from beginning
                idx = original_text.find(span_text)
            if idx == -1:
                logger.warning(
                    "Span not found in original text (skipping): '%s'",
                    span_text[:80],
                )
                continue

            end_idx = idx + len(span_text)
            labels.append(
                SpanLabel(
                    start_char=idx,
                    end_char=end_idx,
                    text=span_text,
                    label=label_lower,  # type: ignore[arg-type]
                    confidence=min(max(confidence, 0.0), 1.0),
                )
            )
            search_start = end_idx

        return labels

    # ------------------------------------------------------------------
    # Public: label one text
    # ------------------------------------------------------------------

    def label_text(self, text: str) -> List[SpanLabel]:
        """Label a single text and return span labels.

        Parameters
        ----------
        text : str
            Biomedical text to annotate.

        Returns
        -------
        List[SpanLabel]
            Character-level span labels covering the input text.
        """
        if not text.strip():
            return []

        raw = self._call_api(text)
        return self._parse_response(raw, text)

    # ------------------------------------------------------------------
    # Batch labeling — multiple texts per API call
    # ------------------------------------------------------------------

    def _call_api_batch(self, texts: List[str]) -> str:
        """Call Claude with a batch message containing multiple texts."""
        client = self._get_client()
        user_message = _build_batch_user_message(texts)

        for attempt in range(1, self.max_retries + 1):
            self._rate_limit()
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                usage = getattr(response, "usage", None)
                if usage:
                    self.cost_tracker.record(
                        input_tokens=getattr(usage, "input_tokens", 0),
                        output_tokens=getattr(usage, "output_tokens", 0),
                    )
                return response.content[0].text
            except Exception as exc:
                logger.warning(
                    "Batch API call attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"All {self.max_retries} batch API attempts failed "
            f"({len(texts)} texts)"
        )

    @staticmethod
    def _parse_batch_response(
        raw: str, texts: List[str]
    ) -> Dict[int, List[SpanLabel]]:
        """Parse a batch response mapping text IDs → span labels.

        Expected JSON: ``{"results": {"0": [...], "1": [...], ...}}``
        Falls back to single-array format for single-text batches.
        """
        # Strip markdown fences if present
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
        text_body = match.group(1) if match else raw.strip()

        try:
            parsed = json.loads(text_body)
        except json.JSONDecodeError:
            # Try finding the outermost { ... }
            brace = re.search(r"\{.*\}", text_body, re.DOTALL)
            if brace:
                parsed = json.loads(brace.group(0))
            else:
                raise

        results: Dict[int, List[SpanLabel]] = {}

        # Format: {"results": {"0": [...], "1": [...]}}
        if isinstance(parsed, dict) and "results" in parsed:
            for key, items in parsed["results"].items():
                idx = int(key)
                if idx < len(texts):
                    results[idx] = LLMLabeler._parse_response_items(
                        items, texts[idx]
                    )
        # Fallback for single-text batch returning bare array
        elif isinstance(parsed, list) and len(texts) == 1:
            results[0] = LLMLabeler._parse_response_items(parsed, texts[0])
        else:
            logger.warning("Unexpected batch response structure; returning empty")

        return results

    @staticmethod
    def _parse_response_items(
        items: List[Dict[str, Any]], original_text: str,
    ) -> List[SpanLabel]:
        """Shared helper: convert a list of raw dicts into SpanLabels."""
        labels: List[SpanLabel] = []
        search_start = 0
        for item in items:
            span_text: str = item.get("span", "")
            label: str = item.get("label", "linguistic")
            confidence: float = float(item.get("confidence", 0.5))

            label_lower = label.lower().strip()
            if label_lower in ("factual", "retrieval"):
                label_lower = "factual"
            elif label_lower in ("linguistic", "generation"):
                label_lower = "linguistic"
            else:
                label_lower = "linguistic"

            idx = original_text.find(span_text, search_start)
            if idx == -1:
                idx = original_text.lower().find(span_text.lower(), search_start)
            if idx == -1:
                idx = original_text.find(span_text)
            if idx == -1:
                continue

            end_idx = idx + len(span_text)
            labels.append(SpanLabel(
                start_char=idx,
                end_char=end_idx,
                text=span_text,
                label=label_lower,  # type: ignore[arg-type]
                confidence=min(max(confidence, 0.0), 1.0),
            ))
            search_start = end_idx
        return labels

    def label_texts_batch(
        self, texts: List[str], *, api_batch_size: int = 50,
    ) -> List[List[SpanLabel]]:
        """Label multiple texts using batched API calls.

        Groups *texts* into chunks of *api_batch_size* and sends each
        chunk in a single API call, dramatically reducing per-text
        prompt overhead.

        Parameters
        ----------
        texts : List[str]
            Texts to label.
        api_batch_size : int
            How many texts to pack into one API call.

        Returns
        -------
        List[List[SpanLabel]]
            One list of span labels per input text (same order).
        """
        all_results: List[List[SpanLabel]] = [[] for _ in texts]

        for batch_start in range(0, len(texts), api_batch_size):
            batch = texts[batch_start:batch_start + api_batch_size]
            non_empty = [(i, t) for i, t in enumerate(batch) if t.strip()]

            if not non_empty:
                continue

            batch_texts = [t for _, t in non_empty]
            try:
                raw = self._call_api_batch(batch_texts)
                parsed = self._parse_batch_response(raw, batch_texts)
            except Exception as exc:
                logger.error(
                    "Batch labeling failed for %d texts: %s",
                    len(batch_texts), exc,
                )
                parsed = {}

            for local_idx, (orig_idx, _) in enumerate(non_empty):
                spans = parsed.get(local_idx, [])
                all_results[batch_start + orig_idx] = spans

        return all_results

    # ------------------------------------------------------------------
    # Token alignment
    # ------------------------------------------------------------------

    @staticmethod
    def align_to_tokens(
        text: str,
        spans: List[SpanLabel],
        tokenizer: Any,
    ) -> List[int]:
        """Convert character-level span labels to token-level binary labels.

        Parameters
        ----------
        text : str
            The original text that was labelled.
        spans : List[SpanLabel]
            Character-level span labels.
        tokenizer
            A HuggingFace tokenizer with ``__call__`` returning
            ``offset_mapping``.

        Returns
        -------
        List[int]
            Token-level labels: 1 = factual (retrieval), 0 = linguistic
            (generation).  Length equals the number of tokens.
        """
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        offsets: List[Tuple[int, int]] = encoding["offset_mapping"]

        # Build character-level array (default = 0 = linguistic)
        char_labels = [0] * len(text)
        for span in spans:
            if span.label == "factual":
                for i in range(span.start_char, min(span.end_char, len(text))):
                    char_labels[i] = 1

        # Map to tokens via majority vote over token's character span
        token_labels: List[int] = []
        for tok_start, tok_end in offsets:
            if tok_start == tok_end:
                token_labels.append(0)
                continue
            factual_chars = sum(char_labels[tok_start:tok_end])
            total_chars = tok_end - tok_start
            token_labels.append(1 if factual_chars > total_chars / 2 else 0)

        return token_labels

    # ------------------------------------------------------------------
    # Batch / corpus labeling
    # ------------------------------------------------------------------

    def label_corpus(
        self,
        texts: List[str],
        output_dir: str,
        *,
        batch_size: int = 5,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Label a corpus with checkpoint / resume support.

        Parameters
        ----------
        texts : List[str]
            List of text chunks to label.
        output_dir : str
            Directory for checkpoint and output files.
        batch_size : int
            Texts per checkpoint file (each text is one API call).
        resume : bool
            If *True*, skip already-labelled chunks.

        Returns
        -------
        Dict[str, Any]
            Summary dict with counts and cost info.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        total_labelled = 0
        total_skipped = 0
        all_labels: List[Dict[str, Any]] = []

        for idx, text in enumerate(texts):
            chunk_path = out_path / f"labels_{idx:06d}.json"

            # Resume: skip if already done
            if resume and chunk_path.exists():
                total_skipped += 1
                continue

            try:
                spans = self.label_text(text)
            except Exception as exc:
                logger.error("Failed to label chunk %d: %s", idx, exc)
                spans = []

            record = {
                "chunk_index": idx,
                "text": text,
                "spans": [s.model_dump() for s in spans],
                "num_spans": len(spans),
            }
            all_labels.append(record)
            total_labelled += 1

            # Write checkpoint atomically
            tmp = chunk_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2, ensure_ascii=False)
            tmp.replace(chunk_path)

            if (total_labelled) % batch_size == 0:
                logger.info(
                    "Labeling progress: %d/%d done, %d skipped, cost $%.4f",
                    total_labelled + total_skipped,
                    len(texts),
                    total_skipped,
                    self.cost_tracker.estimated_cost_usd,
                )

        summary = {
            "total_texts": len(texts),
            "labelled": total_labelled,
            "skipped": total_skipped,
            "cost": self.cost_tracker.summary(),
        }
        # Write corpus-level summary
        summary_path = out_path / "labeling_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Corpus labeling complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: "LabelingConfig") -> "LLMLabeler":
        """Construct from a ``LabelingConfig`` Pydantic model.

        Parameters
        ----------
        cfg : LabelingConfig
            Configuration section from ``config/default.yaml``.

        Returns
        -------
        LLMLabeler
        """
        return cls(
            model=cfg.model,
            api_key=cfg.api_key,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            max_retries=cfg.max_retries,
            retry_delay=cfg.retry_delay,
            rate_limit_rpm=cfg.rate_limit_rpm,
        )
