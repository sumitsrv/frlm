"""
Tests for Phase 5 — Router Label Generation.

Tests cover:
- SpanLabel model: construction, validation, boundary conditions
- LLMLabeler: prompt design, JSON extraction, response parsing,
  token alignment, corpus labeling with checkpoint/resume, factory
- CostTracker: recording, summary, estimated cost
- LabelValidator: statistics, low-confidence filtering, Cohen's kappa
  inter-annotator agreement, threshold validation, Label Studio export
- RouterLabelGenerator alias
- SYSTEM_PROMPT / FEW_SHOT_EXAMPLES constants contain required content

All Claude API calls are mocked — no network access or API key needed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import load_config
from src.labeling.llm_labeler import (
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    CostTracker,
    LLMLabeler,
    SpanLabel,
    _build_user_message,
)
from src.labeling.label_validator import LabelValidator

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_TEXT_FACTUAL = "Gefitinib inhibits EGFR with an IC50 of 33 nM."
_TEXT_LINGUISTIC = "These results suggest that further investigation is warranted."
_TEXT_MIXED = (
    "In this study, we demonstrated that erlotinib (150 mg/day) "
    "significantly improved progression-free survival in EGFR-mutant "
    "NSCLC patients compared to chemotherapy."
)


def _make_span(
    start: int, end: int, text: str, label: str = "factual", conf: float = 0.9
) -> SpanLabel:
    return SpanLabel(
        start_char=start,
        end_char=end,
        text=text,
        label=label,
        confidence=conf,
    )


def _factual_spans() -> List[SpanLabel]:
    """A small list of varied spans for validator tests."""
    return [
        _make_span(0, 47, _TEXT_FACTUAL, "factual", 0.98),
        _make_span(48, 110, _TEXT_LINGUISTIC, "linguistic", 0.95),
        _make_span(111, 130, "some ambiguous text", "factual", 0.60),
    ]


# ===================================================================
# SpanLabel model
# ===================================================================


class TestSpanLabel:
    """Pydantic model validation for SpanLabel."""

    def test_valid_construction(self) -> None:
        s = SpanLabel(
            start_char=0, end_char=10, text="hello test", label="factual", confidence=0.9
        )
        assert s.start_char == 0
        assert s.end_char == 10
        assert s.label == "factual"

    def test_linguistic_label(self) -> None:
        s = SpanLabel(
            start_char=5, end_char=20, text="x" * 15, label="linguistic", confidence=0.5
        )
        assert s.label == "linguistic"

    def test_invalid_label_rejected(self) -> None:
        with pytest.raises(ValueError):
            SpanLabel(
                start_char=0, end_char=5, text="hello", label="unknown", confidence=0.5
            )

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValueError):
            SpanLabel(
                start_char=10, end_char=5, text="hello", label="factual", confidence=0.5
            )

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValueError):
            SpanLabel(
                start_char=-1, end_char=5, text="hello", label="factual", confidence=0.5
            )

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            SpanLabel(
                start_char=0, end_char=5, text="hello", label="factual", confidence=-0.1
            )

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValueError):
            SpanLabel(
                start_char=0, end_char=5, text="hello", label="factual", confidence=1.1
            )

    def test_zero_length_span_allowed(self) -> None:
        s = SpanLabel(
            start_char=5, end_char=5, text="", label="linguistic", confidence=0.5
        )
        assert s.start_char == s.end_char

    def test_model_dump_round_trip(self) -> None:
        s = _make_span(0, 10, "0123456789")
        data = s.model_dump()
        s2 = SpanLabel(**data)
        assert s == s2


# ===================================================================
# CostTracker
# ===================================================================


class TestCostTracker:
    def test_initial_state(self) -> None:
        ct = CostTracker()
        assert ct.total_requests == 0
        assert ct.estimated_cost_usd == 0.0

    def test_record_accumulates(self) -> None:
        ct = CostTracker()
        ct.record(1000, 500)
        ct.record(2000, 1000)
        assert ct.total_input_tokens == 3000
        assert ct.total_output_tokens == 1500
        assert ct.total_requests == 2

    def test_estimated_cost(self) -> None:
        ct = CostTracker()
        ct.record(1_000_000, 0)  # $3.00 input
        assert abs(ct.estimated_cost_usd - 3.00) < 0.01

    def test_estimated_cost_output(self) -> None:
        ct = CostTracker()
        ct.record(0, 1_000_000)  # $15.00 output
        assert abs(ct.estimated_cost_usd - 15.00) < 0.01

    def test_summary_keys(self) -> None:
        ct = CostTracker()
        ct.record(100, 50)
        s = ct.summary()
        assert "total_requests" in s
        assert "total_input_tokens" in s
        assert "total_output_tokens" in s
        assert "estimated_cost_usd" in s


# ===================================================================
# Prompt content verification
# ===================================================================


class TestPromptContent:
    """Verify the Claude prompt contains the required elements."""

    def test_system_prompt_defines_factual(self) -> None:
        assert "factual" in SYSTEM_PROMPT.lower()
        assert "entity" in SYSTEM_PROMPT.lower()
        assert "measurement" in SYSTEM_PROMPT.lower()
        assert "knowledge base" in SYSTEM_PROMPT.lower()

    def test_system_prompt_defines_linguistic(self) -> None:
        assert "linguistic" in SYSTEM_PROMPT.lower()
        assert "discourse" in SYSTEM_PROMPT.lower()
        assert "hedging" in SYSTEM_PROMPT.lower()

    def test_system_prompt_ambiguity_rule(self) -> None:
        assert "ambiguous" in SYSTEM_PROMPT.lower() or "ambiguity" in SYSTEM_PROMPT.lower()

    def test_system_prompt_json_output(self) -> None:
        assert "json" in SYSTEM_PROMPT.lower()
        assert '"span"' in SYSTEM_PROMPT
        assert '"label"' in SYSTEM_PROMPT
        assert '"confidence"' in SYSTEM_PROMPT

    def test_few_shot_example_factual(self) -> None:
        assert "Gefitinib inhibits EGFR" in FEW_SHOT_EXAMPLES
        assert "IC50 of 33 nM" in FEW_SHOT_EXAMPLES

    def test_few_shot_example_linguistic(self) -> None:
        assert "further investigation is warranted" in FEW_SHOT_EXAMPLES

    def test_few_shot_example_ambiguous(self) -> None:
        assert "oxidative stress" in FEW_SHOT_EXAMPLES

    def test_few_shot_has_at_least_three_examples(self) -> None:
        assert FEW_SHOT_EXAMPLES.count("Example") >= 3

    def test_build_user_message_includes_text(self) -> None:
        msg = _build_user_message("Test input.")
        assert "Test input." in msg
        assert "annotate" in msg.lower() or "json" in msg.lower()


# ===================================================================
# LLMLabeler — JSON extraction
# ===================================================================


class TestJSONExtraction:
    """Test _extract_json with various response formats."""

    def test_plain_json_array(self) -> None:
        raw = '[{"span": "hello", "label": "factual", "confidence": 0.9}]'
        result = LLMLabeler._extract_json(raw)
        assert len(result) == 1
        assert result[0]["span"] == "hello"

    def test_markdown_fenced_json(self) -> None:
        raw = '```json\n[{"span": "x", "label": "factual", "confidence": 0.8}]\n```'
        result = LLMLabeler._extract_json(raw)
        assert len(result) == 1

    def test_markdown_fenced_no_lang(self) -> None:
        raw = '```\n[{"span": "y", "label": "linguistic", "confidence": 0.7}]\n```'
        result = LLMLabeler._extract_json(raw)
        assert result[0]["label"] == "linguistic"

    def test_json_with_surrounding_text(self) -> None:
        raw = 'Here is the result:\n[{"span": "z", "label": "factual", "confidence": 0.6}]\nDone.'
        result = LLMLabeler._extract_json(raw)
        assert len(result) == 1

    def test_invalid_json_raises(self) -> None:
        with pytest.raises((json.JSONDecodeError, ValueError)):
            LLMLabeler._extract_json("not json at all")

    def test_non_array_raises(self) -> None:
        with pytest.raises(ValueError):
            LLMLabeler._extract_json('{"span": "hello"}')


# ===================================================================
# LLMLabeler — response parsing
# ===================================================================


class TestResponseParsing:
    """Test _parse_response converting raw JSON to SpanLabel objects."""

    def test_single_span_exact_match(self) -> None:
        text = "Gefitinib inhibits EGFR."
        raw = json.dumps([{"span": text, "label": "factual", "confidence": 0.95}])
        spans = LLMLabeler._parse_response(raw, text)
        assert len(spans) == 1
        assert spans[0].start_char == 0
        assert spans[0].end_char == len(text)
        assert spans[0].label == "factual"

    def test_multiple_spans_sequential(self) -> None:
        text = "Hello world. Goodbye world."
        raw = json.dumps([
            {"span": "Hello world.", "label": "linguistic", "confidence": 0.8},
            {"span": " Goodbye world.", "label": "factual", "confidence": 0.9},
        ])
        spans = LLMLabeler._parse_response(raw, text)
        assert len(spans) == 2
        assert spans[0].end_char <= spans[1].start_char

    def test_label_normalisation_retrieval(self) -> None:
        text = "Drug X."
        raw = json.dumps([{"span": "Drug X.", "label": "RETRIEVAL", "confidence": 0.9}])
        spans = LLMLabeler._parse_response(raw, text)
        assert spans[0].label == "factual"

    def test_label_normalisation_generation(self) -> None:
        text = "Moreover, the data show."
        raw = json.dumps([
            {"span": "Moreover, the data show.", "label": "generation", "confidence": 0.85}
        ])
        spans = LLMLabeler._parse_response(raw, text)
        assert spans[0].label == "linguistic"

    def test_unknown_label_defaults_linguistic(self) -> None:
        text = "Some text."
        raw = json.dumps([{"span": "Some text.", "label": "unknown_type", "confidence": 0.5}])
        spans = LLMLabeler._parse_response(raw, text)
        assert spans[0].label == "linguistic"

    def test_span_not_found_skipped(self) -> None:
        text = "Actual text here."
        raw = json.dumps([
            {"span": "nonexistent text", "label": "factual", "confidence": 0.9}
        ])
        spans = LLMLabeler._parse_response(raw, text)
        assert len(spans) == 0

    def test_confidence_clamped(self) -> None:
        text = "hi"
        raw = json.dumps([{"span": "hi", "label": "factual", "confidence": 1.5}])
        spans = LLMLabeler._parse_response(raw, text)
        assert spans[0].confidence == 1.0


# ===================================================================
# LLMLabeler — token alignment
# ===================================================================


class TestTokenAlignment:
    """Test align_to_tokens with a mock tokenizer."""

    @staticmethod
    def _mock_tokenizer(text: str, **kwargs: Any) -> Dict[str, Any]:
        """Simple word-level mock tokenizer with offset_mapping."""
        offsets = []
        i = 0
        for word in text.split():
            start = text.index(word, i)
            end = start + len(word)
            offsets.append((start, end))
            i = end
        return {"offset_mapping": offsets}

    def test_all_factual(self) -> None:
        text = "Drug A inhibits target B."
        spans = [_make_span(0, len(text), text, "factual", 0.95)]
        labels = LLMLabeler.align_to_tokens(text, spans, self._mock_tokenizer)
        assert all(l == 1 for l in labels)

    def test_all_linguistic(self) -> None:
        text = "However, we observed that"
        spans = [_make_span(0, len(text), text, "linguistic", 0.9)]
        labels = LLMLabeler.align_to_tokens(text, spans, self._mock_tokenizer)
        assert all(l == 0 for l in labels)

    def test_mixed_labels(self) -> None:
        text = "Moreover, drug X inhibits Y."
        # "Moreover," = linguistic, rest = factual
        spans = [
            _make_span(0, 9, "Moreover,", "linguistic", 0.9),
            _make_span(10, len(text), "drug X inhibits Y.", "factual", 0.9),
        ]
        labels = LLMLabeler.align_to_tokens(text, spans, self._mock_tokenizer)
        # First token "Moreover," → 0, rest → 1
        assert labels[0] == 0
        assert all(l == 1 for l in labels[1:])

    def test_empty_spans(self) -> None:
        text = "Some text"
        labels = LLMLabeler.align_to_tokens(text, [], self._mock_tokenizer)
        assert all(l == 0 for l in labels)


# ===================================================================
# LLMLabeler — label_text (mocked API)
# ===================================================================


class TestLabelText:
    """Test label_text with mocked Claude API."""

    def _mock_labeler(self) -> LLMLabeler:
        labeler = LLMLabeler(api_key="test-key", max_retries=1)
        labeler._client = MagicMock()
        return labeler

    def test_label_text_returns_spans(self) -> None:
        labeler = self._mock_labeler()
        text = "Drug A inhibits target B."
        response_json = json.dumps([
            {"span": text, "label": "factual", "confidence": 0.95}
        ])
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=response_json)]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        labeler._client.messages.create.return_value = mock_response

        spans = labeler.label_text(text)
        assert len(spans) == 1
        assert spans[0].label == "factual"
        assert labeler.cost_tracker.total_requests == 1

    def test_label_text_empty_string(self) -> None:
        labeler = self._mock_labeler()
        spans = labeler.label_text("")
        assert spans == []

    def test_label_text_whitespace_only(self) -> None:
        labeler = self._mock_labeler()
        spans = labeler.label_text("   \n\t  ")
        assert spans == []

    def test_all_retries_exhausted_raises(self) -> None:
        labeler = self._mock_labeler()
        labeler._client.messages.create.side_effect = RuntimeError("API down")
        with pytest.raises(RuntimeError, match="API attempts failed"):
            labeler.label_text("Some text")


# ===================================================================
# LLMLabeler — corpus labeling with checkpoint/resume
# ===================================================================


class TestLabelCorpus:
    """Test label_corpus batch processing and resume."""

    def _mock_labeler_for_corpus(self) -> LLMLabeler:
        labeler = LLMLabeler(api_key="test-key", max_retries=1)
        labeler._client = MagicMock()

        def fake_create(**kwargs):
            msg = kwargs.get("messages", [{}])[0].get("content", "")
            response_json = json.dumps([
                {"span": "some text", "label": "factual", "confidence": 0.9}
            ])
            mock_resp = MagicMock()
            mock_resp.content = [MagicMock(text=response_json)]
            mock_resp.usage = MagicMock(input_tokens=50, output_tokens=30)
            return mock_resp

        labeler._client.messages.create.side_effect = fake_create
        return labeler

    def test_corpus_labeling_creates_files(self) -> None:
        labeler = self._mock_labeler_for_corpus()
        texts = ["Text one.", "Text two.", "Text three."]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = labeler.label_corpus(texts, tmpdir, batch_size=2)
            assert summary["labelled"] == 3
            assert summary["skipped"] == 0
            # Check files created
            files = list(Path(tmpdir).glob("labels_*.json"))
            assert len(files) == 3
            # Check summary file
            assert (Path(tmpdir) / "labeling_summary.json").exists()

    def test_corpus_resume_skips_existing(self) -> None:
        labeler = self._mock_labeler_for_corpus()
        texts = ["Text one.", "Text two."]
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create one checkpoint
            (Path(tmpdir) / "labels_000000.json").write_text("{}")
            summary = labeler.label_corpus(texts, tmpdir, batch_size=5)
            assert summary["skipped"] == 1
            assert summary["labelled"] == 1

    def test_corpus_no_resume(self) -> None:
        labeler = self._mock_labeler_for_corpus()
        texts = ["Text one."]
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "labels_000000.json").write_text("{}")
            summary = labeler.label_corpus(texts, tmpdir, resume=False)
            assert summary["labelled"] == 1
            assert summary["skipped"] == 0


# ===================================================================
# LLMLabeler — from_config factory
# ===================================================================


class TestLLMLabelerFactory:
    def test_from_config(self) -> None:
        cfg = load_config()
        labeler = LLMLabeler.from_config(cfg.labeling)
        assert labeler.model == cfg.labeling.model
        assert labeler.max_tokens == cfg.labeling.max_tokens
        assert labeler.temperature == cfg.labeling.temperature
        assert labeler.max_retries == cfg.labeling.max_retries


# ===================================================================
# LabelValidator — statistics
# ===================================================================


class TestValidatorStatistics:
    def test_empty_labels(self) -> None:
        stats = LabelValidator.compute_statistics([])
        assert stats["total_spans"] == 0
        assert stats["class_distribution"] == {}

    def test_basic_statistics(self) -> None:
        spans = _factual_spans()
        stats = LabelValidator.compute_statistics(spans)
        assert stats["total_spans"] == 3
        assert "factual" in stats["class_distribution"]
        assert "linguistic" in stats["class_distribution"]
        assert stats["class_distribution"]["factual"] == 2
        assert stats["class_distribution"]["linguistic"] == 1

    def test_class_ratio(self) -> None:
        spans = _factual_spans()
        stats = LabelValidator.compute_statistics(spans)
        assert abs(stats["class_ratio"]["factual"] - 2 / 3) < 0.01

    def test_avg_confidence(self) -> None:
        spans = _factual_spans()
        stats = LabelValidator.compute_statistics(spans)
        expected = (0.98 + 0.95 + 0.60) / 3
        assert abs(stats["avg_confidence"] - expected) < 0.01

    def test_span_length_stats_keys(self) -> None:
        spans = _factual_spans()
        stats = LabelValidator.compute_statistics(spans)
        for key in ("min", "max", "mean", "median"):
            assert key in stats["span_length_stats"]

    def test_per_class_confidence(self) -> None:
        spans = _factual_spans()
        stats = LabelValidator.compute_statistics(spans)
        assert "factual" in stats["avg_confidence_per_class"]
        assert "linguistic" in stats["avg_confidence_per_class"]


# ===================================================================
# LabelValidator — low confidence filtering
# ===================================================================


class TestLowConfidence:
    def test_find_low_confidence_default_threshold(self) -> None:
        spans = _factual_spans()
        low = LabelValidator.find_low_confidence(spans)
        assert len(low) == 1
        assert low[0].confidence == 0.60

    def test_find_low_confidence_custom_threshold(self) -> None:
        spans = _factual_spans()
        low = LabelValidator.find_low_confidence(spans, threshold=0.96)
        assert len(low) == 2  # 0.60 and 0.95

    def test_find_low_confidence_sorted(self) -> None:
        spans = _factual_spans()
        low = LabelValidator.find_low_confidence(spans, threshold=1.0)
        for i in range(len(low) - 1):
            assert low[i].confidence <= low[i + 1].confidence

    def test_find_low_confidence_none_below(self) -> None:
        spans = [_make_span(0, 5, "hello", "factual", 0.99)]
        low = LabelValidator.find_low_confidence(spans, threshold=0.7)
        assert len(low) == 0


# ===================================================================
# LabelValidator — inter-annotator agreement
# ===================================================================


class TestInterAnnotatorAgreement:
    def test_perfect_agreement(self) -> None:
        spans = [_make_span(0, 10, "0123456789", "factual", 0.9)]
        result = LabelValidator.inter_annotator_agreement(spans, spans, 10)
        assert result["kappa"] == 1.0
        assert result["observed_agreement"] == 1.0

    def test_zero_length_text(self) -> None:
        result = LabelValidator.inter_annotator_agreement([], [], 0)
        assert result["kappa"] == 0.0

    def test_complete_disagreement(self) -> None:
        # LLM marks first half factual, human marks second half factual
        # This creates maximum disagreement (kappa < 0)
        llm = [_make_span(0, 10, "0123456789", "factual", 0.9)]
        human = [_make_span(10, 20, "abcdefghij", "factual", 0.9)]
        result = LabelValidator.inter_annotator_agreement(llm, human, 20)
        assert result["kappa"] < 0.0  # negative kappa = worse than chance

    def test_partial_agreement(self) -> None:
        # 20-char text; LLM: first 10 factual, human: first 15 factual
        llm = [_make_span(0, 10, "x" * 10, "factual", 0.9)]
        human = [_make_span(0, 15, "x" * 15, "factual", 0.9)]
        result = LabelValidator.inter_annotator_agreement(llm, human, 20)
        assert 0.0 < result["kappa"] < 1.0
        assert result["confusion_matrix"]["tp"] == 10
        assert result["confusion_matrix"]["fn"] == 5

    def test_confusion_matrix_sums(self) -> None:
        llm = [_make_span(0, 5, "hello", "factual", 0.9)]
        human = [_make_span(3, 8, "lo wo", "factual", 0.9)]
        result = LabelValidator.inter_annotator_agreement(llm, human, 10)
        cm = result["confusion_matrix"]
        assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == 10


# ===================================================================
# LabelValidator — threshold validation
# ===================================================================


class TestValidateLabels:
    def test_valid_labels(self) -> None:
        # 100-char text, 40 chars factual = 0.4 ratio (within 0.15–0.70)
        spans = [
            _make_span(0, 40, "f" * 40, "factual", 0.9),
            _make_span(40, 100, "l" * 60, "linguistic", 0.9),
        ]
        is_valid, issues = LabelValidator.validate_labels(spans, 100)
        assert is_valid is True
        assert issues == []

    def test_empty_labels_invalid(self) -> None:
        is_valid, issues = LabelValidator.validate_labels([], 100)
        assert is_valid is False
        assert "No labels" in issues[0]

    def test_ratio_too_low(self) -> None:
        # 100-char text, 5 chars factual = 0.05 ratio < 0.15
        spans = [
            _make_span(0, 5, "f" * 5, "factual", 0.9),
            _make_span(5, 100, "l" * 95, "linguistic", 0.9),
        ]
        is_valid, issues = LabelValidator.validate_labels(spans, 100)
        assert is_valid is False
        assert any("too low" in i.lower() for i in issues)

    def test_ratio_too_high(self) -> None:
        # 100-char text, 90 chars factual = 0.90 ratio > 0.85 (default max)
        spans = [
            _make_span(0, 90, "f" * 90, "factual", 0.9),
            _make_span(90, 100, "l" * 10, "linguistic", 0.9),
        ]
        is_valid, issues = LabelValidator.validate_labels(spans, 100)
        assert is_valid is False
        assert any("too high" in i.lower() for i in issues)

    def test_too_many_spans(self) -> None:
        spans = [_make_span(i, i + 1, "x", "factual", 0.9) for i in range(60)]
        is_valid, issues = LabelValidator.validate_labels(
            spans, 100, max_spans_per_chunk=50
        )
        assert is_valid is False
        assert any("too many" in i.lower() for i in issues)

    def test_custom_thresholds(self) -> None:
        spans = [_make_span(0, 50, "f" * 50, "factual", 0.9)]
        is_valid, _ = LabelValidator.validate_labels(
            spans, 100, min_retrieval_ratio=0.0, max_retrieval_ratio=1.0
        )
        assert is_valid is True


# ===================================================================
# LabelValidator — Label Studio export
# ===================================================================


class TestExportForReview:
    def test_single_text_export(self) -> None:
        spans = [_make_span(0, 10, "0123456789", "factual", 0.95)]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = LabelValidator.export_for_review(
                "0123456789 and more",
                spans,
                Path(tmpdir) / "review.json",
                source_id="test_001",
            )
            assert out.exists()
            with open(out) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1
            task = data[0]
            assert task["data"]["text"] == "0123456789 and more"
            assert task["data"]["source_id"] == "test_001"
            results = task["annotations"][0]["result"]
            assert len(results) == 1
            assert results[0]["value"]["labels"] == ["factual"]
            assert results[0]["score"] == 0.95

    def test_export_creates_parent_dirs(self) -> None:
        spans = [_make_span(0, 3, "abc", "linguistic", 0.8)]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = LabelValidator.export_for_review(
                "abc", spans, Path(tmpdir) / "sub" / "deep" / "review.json"
            )
            assert out.exists()

    def test_corpus_export_multiple_tasks(self) -> None:
        records = [
            {
                "text": "Text one.",
                "spans": [_make_span(0, 9, "Text one.", "factual", 0.9)],
            },
            {
                "text": "Text two.",
                "spans": [
                    {"start_char": 0, "end_char": 9, "text": "Text two.",
                     "label": "linguistic", "confidence": 0.85}
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = LabelValidator.export_corpus_for_review(
                records, Path(tmpdir) / "corpus_review.json"
            )
            assert out.exists()
            with open(out) as f:
                data = json.load(f)
            assert len(data) == 2
            assert data[0]["data"]["text"] == "Text one."
            assert data[1]["annotations"][0]["result"][0]["value"]["labels"] == ["linguistic"]


# ===================================================================
# Config integration
# ===================================================================


class TestLabelingConfig:
    """Verify labeling section is present and valid in default config."""

    def test_labeling_section_exists(self) -> None:
        cfg = load_config()
        assert hasattr(cfg, "labeling")

    def test_labeling_model(self) -> None:
        cfg = load_config()
        assert "claude" in cfg.labeling.model.lower() or "sonnet" in cfg.labeling.model.lower()

    def test_labeling_temperature(self) -> None:
        cfg = load_config()
        assert cfg.labeling.temperature == 0.0

    def test_labeling_validation_ratios(self) -> None:
        cfg = load_config()
        v = cfg.labeling.validation
        assert 0.0 < v.min_retrieval_ratio < v.max_retrieval_ratio < 1.0

    def test_labeling_max_retries(self) -> None:
        cfg = load_config()
        assert cfg.labeling.max_retries >= 1

    def test_labeling_rate_limit(self) -> None:
        cfg = load_config()
        assert cfg.labeling.rate_limit_rpm > 0

    def test_iaa_settings(self) -> None:
        cfg = load_config()
        assert cfg.labeling.inter_annotator_samples > 0
        assert 0.0 < cfg.labeling.min_agreement_threshold <= 1.0


# ===================================================================
# RouterLabelGenerator alias
# ===================================================================


class TestRouterLabelGeneratorAlias:
    def test_alias_is_same_class(self) -> None:
        from src.labeling import RouterLabelGenerator
        assert RouterLabelGenerator is LLMLabeler

    def test_alias_exported_in_all(self) -> None:
        import src.labeling as mod
        assert "RouterLabelGenerator" in mod.__all__

    def test_span_label_exported(self) -> None:
        import src.labeling as mod
        assert "SpanLabel" in mod.__all__
