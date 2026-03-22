"""
Tests for Phase 8 — Inference Pipeline & Server.

Tests cover:
- RetrievedFact dataclass: construction, to_dict, optional fields
- RouterDecision dataclass: construction, to_dict
- FRLMResponse dataclass: construction, to_dict, computed properties
- InferencePipeline: construction, config extraction, _convert_facts,
  _convert_decisions, get_config_summary
- FastAPI server: create_app returns app, endpoints exist, health/config,
  request/response schemas
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- Inference imports ----
from src.inference.pipeline import (
    FRLMResponse,
    InferencePipeline,
    RetrievedFact,
    RouterDecision,
)
from src.inference.server import (
    ConfigResponse,
    EntityFactsResponse,
    FactResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    RetrievedFactResponse,
    RouterDecisionResponse,
    create_app,
)


# ====================================================================
# SECTION 1 — Response Dataclasses
# ====================================================================


class TestRetrievedFact:
    """Tests for RetrievedFact dataclass."""

    def test_defaults(self) -> None:
        fact = RetrievedFact()
        assert fact.fact_id == ""
        assert fact.subject_label == ""
        assert fact.confidence == 0.0
        assert fact.temporal_mode == "CURRENT"
        assert fact.valid_from is None
        assert fact.valid_to is None

    def test_construction(self) -> None:
        fact = RetrievedFact(
            fact_id="abc123",
            subject_label="Aspirin",
            relation_type="TREATS",
            object_label="Headache",
            confidence=0.95,
            temporal_mode="CURRENT",
            source="PubMed",
        )
        assert fact.fact_id == "abc123"
        assert fact.subject_label == "Aspirin"
        assert fact.confidence == 0.95

    def test_to_dict(self) -> None:
        fact = RetrievedFact(
            fact_id="abc",
            subject_label="A",
            relation_type="INHIBITS",
            object_label="B",
            confidence=0.8,
        )
        d = fact.to_dict()
        assert d["fact_id"] == "abc"
        assert d["subject"] == "A"
        assert d["relation"] == "INHIBITS"
        assert d["object"] == "B"
        assert d["confidence"] == 0.8

    def test_to_dict_with_temporal(self) -> None:
        fact = RetrievedFact(
            fact_id="x",
            valid_from="2020-01-01",
            valid_to="2023-12-31",
            source="PMC123",
        )
        d = fact.to_dict()
        assert d["valid_from"] == "2020-01-01"
        assert d["valid_to"] == "2023-12-31"
        assert d["source"] == "PMC123"

    def test_to_dict_excludes_none_temporal(self) -> None:
        fact = RetrievedFact(fact_id="y")
        d = fact.to_dict()
        assert "valid_from" not in d
        assert "valid_to" not in d


class TestRouterDecision:
    """Tests for RouterDecision dataclass."""

    def test_defaults(self) -> None:
        dec = RouterDecision()
        assert dec.step == 0
        assert dec.probability == 0.0
        assert dec.decision == "generation"
        assert dec.num_facts_retrieved == 0

    def test_retrieval_decision(self) -> None:
        dec = RouterDecision(step=5, probability=0.85, decision="retrieval", num_facts_retrieved=3)
        assert dec.decision == "retrieval"
        assert dec.num_facts_retrieved == 3

    def test_to_dict(self) -> None:
        dec = RouterDecision(step=2, probability=0.7, decision="retrieval")
        d = dec.to_dict()
        assert d["step"] == 2
        assert d["probability"] == 0.7
        assert d["decision"] == "retrieval"


class TestFRLMResponse:
    """Tests for FRLMResponse dataclass."""

    def test_defaults(self) -> None:
        resp = FRLMResponse()
        assert resp.generated_text == ""
        assert resp.token_ids is None
        assert resp.retrieved_facts == []
        assert resp.router_decisions == []
        assert resp.num_tokens_generated == 0
        assert resp.inference_time_ms == 0.0
        assert resp.prompt == ""

    def test_num_retrieval_steps(self) -> None:
        resp = FRLMResponse(
            router_decisions=[
                RouterDecision(step=0, decision="generation"),
                RouterDecision(step=1, decision="retrieval"),
                RouterDecision(step=2, decision="retrieval"),
                RouterDecision(step=3, decision="generation"),
            ]
        )
        assert resp.num_retrieval_steps == 2
        assert resp.num_generation_steps == 2

    def test_num_steps_empty(self) -> None:
        resp = FRLMResponse()
        assert resp.num_retrieval_steps == 0
        assert resp.num_generation_steps == 0

    def test_to_dict(self) -> None:
        resp = FRLMResponse(
            prompt="test prompt",
            generated_text="generated output",
            num_tokens_generated=10,
            retrieval_fraction=0.3,
            generation_fraction=0.7,
            inference_time_ms=42.5,
        )
        d = resp.to_dict()
        assert d["prompt"] == "test prompt"
        assert d["generated_text"] == "generated output"
        assert d["num_tokens_generated"] == 10
        assert d["retrieval_fraction"] == 0.3
        assert d["generation_fraction"] == 0.7
        assert d["inference_time_ms"] == 42.5

    def test_to_dict_with_facts(self) -> None:
        resp = FRLMResponse(
            retrieved_facts=[
                RetrievedFact(fact_id="f1", subject_label="A", object_label="B"),
            ],
            router_decisions=[
                RouterDecision(step=0, decision="retrieval"),
            ],
        )
        d = resp.to_dict()
        assert len(d["retrieved_facts"]) == 1
        assert d["retrieved_facts"][0]["fact_id"] == "f1"
        assert len(d["router_decisions"]) == 1


# ====================================================================
# SECTION 2 — InferencePipeline
# ====================================================================


def _make_mock_model() -> MagicMock:
    """Create a mock FRLMModel for pipeline tests."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    return model


def _make_mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tokenizer.decode = MagicMock(return_value="decoded text")
    return tokenizer


class TestInferencePipelineConstruction:
    """Tests for InferencePipeline initialization."""

    def test_default_config(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer)
        assert pipeline.max_length == 512
        assert pipeline.temperature == 0.7
        assert pipeline.top_k == 50
        assert pipeline.top_p == 0.95
        assert pipeline.router_threshold == 0.3
        assert pipeline.device == "cpu"

    def test_custom_config(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        config = MagicMock()
        config.max_length = 256
        config.temperature = 0.5
        config.top_k = 30
        config.top_p = 0.9
        config.do_sample = True
        config.repetition_penalty = 1.2
        config.router_threshold = 0.6
        pipeline = InferencePipeline(
            model=model, tokenizer=tokenizer, config=config
        )
        assert pipeline.max_length == 256
        assert pipeline.temperature == 0.5
        assert pipeline.top_k == 30
        assert pipeline.router_threshold == 0.6

    def test_model_put_to_device_and_eval(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer, device="cpu")
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once()

    def test_with_faiss_and_kg(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        faiss_mock = MagicMock()
        kg_mock = MagicMock()
        pipeline = InferencePipeline(
            model=model, tokenizer=tokenizer,
            faiss_index=faiss_mock, kg_client=kg_mock,
        )
        assert pipeline.faiss_index is faiss_mock
        assert pipeline.kg_client is kg_mock


class TestInferencePipelineFromConfig:
    """Tests for InferencePipeline.from_config factory."""

    def test_from_config_extracts_inference_section(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        full_config = MagicMock()
        full_config.inference.device = "cpu"
        full_config.inference.max_length = 128
        full_config.inference.temperature = 0.3
        full_config.inference.top_k = 20
        full_config.inference.top_p = 0.8
        full_config.inference.do_sample = True
        full_config.inference.repetition_penalty = 1.0
        full_config.inference.router_threshold = 0.4
        pipeline = InferencePipeline.from_config(
            model=model, tokenizer=tokenizer, config=full_config
        )
        assert pipeline.max_length == 128
        assert pipeline.temperature == 0.3
        assert pipeline.router_threshold == 0.4

    def test_from_config_no_config(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        pipeline = InferencePipeline.from_config(
            model=model, tokenizer=tokenizer, config=None
        )
        assert pipeline.device == "cpu"
        assert pipeline.max_length == 512


class TestConvertFacts:
    """Tests for InferencePipeline._convert_facts."""

    def test_convert_dict_facts(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer)
        raw_facts = [
            {
                "fact_id": "f1",
                "subject_label": "Aspirin",
                "relation_type": "TREATS",
                "object_label": "Headache",
                "confidence": 0.95,
                "temporal_mode": "CURRENT",
            },
            {
                "fact_id": "f2",
                "subject_label": "Drug-X",
                "relation_type": "INHIBITS",
                "object_label": "Target-Y",
                "confidence": 0.80,
            },
        ]
        converted = pipeline._convert_facts(raw_facts)
        assert len(converted) == 2
        assert converted[0].fact_id == "f1"
        assert converted[0].subject_label == "Aspirin"
        assert converted[0].confidence == 0.95
        assert converted[1].fact_id == "f2"

    def test_convert_empty_facts(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer)
        assert pipeline._convert_facts([]) == []

    def test_convert_fact_objects(self) -> None:
        """Convert fact-like objects with attributes."""
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer)

        fact_obj = MagicMock()
        fact_obj.fact_id = "f3"
        fact_obj.subject.label = "Protein-A"
        fact_obj.relation.type.value = "ACTIVATES"
        fact_obj.object.label = "Protein-B"
        fact_obj.confidence = 0.75
        fact_obj.is_current = True
        fact_obj.temporal.valid_from = "2020-01-01"
        fact_obj.temporal.valid_to = None
        fact_obj.source = "PMC999"

        converted = pipeline._convert_facts([fact_obj])
        assert len(converted) == 1
        assert converted[0].fact_id == "f3"
        assert converted[0].subject_label == "Protein-A"


class TestConvertDecisions:
    """Tests for InferencePipeline._convert_decisions."""

    def test_convert_dict_decisions(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer)
        raw = [
            {"probability": 0.8, "retrieval": True, "num_facts": 3},
            {"probability": 0.2, "retrieval": False, "num_facts": 0},
        ]
        converted = pipeline._convert_decisions(raw, threshold=0.5)
        assert len(converted) == 2
        assert converted[0].decision == "retrieval"
        assert converted[0].num_facts_retrieved == 3
        assert converted[1].decision == "generation"

    def test_convert_float_decisions(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer)
        raw = [0.8, 0.3, 0.6]
        converted = pipeline._convert_decisions(raw, threshold=0.5)
        assert converted[0].decision == "retrieval"
        assert converted[1].decision == "generation"
        assert converted[2].decision == "retrieval"

    def test_convert_empty_decisions(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer)
        assert pipeline._convert_decisions([], threshold=0.5) == []

    def test_step_indices(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer)
        raw = [0.1, 0.9, 0.5]
        converted = pipeline._convert_decisions(raw, threshold=0.5)
        assert converted[0].step == 0
        assert converted[1].step == 1
        assert converted[2].step == 2


class TestGetConfigSummary:
    """Tests for InferencePipeline.get_config_summary."""

    def test_returns_all_keys(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer)
        summary = pipeline.get_config_summary()
        expected_keys = {
            "device", "max_length", "temperature", "top_k", "top_p",
            "router_threshold", "repetition_penalty",
            "has_faiss_index", "has_kg_client",
        }
        assert set(summary.keys()) == expected_keys

    def test_faiss_and_kg_flags(self) -> None:
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        # No FAISS / KG
        pipeline = InferencePipeline(model=model, tokenizer=tokenizer)
        summary = pipeline.get_config_summary()
        assert summary["has_faiss_index"] is False
        assert summary["has_kg_client"] is False
        # With FAISS / KG
        pipeline2 = InferencePipeline(
            model=model, tokenizer=tokenizer,
            faiss_index=MagicMock(), kg_client=MagicMock(),
        )
        summary2 = pipeline2.get_config_summary()
        assert summary2["has_faiss_index"] is True
        assert summary2["has_kg_client"] is True


# ====================================================================
# SECTION 3 — FastAPI Server
# ====================================================================


class TestCreateApp:
    """Tests for create_app() factory function."""

    def test_returns_fastapi_instance(self) -> None:
        from fastapi import FastAPI
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_has_endpoints(self) -> None:
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/generate" in routes
        assert "/health" in routes
        assert "/config" in routes
        assert "/fact/{fact_id}" in routes
        assert "/entity/{entity_id}/facts" in routes

    def test_app_with_pipeline(self) -> None:
        pipeline_mock = MagicMock()
        pipeline_mock.get_config_summary.return_value = {
            "device": "cpu",
            "max_length": 512,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "router_threshold": 0.5,
            "has_faiss_index": False,
            "has_kg_client": False,
        }
        app = create_app(pipeline=pipeline_mock)
        assert app is not None

    def test_app_with_config(self) -> None:
        config = MagicMock()
        config.serving.cors_origins = ["http://localhost:3000"]
        config.serving.max_concurrent_requests = 8
        config.serving.request_timeout = 30
        app = create_app(config=config)
        assert app is not None


class TestRequestSchemas:
    """Tests for Pydantic request/response schemas."""

    def test_generate_request_minimal(self) -> None:
        req = GenerateRequest(prompt="test prompt")
        assert req.prompt == "test prompt"
        assert req.max_length is None
        assert req.temperature is None

    def test_generate_request_full(self) -> None:
        req = GenerateRequest(
            prompt="test",
            max_length=256,
            temperature=0.5,
            top_k=30,
            top_p=0.9,
            router_threshold=0.6,
        )
        assert req.max_length == 256
        assert req.temperature == 0.5

    def test_generate_request_validation(self) -> None:
        """Empty prompt should fail validation."""
        with pytest.raises(Exception):
            GenerateRequest(prompt="")

    def test_generate_response_defaults(self) -> None:
        resp = GenerateResponse()
        assert resp.prompt == ""
        assert resp.generated_text == ""
        assert resp.num_tokens_generated == 0
        assert resp.retrieved_facts == []
        assert resp.router_decisions == []

    def test_health_response(self) -> None:
        hr = HealthResponse(
            status="ok",
            model_loaded=True,
            faiss_loaded=True,
            kg_connected=False,
            uptime_seconds=123.4,
        )
        assert hr.status == "ok"
        assert hr.model_loaded is True
        assert hr.kg_connected is False

    def test_config_response(self) -> None:
        cr = ConfigResponse(
            device="cuda",
            max_length=1024,
            temperature=0.8,
            has_faiss_index=True,
        )
        assert cr.device == "cuda"
        assert cr.has_faiss_index is True

    def test_fact_response(self) -> None:
        fr = FactResponse(
            fact_id="abc",
            subject_label="Aspirin",
            relation_type="TREATS",
            object_label="Headache",
        )
        assert fr.fact_id == "abc"

    def test_entity_facts_response(self) -> None:
        efr = EntityFactsResponse(
            entity_id="C0004057",
            entity_label="Aspirin",
            facts=[],
            total_count=0,
        )
        assert efr.entity_id == "C0004057"
        assert efr.total_count == 0

    def test_retrieved_fact_response(self) -> None:
        rfr = RetrievedFactResponse(
            fact_id="f1",
            subject="A",
            relation="INHIBITS",
            object="B",
        )
        assert rfr.fact_id == "f1"

    def test_router_decision_response(self) -> None:
        rdr = RouterDecisionResponse(
            step=3,
            probability=0.75,
            decision="retrieval",
            num_facts_retrieved=2,
        )
        assert rdr.step == 3
        assert rdr.decision == "retrieval"


class TestServerEndpointsIntegration:
    """Integration tests for server endpoints via async calls."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self) -> None:
        """Test health endpoint returns correct structure."""
        from src.inference.server import _state
        app = create_app()
        # Directly call the health endpoint handler
        for route in app.routes:
            if getattr(route, "path", None) == "/health":
                endpoint = route.endpoint
                resp = await endpoint()
                assert resp.status == "ok"
                assert resp.model_loaded is False
                break

    @pytest.mark.asyncio
    async def test_config_endpoint_no_pipeline(self) -> None:
        """Config endpoint without pipeline returns defaults."""
        from src.inference.server import _state
        _state.pipeline = None
        app = create_app()
        for route in app.routes:
            if getattr(route, "path", None) == "/config":
                endpoint = route.endpoint
                resp = await endpoint()
                assert resp.device == "cpu"
                assert resp.max_length == 512
                break

    @pytest.mark.asyncio
    async def test_generate_no_pipeline_raises(self) -> None:
        """Generate without pipeline raises 503."""
        from fastapi import HTTPException
        from src.inference.server import _state
        _state.pipeline = None
        app = create_app()
        for route in app.routes:
            if getattr(route, "path", None) == "/generate":
                endpoint = route.endpoint
                req = GenerateRequest(prompt="test")
                with pytest.raises(HTTPException) as exc_info:
                    await endpoint(req)
                assert exc_info.value.status_code == 503
                break

    @pytest.mark.asyncio
    async def test_fact_no_kg_raises(self) -> None:
        """Fact endpoint without KG raises 503."""
        from fastapi import HTTPException
        from src.inference.server import _state
        _state.kg_client = None
        app = create_app()
        for route in app.routes:
            if getattr(route, "path", None) == "/fact/{fact_id}":
                endpoint = route.endpoint
                with pytest.raises(HTTPException) as exc_info:
                    await endpoint("some-id")
                assert exc_info.value.status_code == 503
                break

    @pytest.mark.asyncio
    async def test_entity_facts_no_kg_raises(self) -> None:
        """Entity facts without KG raises 503."""
        from fastapi import HTTPException
        from src.inference.server import _state
        _state.kg_client = None
        app = create_app()
        for route in app.routes:
            if getattr(route, "path", None) == "/entity/{entity_id}/facts":
                endpoint = route.endpoint
                with pytest.raises(HTTPException) as exc_info:
                    await endpoint("some-id")
                assert exc_info.value.status_code == 503
                break


class TestServerWithPipeline:
    """Tests with a mock pipeline configured."""

    @pytest.mark.asyncio
    async def test_health_with_pipeline(self) -> None:
        pipeline_mock = MagicMock()
        pipeline_mock.faiss_index = MagicMock()
        pipeline_mock.get_config_summary.return_value = {
            "device": "cpu", "max_length": 512, "temperature": 0.7,
            "top_k": 50, "top_p": 0.95, "router_threshold": 0.5,
            "has_faiss_index": True, "has_kg_client": True,
        }
        app = create_app(pipeline=pipeline_mock)
        for route in app.routes:
            if getattr(route, "path", None) == "/health":
                resp = await route.endpoint()
                assert resp.model_loaded is True
                assert resp.faiss_loaded is True
                break

    @pytest.mark.asyncio
    async def test_config_with_pipeline(self) -> None:
        pipeline_mock = MagicMock()
        pipeline_mock.get_config_summary.return_value = {
            "device": "cpu", "max_length": 512, "temperature": 0.7,
            "top_k": 50, "top_p": 0.95, "router_threshold": 0.5,
            "has_faiss_index": True, "has_kg_client": True,
        }
        app = create_app(pipeline=pipeline_mock)
        for route in app.routes:
            if getattr(route, "path", None) == "/config":
                resp = await route.endpoint()
                assert resp.device == "cpu"
                assert resp.max_length == 512
                assert resp.has_faiss_index is True
                break

    @pytest.mark.asyncio
    async def test_generate_with_pipeline(self) -> None:
        import asyncio
        pipeline_mock = MagicMock()
        mock_response = FRLMResponse(
            generated_text="Aspirin treats Headache",
            token_ids=[1, 2, 3],
            retrieved_facts=[
                RetrievedFact(fact_id="f1", subject_label="Aspirin",
                              relation_type="TREATS", object_label="Headache",
                              confidence=0.95),
            ],
            router_decisions=[
                RouterDecision(step=0, probability=0.8, decision="retrieval",
                               num_facts_retrieved=1),
            ],
            retrieval_fraction=1.0, generation_fraction=0.0,
            num_tokens_generated=3, inference_time_ms=50.0, prompt="test",
        )
        pipeline_mock.generate.return_value = mock_response
        pipeline_mock.get_config_summary.return_value = {
            "device": "cpu", "max_length": 512, "temperature": 0.7,
            "top_k": 50, "top_p": 0.95, "router_threshold": 0.5,
            "has_faiss_index": True, "has_kg_client": True,
        }
        app = create_app(pipeline=pipeline_mock)
        for route in app.routes:
            if getattr(route, "path", None) == "/generate":
                req = GenerateRequest(prompt="test prompt")
                resp = await route.endpoint(req)
                assert resp.generated_text == "Aspirin treats Headache"
                assert resp.num_tokens_generated == 3
                assert len(resp.retrieved_facts) == 1
                assert resp.retrieved_facts[0].fact_id == "f1"
                break


# ====================================================================
# SECTION 4 — Module-level exports
# ====================================================================


class TestInferenceModuleExports:
    """Ensure the inference __init__.py re-exports everything."""

    def test_pipeline_exports(self) -> None:
        from src.inference import (
            InferencePipeline,
            FRLMResponse,
            RetrievedFact,
            RouterDecision,
        )

    def test_server_export(self) -> None:
        from src.inference import create_app
        from fastapi import FastAPI
        app = create_app()
        assert isinstance(app, FastAPI)
