"""
FRLM Inference Server — FastAPI application for real-time inference.

Endpoints
---------
- ``POST /generate``     — generate text from a prompt
- ``GET  /fact/{id}``    — retrieve a single fact by ID
- ``GET  /entity/{id}/facts`` — retrieve all facts for an entity
- ``GET  /health``       — health check
- ``GET  /config``       — current pipeline configuration
- ``GET  /docs``         — Swagger UI (auto-generated)

The server wraps :class:`InferencePipeline` and exposes it as a REST API
with CORS, request timeouts, and concurrency limits.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ===========================================================================
# Request / Response schemas
# ===========================================================================


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_length: Optional[int] = Field(
        default=None, ge=1, le=4096, description="Maximum generation length"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=1000, description="Top-k sampling"
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Top-p (nucleus) sampling"
    )
    router_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Router decision threshold"
    )


class RetrievedFactResponse(BaseModel):
    """Schema for a retrieved fact in the response."""

    fact_id: str = ""
    subject: str = ""
    relation: str = ""
    object: str = ""
    confidence: float = 0.0
    temporal_mode: str = "CURRENT"
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source: str = ""


class RouterDecisionResponse(BaseModel):
    """Schema for a router decision in the response."""

    step: int = 0
    probability: float = 0.0
    decision: str = "generation"
    num_facts_retrieved: int = 0


class GenerateResponse(BaseModel):
    """Response body for the /generate endpoint."""

    prompt: str = ""
    generated_text: str = ""
    num_tokens_generated: int = 0
    retrieval_fraction: float = 0.0
    generation_fraction: float = 0.0
    retrieved_facts: List[RetrievedFactResponse] = Field(default_factory=list)
    router_decisions: List[RouterDecisionResponse] = Field(default_factory=list)
    inference_time_ms: float = 0.0


class FactResponse(BaseModel):
    """Response body for the /fact/{id} endpoint."""

    fact_id: str
    subject_id: str = ""
    subject_label: str = ""
    relation_type: str = ""
    object_id: str = ""
    object_label: str = ""
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    is_current: bool = True
    source: str = ""
    confidence: float = 0.0


class EntityFactsResponse(BaseModel):
    """Response body for the /entity/{id}/facts endpoint."""

    entity_id: str
    entity_label: str = ""
    facts: List[FactResponse] = Field(default_factory=list)
    total_count: int = 0


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str = "ok"
    model_loaded: bool = False
    faiss_loaded: bool = False
    kg_connected: bool = False
    uptime_seconds: float = 0.0


class ConfigResponse(BaseModel):
    """Response body for the /config endpoint."""

    device: str = "cpu"
    max_length: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    router_threshold: float = 0.5
    has_faiss_index: bool = False
    has_kg_client: bool = False


# ===========================================================================
# Application state
# ===========================================================================


class _AppState:
    """Holds runtime state for the server."""

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.kg_client: Any = None
        self.start_time: float = time.time()
        self.request_semaphore: Optional[asyncio.Semaphore] = None
        self.request_timeout: int = 60

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time


_state = _AppState()


# ===========================================================================
# Application factory
# ===========================================================================


def create_app(
    pipeline: Any = None,
    kg_client: Any = None,
    config: Any = None,
) -> FastAPI:
    """Create the FastAPI application.

    Parameters
    ----------
    pipeline : InferencePipeline, optional
        Pre-initialized inference pipeline.
    kg_client : Neo4jClient, optional
        KG client for fact lookup endpoints.
    config : ServingConfig or FRLMConfig, optional
        Server configuration.

    Returns
    -------
    FastAPI
        Configured FastAPI application.
    """
    # Extract serving config
    if config is not None and hasattr(config, "serving"):
        serving_cfg = config.serving
    elif config is not None and hasattr(config, "cors_origins"):
        serving_cfg = config
    else:
        serving_cfg = None

    cors_origins = (
        getattr(serving_cfg, "cors_origins", ["*"]) if serving_cfg else ["*"]
    )
    max_concurrent = (
        getattr(serving_cfg, "max_concurrent_requests", 16) if serving_cfg else 16
    )
    request_timeout = (
        getattr(serving_cfg, "request_timeout", 60) if serving_cfg else 60
    )

    app = FastAPI(
        title="FRLM Inference API",
        description=(
            "Factual Retrieval Language Model — real-time inference API "
            "with interleaved knowledge graph retrieval and language generation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store state
    _state.pipeline = pipeline
    _state.kg_client = kg_client
    _state.request_semaphore = asyncio.Semaphore(max_concurrent)
    _state.request_timeout = request_timeout
    _state.start_time = time.time()

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
    async def generate(request: GenerateRequest) -> GenerateResponse:
        """Generate text with interleaved knowledge retrieval.

        The FRLM model processes the prompt through its backbone,
        router, retrieval head, and generation head to produce
        factually-grounded text.
        """
        if _state.pipeline is None:
            raise HTTPException(
                status_code=503, detail="Model not loaded. Pipeline unavailable."
            )

        async with _state.request_semaphore:
            try:
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: _state.pipeline.generate(
                            prompt=request.prompt,
                            max_length=request.max_length,
                            temperature=request.temperature,
                            top_k=request.top_k,
                            top_p=request.top_p,
                            router_threshold=request.router_threshold,
                        ),
                    ),
                    timeout=_state.request_timeout,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Request timed out after {_state.request_timeout}s",
                )
            except Exception as e:
                logger.exception("Generation failed")
                raise HTTPException(status_code=500, detail=str(e))

        # Convert pipeline response to API response
        facts = [
            RetrievedFactResponse(
                fact_id=f.fact_id,
                subject=f.subject_label,
                relation=f.relation_type,
                object=f.object_label,
                confidence=f.confidence,
                temporal_mode=f.temporal_mode,
                valid_from=f.valid_from,
                valid_to=f.valid_to,
                source=f.source,
            )
            for f in response.retrieved_facts
        ]

        decisions = [
            RouterDecisionResponse(
                step=d.step,
                probability=d.probability,
                decision=d.decision,
                num_facts_retrieved=d.num_facts_retrieved,
            )
            for d in response.router_decisions
        ]

        return GenerateResponse(
            prompt=response.prompt,
            generated_text=response.generated_text,
            num_tokens_generated=response.num_tokens_generated,
            retrieval_fraction=response.retrieval_fraction,
            generation_fraction=response.generation_fraction,
            retrieved_facts=facts,
            router_decisions=decisions,
            inference_time_ms=response.inference_time_ms,
        )

    @app.get("/fact/{fact_id}", response_model=FactResponse, tags=["Knowledge Graph"])
    async def get_fact(fact_id: str) -> FactResponse:
        """Retrieve a single fact by its SHA-256 fact ID."""
        if _state.kg_client is None:
            raise HTTPException(
                status_code=503, detail="KG client not available."
            )

        fact = _state.kg_client.get_fact_by_id(fact_id)
        if fact is None:
            raise HTTPException(status_code=404, detail=f"Fact '{fact_id}' not found.")

        return FactResponse(
            fact_id=fact.fact_id,
            subject_id=fact.subject.canonical_id,
            subject_label=fact.subject.label,
            relation_type=fact.relation.type.value,
            object_id=fact.object.canonical_id,
            object_label=fact.object.label,
            valid_from=str(fact.temporal.valid_from),
            valid_to=str(fact.temporal.valid_to) if fact.temporal.valid_to else None,
            is_current=fact.is_current,
            source=fact.source,
            confidence=fact.confidence,
        )

    @app.get(
        "/entity/{entity_id}/facts",
        response_model=EntityFactsResponse,
        tags=["Knowledge Graph"],
    )
    async def get_entity_facts(
        entity_id: str,
        current_only: bool = False,
        limit: int = 100,
    ) -> EntityFactsResponse:
        """Retrieve all facts associated with an entity.

        Parameters
        ----------
        entity_id : str
            Canonical entity ID (e.g., UMLS CUI).
        current_only : bool
            If True, only return currently valid facts.
        limit : int
            Maximum number of facts to return.
        """
        if _state.kg_client is None:
            raise HTTPException(
                status_code=503, detail="KG client not available."
            )

        # Query KG for facts involving this entity
        try:
            facts = _state.kg_client.get_facts_for_entity(
                entity_id, current_only=current_only, limit=limit
            )
        except Exception as e:
            logger.exception("Failed to fetch facts for entity %s", entity_id)
            raise HTTPException(status_code=500, detail=str(e))

        fact_responses = []
        entity_label = ""
        for fact in facts:
            if fact.subject.canonical_id == entity_id:
                entity_label = fact.subject.label
            elif fact.object.canonical_id == entity_id:
                entity_label = fact.object.label

            fact_responses.append(
                FactResponse(
                    fact_id=fact.fact_id,
                    subject_id=fact.subject.canonical_id,
                    subject_label=fact.subject.label,
                    relation_type=fact.relation.type.value,
                    object_id=fact.object.canonical_id,
                    object_label=fact.object.label,
                    valid_from=str(fact.temporal.valid_from),
                    valid_to=str(fact.temporal.valid_to)
                    if fact.temporal.valid_to
                    else None,
                    is_current=fact.is_current,
                    source=fact.source,
                    confidence=fact.confidence,
                )
            )

        return EntityFactsResponse(
            entity_id=entity_id,
            entity_label=entity_label,
            facts=fact_responses[:limit],
            total_count=len(fact_responses),
        )

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            model_loaded=_state.pipeline is not None,
            faiss_loaded=(
                _state.pipeline is not None
                and getattr(_state.pipeline, "faiss_index", None) is not None
            ),
            kg_connected=_state.kg_client is not None,
            uptime_seconds=round(_state.uptime, 2),
        )

    @app.get("/config", response_model=ConfigResponse, tags=["System"])
    async def get_config() -> ConfigResponse:
        """Return current pipeline configuration."""
        if _state.pipeline is None:
            return ConfigResponse()

        cfg = _state.pipeline.get_config_summary()
        return ConfigResponse(
            device=cfg.get("device", "cpu"),
            max_length=cfg.get("max_length", 512),
            temperature=cfg.get("temperature", 0.7),
            top_k=cfg.get("top_k", 50),
            top_p=cfg.get("top_p", 0.95),
            router_threshold=cfg.get("router_threshold", 0.5),
            has_faiss_index=cfg.get("has_faiss_index", False),
            has_kg_client=cfg.get("has_kg_client", False),
        )

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Any:
        """Log incoming requests."""
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    logger.info(
        "FRLM server created: cors=%s, max_concurrent=%d, timeout=%ds",
        cors_origins,
        max_concurrent,
        request_timeout,
    )
    return app


# ===========================================================================
# CLI entry point
# ===========================================================================


def main() -> None:
    """Launch the FRLM inference server via uvicorn.

    This is the entry point referenced by ``setup.py`` console_scripts
    (``frlm-server = src.inference.server:main``).
    """
    import argparse

    import uvicorn

    from config.config import load_config, setup_logging

    parser = argparse.ArgumentParser(
        description="FRLM Inference Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to FRLM config YAML.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind host.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Bind port.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of uvicorn workers.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    app = create_app(config=cfg)

    logger.info(
        "Starting FRLM server on %s:%d (workers=%d)",
        args.host, args.port, args.workers,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
