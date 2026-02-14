#!/usr/bin/env python3
"""
11_run_inference.py - Run inference in batch or serve mode.

In batch mode: processes input texts via InferencePipeline.
In serve mode: starts the FastAPI server from src.inference.server.

Pipeline position: Step 11 of 11
Reads from:  Trained model checkpoint, FAISS index, KG
Config used: config.inference, config.serving, config.model
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


# ===========================================================================
# Resource loading
# ===========================================================================


def _load_model(cfg: FRLMConfig) -> Any:
    """Load the trained FRLM model from the latest joint checkpoint."""
    from src.model.frlm import FRLMModel

    checkpoint_dir = cfg.paths.resolve("checkpoints_dir") / "joint"
    logger.info("Loading FRLM model from %s", checkpoint_dir)
    logger.info("Backbone: %s", cfg.model.backbone.name)
    logger.info("Device: %s, Dtype: %s", cfg.inference.device, cfg.inference.dtype)

    if checkpoint_dir.exists():
        model = FRLMModel.from_pretrained(str(checkpoint_dir), device=cfg.inference.device)
    else:
        logger.warning("Checkpoint not found — building from config")
        model = FRLMModel.from_config(cfg)

    model.eval()
    logger.info("Model loaded successfully")
    return model


def _load_tokenizer(cfg: FRLMConfig) -> Any:
    """Load the tokenizer matching the backbone."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded: %s (vocab=%d)", cfg.model.backbone.name, len(tokenizer))
    return tokenizer


def _load_faiss_index(cfg: FRLMConfig) -> Any:
    """Load the FAISS index for retrieval."""
    index_dir = cfg.paths.resolve("faiss_index_dir")
    level = cfg.faiss.hierarchical.default_level
    level_name = getattr(cfg.faiss.hierarchical, f"level_{level}")
    index_path = index_dir / f"index_level_{level}_{level_name}.faiss"

    if not index_path.exists():
        logger.warning("FAISS index not found at %s", index_path)
        return None

    import faiss

    index = faiss.read_index(str(index_path))
    if cfg.faiss.use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, cfg.faiss.gpu_id, index)
    index.nprobe = cfg.faiss.nprobe
    logger.info("FAISS index loaded: %s (%d vectors)", index_path, index.ntotal)
    return index


def _load_kg_client(cfg: FRLMConfig) -> Any:
    """Connect to the Neo4j KG."""
    from src.kg.neo4j_client import Neo4jClient

    try:
        client = Neo4jClient.from_config(cfg.neo4j)
        logger.info("KG client connected: %s", cfg.neo4j.uri)
        return client
    except Exception as e:
        logger.warning("KG client connection failed: %s", e)
        return None


# ===========================================================================
# Batch inference
# ===========================================================================


def run_batch(cfg: FRLMConfig, input_file: Optional[str] = None) -> None:
    """Run batch inference on input texts using InferencePipeline."""
    from src.inference.pipeline import InferencePipeline

    logger.info("=== Batch Inference ===")

    model = _load_model(cfg)
    tokenizer = _load_tokenizer(cfg)
    faiss_index = _load_faiss_index(cfg)
    kg_client = _load_kg_client(cfg)

    pipeline = InferencePipeline(
        model=model,
        tokenizer=tokenizer,
        faiss_index=faiss_index,
        kg_client=kg_client,
        config=cfg.inference,
        device=cfg.inference.device,
    )

    # Load inputs
    inputs: List[str] = []
    if input_file:
        input_path = Path(input_file)
        if input_path.exists():
            with open(input_path, "r", encoding="utf-8") as fh:
                if input_path.suffix == ".json":
                    data = json.load(fh)
                    inputs = data if isinstance(data, list) else [data.get("text", str(data))]
                else:
                    inputs = [line.strip() for line in fh if line.strip()]
        logger.info("Loaded %d inputs from %s", len(inputs), input_file)
    else:
        inputs = [
            "What is the mechanism of action of pembrolizumab in non-small cell lung cancer?",
            "Describe the drug interactions between tamoxifen and CYP2D6 inhibitors.",
            "What are the current biomarkers for predicting response to immunotherapy?",
        ]
        logger.info("Using %d demo inputs", len(inputs))

    # Run pipeline
    responses = pipeline.generate_batch(inputs)

    # Serialize results
    results = [resp.to_dict() for resp in responses]

    output_path = cfg.paths.resolve("export_dir") / "inference_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    logger.info("Batch inference complete: %d results saved to %s", len(results), output_path)

    # Print summary
    for resp in responses:
        logger.info(
            "  [%d tok, %.1fms, ret=%.0f%%] %s... → %s...",
            resp.num_tokens_generated,
            resp.inference_time_ms,
            resp.retrieval_fraction * 100,
            resp.prompt[:40],
            resp.generated_text[:60],
        )


# ===========================================================================
# Server mode
# ===========================================================================


def run_server(cfg: FRLMConfig) -> None:
    """Start the FastAPI inference server."""
    from src.inference.pipeline import InferencePipeline
    from src.inference.server import create_app

    serving_cfg = cfg.serving
    logger.info("=== Starting Inference Server ===")
    logger.info(
        "Host: %s, Port: %d, Workers: %d",
        serving_cfg.host,
        serving_cfg.port,
        serving_cfg.workers,
    )

    # Load resources
    model = _load_model(cfg)
    tokenizer = _load_tokenizer(cfg)
    faiss_index = _load_faiss_index(cfg)
    kg_client = _load_kg_client(cfg)

    pipeline = InferencePipeline(
        model=model,
        tokenizer=tokenizer,
        faiss_index=faiss_index,
        kg_client=kg_client,
        config=cfg.inference,
        device=cfg.inference.device,
    )

    if serving_cfg.model_warmup:
        pipeline.warmup()

    app = create_app(pipeline=pipeline, kg_client=kg_client, config=cfg)

    import uvicorn

    uvicorn.run(
        app,
        host=serving_cfg.host,
        port=serving_cfg.port,
        workers=serving_cfg.workers,
        log_level=serving_cfg.log_level,
        reload=serving_cfg.reload,
    )


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    """Parse arguments, load config, and run inference."""
    parser = argparse.ArgumentParser(
        description="Run FRLM inference in batch or serve mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "serve"],
        default="batch",
        help="Inference mode: batch processing or HTTP server.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file for batch mode (JSON or text, one per line).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    logger.info("Starting 11_run_inference (mode=%s) with config: %s", args.mode, args.config)

    try:
        if args.mode == "serve":
            run_server(cfg)
        else:
            run_batch(cfg, input_file=args.input)
    except KeyboardInterrupt:
        logger.warning("Inference interrupted.")
        sys.exit(130)
    except Exception:
        logger.exception("Inference failed.")
        sys.exit(1)

    logger.info("11_run_inference completed successfully.")


if __name__ == "__main__":
    main()