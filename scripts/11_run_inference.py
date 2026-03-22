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


def _find_latest_training_checkpoint(cfg: FRLMConfig) -> Optional[Path]:
    """Find the latest model.pt from training checkpoints."""
    try:
        ckpt_base = Path(cfg.training.output_dir)
    except Exception:
        ckpt_base = cfg.paths.resolve("checkpoints_dir")

    for phase_dir in ["phase3_joint", "phase2_retrieval", "phase1_router"]:
        phase_path = ckpt_base / phase_dir
        if phase_path.exists():
            ckpt_dirs = sorted([d for d in phase_path.iterdir() if d.is_dir()])
            if ckpt_dirs:
                model_pt = ckpt_dirs[-1] / "model.pt"
                if model_pt.exists():
                    return model_pt
    return None


def _load_model(cfg: FRLMConfig) -> Any:
    """Load the trained FRLM model from the latest checkpoint."""
    import torch
    from src.model.frlm import FRLMModel

    # Resolve device: fall back to CPU when CUDA is requested but unavailable
    device = cfg.inference.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU for inference")
        device = "cpu"

    # Try save_pretrained format first
    ckpt_dir = cfg.paths.resolve("checkpoints_dir") / "joint"
    if ckpt_dir.exists() and (ckpt_dir / "config.json").exists():
        logger.info("Loading model from save_pretrained: %s", ckpt_dir)
        model = FRLMModel.from_pretrained(str(ckpt_dir), device=device)
        model.eval()
        return model

    # Find latest training checkpoint
    model_pt = _find_latest_training_checkpoint(cfg)
    if model_pt is not None:
        logger.info("Loading trained weights from %s", model_pt)
        logger.info("Backbone: %s", cfg.model.backbone.name)
        logger.info("Device: %s, Dtype: %s", device, cfg.inference.dtype)
        model = FRLMModel.from_config(cfg)
        state_dict = torch.load(str(model_pt), map_location=device)
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                logger.error(
                    "Checkpoint shape mismatch — the checkpoint at %s was "
                    "likely trained with a different backbone/config than "
                    "the current one (%s). If you trained with "
                    "config/test.yaml (GPT-2), re-run with "
                    "--config config/test.yaml.",
                    model_pt, cfg.model.backbone.name,
                )
            raise
        logger.info("Trained weights loaded from %s", model_pt.parent.name)
        model.eval()
        return model

    # Fallback
    logger.warning("No checkpoint found — building model with random weights")
    logger.info("Backbone: %s", cfg.model.backbone.name)
    model = FRLMModel.from_config(cfg)
    model.eval()
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
    from src.embeddings.faiss_index import FAISSFactIndex

    index_dir = cfg.paths.resolve("faiss_index_dir")
    level = cfg.faiss.hierarchical.default_level
    level_name = getattr(cfg.faiss.hierarchical, f"level_{level}")
    index_base = index_dir / f"level_{level}_{level_name}"
    faiss_path = index_base.with_suffix(".faiss")

    if not faiss_path.exists():
        logger.warning("FAISS index not found at %s", faiss_path)
        return None

    fact_index = FAISSFactIndex(
        embedding_dim=cfg.faiss.embedding_dim,
        index_type=cfg.faiss.index_type,
        metric=cfg.faiss.metric,
        nprobe=cfg.faiss.nprobe,
        use_gpu=cfg.faiss.use_gpu,
        gpu_id=cfg.faiss.gpu_id,
    )
    fact_index.load_index(index_base)
    logger.info("FAISS index loaded: %s (%d vectors)", faiss_path, fact_index.ntotal)
    return fact_index


def _load_kg_client(cfg: FRLMConfig) -> Any:
    """Connect to the Neo4j KG."""
    from src.kg.neo4j_client import Neo4jClient

    try:
        client = Neo4jClient.from_config(cfg)
        client.connect()
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