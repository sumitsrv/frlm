#!/usr/bin/env python3
"""
11_run_inference.py - Run inference in batch or serve mode.

In batch mode: processes input texts and generates outputs.
In serve mode: starts a FastAPI server for real-time inference.

Pipeline position: Step 11 of 11
Reads from:  Trained model checkpoint, FAISS index
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


def _load_model(cfg: FRLMConfig) -> Any:
    """Load the trained FRLM model from the latest joint checkpoint."""
    checkpoint_dir = cfg.paths.resolve("checkpoints_dir") / "joint"

    logger.info("Loading FRLM model from %s", checkpoint_dir)
    logger.info("Backbone: %s", cfg.model.backbone.name)
    logger.info("Device: %s, Dtype: %s", cfg.inference.device, cfg.inference.dtype)

    # Production:
    #   from src.model.frlm import FRLMModel
    #   model = FRLMModel(cfg)
    #   checkpoint = torch.load(latest_ckpt, map_location=cfg.inference.device)
    #   model.load_state_dict(checkpoint["model_state_dict"])
    #   model.eval()
    #   return model

    logger.info("Model loaded (stub mode)")
    return None


def _load_faiss_index(cfg: FRLMConfig) -> Any:
    """Load the FAISS index for retrieval."""
    index_dir = cfg.paths.resolve("faiss_index_dir")
    level = cfg.faiss.hierarchical.default_level
    level_name = getattr(cfg.faiss.hierarchical, f"level_{level}")
    index_path = index_dir / f"index_level_{level}_{level_name}.faiss"

    logger.info("Loading FAISS index: %s (level %d: %s)", index_path, level, level_name)

    # Production:
    #   import faiss
    #   index = faiss.read_index(str(index_path))
    #   if cfg.faiss.use_gpu:
    #       res = faiss.StandardGpuResources()
    #       index = faiss.index_cpu_to_gpu(res, cfg.faiss.gpu_id, index)
    #   index.nprobe = cfg.faiss.nprobe
    #   return index

    logger.info("FAISS index loaded (stub mode)")
    return None


def _run_batch_inference(
    model: Any,
    faiss_index: Any,
    inputs: List[str],
    cfg: FRLMConfig,
) -> List[Dict[str, Any]]:
    """Run inference on a batch of inputs.

    For each input:
    1. Tokenize and encode through backbone.
    2. Router decides retrieval vs. generation per token.
    3. Retrieval tokens: query FAISS, fetch facts from KG.
    4. Generation tokens: standard next-token prediction.
    5. Assemble final output.
    """
    results: List[Dict[str, Any]] = []
    inf_cfg = cfg.inference

    for text in inputs:
        start = time.time()

        # Production:
        #   tokens = tokenizer(text, return_tensors="pt").to(inf_cfg.device)
        #   with torch.no_grad():
        #       hidden = backbone(tokens)
        #       router_probs = torch.sigmoid(router_head(hidden))
        #       retrieval_mask = router_probs > inf_cfg.router_threshold
        #       if retrieval_mask.any():
        #           query_emb = retrieval_head.semantic(hidden[retrieval_mask])
        #           D, I = faiss_index.search(query_emb.cpu().numpy(), cfg.faiss.search_k)
        #           facts = fetch_facts_from_kg(I)
        #       gen_logits = generation_head(hidden)
        #       output = assemble_output(gen_logits, facts, router_probs)

        result = {
            "input": text,
            "output": f"[FRLM output for: {text[:50]}...]",
            "router_decisions": {
                "retrieval_fraction": 0.35,
                "generation_fraction": 0.65,
            },
            "retrieved_facts": [],
            "inference_time_ms": round((time.time() - start) * 1000, 2),
        }
        results.append(result)
        logger.debug("Processed: %s... (%.1fms)", text[:30], result["inference_time_ms"])

    return results


def run_batch(cfg: FRLMConfig, input_file: Optional[str] = None) -> None:
    """Run batch inference on input texts."""
    logger.info("=== Batch Inference ===")

    model = _load_model(cfg)
    faiss_index = _load_faiss_index(cfg)

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

    results = _run_batch_inference(model, faiss_index, inputs, cfg)

    output_path = cfg.paths.resolve("export_dir") / "inference_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    logger.info("Batch inference complete: %d results saved to %s", len(results), output_path)


def run_server(cfg: FRLMConfig) -> None:
    """Start the FastAPI inference server."""
    serving_cfg = cfg.serving

    logger.info("=== Starting Inference Server ===")
    logger.info("Host: %s, Port: %d, Workers: %d",
                serving_cfg.host, serving_cfg.port, serving_cfg.workers)

    # Production:
    #   from src.inference.server import create_app
    #   app = create_app(cfg)
    #   import uvicorn
    #   uvicorn.run(
    #       app,
    #       host=serving_cfg.host,
    #       port=serving_cfg.port,
    #       workers=serving_cfg.workers,
    #       log_level=serving_cfg.log_level,
    #       reload=serving_cfg.reload,
    #   )

    logger.info("Server would start at http://%s:%d (stub mode)", serving_cfg.host, serving_cfg.port)
    logger.info("CORS origins: %s", serving_cfg.cors_origins)
    logger.info("Max concurrent requests: %d", serving_cfg.max_concurrent_requests)
    logger.info("Request timeout: %ds", serving_cfg.request_timeout)

    if serving_cfg.model_warmup:
        logger.info("Running model warmup inference...")

    try:
        import uvicorn

        # In production, would run the actual FastAPI app
        logger.info("uvicorn available — server would start normally")
        logger.info("Press Ctrl+C to stop (stub mode, not actually serving)")

    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn[standard]")
        sys.exit(1)


def main() -> None:
    """Parse arguments, load config, and run inference."""
    parser = argparse.ArgumentParser(
        description="Run FRLM inference in batch or serve mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--mode", type=str, choices=["batch", "serve"], default="batch",
                        help="Inference mode: batch processing or HTTP server.")
    parser.add_argument("--input", type=str, default=None,
                        help="Input file for batch mode (JSON or text, one per line).")
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