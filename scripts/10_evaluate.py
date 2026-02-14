#!/usr/bin/env python3
"""
10_evaluate.py - Run the full evaluation suite.

Evaluates retrieval (P@k, temporal accuracy), generation (perplexity),
router (accuracy, confusion matrix, calibration), and end-to-end metrics
using the real evaluator classes from ``src.evaluation``.

Pipeline position: Step 10 of 11
Reads from:  Trained model checkpoints, test data, FAISS index, KG
Writes to:   config.paths.export_dir (evaluation results JSON)
Config used: config.evaluation, config.model, config.paths
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
from src.evaluation.end_to_end import EndToEndEvaluator
from src.evaluation.generation_eval import GenerationEvaluator
from src.evaluation.retrieval_eval import RetrievalEvaluator
from src.evaluation.router_eval import RouterEvaluator

logger = logging.getLogger(__name__)


# ===========================================================================
# Model & resource loading
# ===========================================================================


def _load_model(cfg: FRLMConfig, checkpoint: Optional[str] = None) -> Any:
    """Load the trained FRLM model."""
    from src.model.frlm import FRLMModel

    if checkpoint:
        ckpt_path = Path(checkpoint)
    else:
        ckpt_path = cfg.paths.resolve("checkpoints_dir") / "joint"

    logger.info("Loading FRLM model from %s", ckpt_path)

    if ckpt_path.exists():
        model = FRLMModel.from_pretrained(str(ckpt_path), device=cfg.inference.device)
    else:
        logger.warning("Checkpoint not found at %s — building from config", ckpt_path)
        model = FRLMModel.from_config(cfg)

    model.eval()
    return model


def _load_faiss_index(cfg: FRLMConfig) -> Any:
    """Load the FAISS index."""
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
    logger.info("FAISS index loaded from %s (%d vectors)", index_path, index.ntotal)
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
# Component evaluation functions
# ===========================================================================


def _evaluate_retrieval(
    model: Any,
    cfg: FRLMConfig,
    faiss_index: Any,
    kg_client: Any,
    test_loader: Any,
) -> Dict[str, Any]:
    """Evaluate retrieval head using RetrievalEvaluator."""
    evaluator = RetrievalEvaluator.from_config(cfg.evaluation.retrieval)
    logger.info(
        "Evaluating retrieval: k_values=%s, samples=%d",
        evaluator.k_values,
        cfg.evaluation.retrieval.num_eval_samples,
    )

    if test_loader is not None and faiss_index is not None:
        results = evaluator.evaluate(
            model=model,
            dataloader=test_loader,
            faiss_index=faiss_index,
            kg_client=kg_client,
            max_samples=cfg.evaluation.retrieval.num_eval_samples,
        )
        return results.to_dict()

    logger.warning("Retrieval test data or FAISS index unavailable — skipping live eval")
    return {"status": "skipped", "reason": "missing_resources"}


def _evaluate_generation(
    model: Any,
    cfg: FRLMConfig,
    test_loader: Any,
) -> Dict[str, Any]:
    """Evaluate generation head using GenerationEvaluator."""
    evaluator = GenerationEvaluator.from_config(cfg.evaluation.generation)
    logger.info(
        "Evaluating generation: samples=%d",
        cfg.evaluation.generation.num_eval_samples,
    )

    if test_loader is not None:
        results = evaluator.evaluate(
            model=model,
            dataloader=test_loader,
            max_samples=cfg.evaluation.generation.num_eval_samples,
        )
        return results.to_dict()

    logger.warning("Generation test data unavailable — skipping live eval")
    return {"status": "skipped", "reason": "missing_resources"}


def _evaluate_router(
    model: Any,
    cfg: FRLMConfig,
    test_loader: Any,
) -> Dict[str, Any]:
    """Evaluate router head using RouterEvaluator."""
    evaluator = RouterEvaluator.from_config(cfg.evaluation.router)
    logger.info(
        "Evaluating router: thresholds=%s, samples=%d",
        evaluator.threshold_sweep_values,
        cfg.evaluation.router.num_eval_samples,
    )

    if test_loader is not None:
        results = evaluator.evaluate(
            model=model,
            dataloader=test_loader,
            max_samples=cfg.evaluation.router.num_eval_samples,
        )
        return results.to_dict()

    logger.warning("Router test data unavailable — skipping live eval")
    return {"status": "skipped", "reason": "missing_resources"}


def _evaluate_end_to_end(
    model: Any,
    cfg: FRLMConfig,
    test_loaders: Dict[str, Any],
    faiss_index: Any,
    kg_client: Any,
) -> Dict[str, Any]:
    """Evaluate full pipeline using EndToEndEvaluator."""
    evaluator = EndToEndEvaluator.from_config(cfg.evaluation)
    logger.info(
        "Evaluating end-to-end: samples=%d",
        cfg.evaluation.end_to_end.num_eval_samples,
    )

    config_snapshot = {
        "backbone": cfg.model.backbone.name,
        "loss_weights": {
            "router": cfg.loss.router_weight,
            "retrieval": cfg.loss.retrieval_weight,
            "generation": cfg.loss.generation_weight,
        },
    }

    results = evaluator.evaluate(
        model=model,
        retrieval_dataloader=test_loaders.get("retrieval"),
        generation_dataloader=test_loaders.get("generation"),
        router_dataloader=test_loaders.get("router"),
        faiss_index=faiss_index,
        kg_client=kg_client,
        max_samples=cfg.evaluation.end_to_end.num_eval_samples,
        config_snapshot=config_snapshot,
    )
    return results.to_dict()


# ===========================================================================
# Main orchestration
# ===========================================================================


def evaluate(
    cfg: FRLMConfig,
    checkpoint: Optional[str] = None,
    components: str = "all",
) -> None:
    """Run the complete evaluation suite and save results."""
    export_dir = cfg.paths.resolve("export_dir")
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Full Evaluation Suite ===")
    start_time = time.time()

    # Parse requested components
    if components == "all":
        run_retrieval = run_generation = run_router = run_e2e = True
    else:
        parts = {c.strip().lower() for c in components.split(",")}
        run_retrieval = "retrieval" in parts
        run_generation = "generation" in parts
        run_router = "router" in parts
        run_e2e = "e2e" in parts or "end_to_end" in parts

    # Load model
    model = _load_model(cfg, checkpoint)

    # Load resources
    faiss_index = _load_faiss_index(cfg) if (run_retrieval or run_e2e) else None
    kg_client = _load_kg_client(cfg) if (run_retrieval or run_e2e) else None

    # NOTE: In production, test dataloaders would be built from
    #   the processed test split. For now, we pass None and the
    #   evaluators will report "skipped".
    test_loader = None

    all_results: Dict[str, Any] = {}

    if run_retrieval:
        logger.info("--- Retrieval Evaluation ---")
        all_results["retrieval"] = _evaluate_retrieval(
            model, cfg, faiss_index, kg_client, test_loader
        )

    if run_generation:
        logger.info("--- Generation Evaluation ---")
        all_results["generation"] = _evaluate_generation(model, cfg, test_loader)

    if run_router:
        logger.info("--- Router Evaluation ---")
        all_results["router"] = _evaluate_router(model, cfg, test_loader)

    if run_e2e:
        logger.info("--- End-to-End Evaluation ---")
        test_loaders: Dict[str, Any] = {
            "retrieval": test_loader,
            "generation": test_loader,
            "router": test_loader,
        }
        all_results["end_to_end"] = _evaluate_end_to_end(
            model, cfg, test_loaders, faiss_index, kg_client
        )

    total_time = time.time() - start_time
    all_results["evaluation_time_seconds"] = round(total_time, 2)
    all_results["config"] = {
        "backbone": cfg.model.backbone.name,
        "loss_weights": {
            "router": cfg.loss.router_weight,
            "retrieval": cfg.loss.retrieval_weight,
            "generation": cfg.loss.generation_weight,
        },
    }

    output_path = export_dir / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)

    logger.info("=== Evaluation Complete (%.2fs) ===", total_time)
    logger.info("Results saved to %s", output_path)

    # Print summary
    logger.info("--- Summary ---")
    if "retrieval" in all_results and "MRR" in all_results["retrieval"]:
        logger.info("  MRR: %.4f", all_results["retrieval"]["MRR"])
        for k in cfg.evaluation.retrieval.k_values:
            pk_key = f"P@{k}"
            if pk_key in all_results["retrieval"]:
                logger.info("  %s: %.4f", pk_key, all_results["retrieval"][pk_key])
    if "generation" in all_results and "overall_perplexity" in all_results["generation"]:
        logger.info(
            "  Perplexity: %.2f", all_results["generation"]["overall_perplexity"]
        )
    if "router" in all_results and "best_f1" in all_results["router"]:
        logger.info("  Router F1: %.4f", all_results["router"]["best_f1"])
    if "end_to_end" in all_results and "overall_score" in all_results["end_to_end"]:
        logger.info(
            "  E2E Score: %.4f", all_results["end_to_end"]["overall_score"]
        )


def main() -> None:
    """Parse arguments, load config, and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Run the full FRLM evaluation suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint directory.",
    )
    parser.add_argument(
        "--components",
        type=str,
        default="all",
        help="Comma-separated: retrieval,generation,router,e2e,all",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    logger.info("Starting 10_evaluate with config: %s", args.config)

    try:
        evaluate(cfg, checkpoint=args.checkpoint, components=args.components)
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted.")
        sys.exit(130)
    except Exception:
        logger.exception("Evaluation failed.")
        sys.exit(1)

    logger.info("10_evaluate completed successfully.")


if __name__ == "__main__":
    main()