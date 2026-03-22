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


def _find_latest_training_checkpoint(cfg: FRLMConfig) -> Optional[Path]:
    """Find the latest model.pt from training checkpoints (phase3 > phase2 > phase1)."""
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


def _load_model(cfg: FRLMConfig, checkpoint: Optional[str] = None) -> Any:
    """Load the trained FRLM model.

    Tries in order:
    1. Explicit --checkpoint path (save_pretrained format)
    2. Latest training checkpoint (model.pt state dict)
    3. Build from config with random weights (fallback)
    """
    import torch
    from src.model.frlm import FRLMModel

    # Resolve device: fall back to CPU when CUDA is requested but unavailable
    device = cfg.inference.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU for evaluation")
        device = "cpu"

    # 1. Explicit checkpoint in save_pretrained format
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists() and (ckpt_path / "config.json").exists():
            logger.info("Loading model from save_pretrained: %s", ckpt_path)
            model = FRLMModel.from_pretrained(str(ckpt_path), device=device)
            model.eval()
            return model

    # 2. Find latest training checkpoint (model.pt)
    model_pt = _find_latest_training_checkpoint(cfg)
    if model_pt is not None:
        logger.info("Loading trained weights from %s", model_pt)
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
                    "config/test.yaml (GPT-2), re-run evaluation with "
                    "--config config/test.yaml.",
                    model_pt, cfg.model.backbone.name,
                )
            raise
        logger.info("Trained weights loaded successfully from %s", model_pt.parent.name)
        model.eval()
        return model

    # 3. Fallback: random weights
    logger.warning("No trained checkpoint found — building model with random weights")
    model = FRLMModel.from_config(cfg)
    model.eval()
    return model


def _build_test_loader(cfg: FRLMConfig) -> Any:
    """Build a test DataLoader from available tokenized data.

    Uses the validation split of the RouterDataset (same seed as training)
    so evaluation is on held-out data.  Adds ``labels`` (= input_ids with
    padding masked to -100) needed by the generation evaluator.
    """
    import torch
    from torch.utils.data import DataLoader, random_split

    labels_dir = cfg.paths.resolve("labels_dir")
    tokenized_dir = labels_dir / "tokenized"

    if not tokenized_dir.exists() or not any(tokenized_dir.iterdir()):
        logger.warning("No tokenized data found at %s — test loader unavailable", tokenized_dir)
        return None

    try:
        from src.training.dataset import RouterDataset

        max_seq = cfg.model.backbone.max_seq_length
        full_ds = RouterDataset(data_dir=tokenized_dir, max_seq_length=max_seq)

        if len(full_ds) == 0:
            logger.warning("RouterDataset is empty — no test data")
            return None

        # Use same seed/split as training to get the validation portion
        val_frac = cfg.training.splits.validation
        val_size = int(len(full_ds) * val_frac)
        train_size = len(full_ds) - val_size
        gen = torch.Generator().manual_seed(cfg.training.seed)
        _, val_ds = random_split(full_ds, [train_size, val_size], generator=gen)

        logger.info("Test dataloader: %d examples (val split from %d total)", len(val_ds), len(full_ds))

        def _collate_with_labels(batch):
            """Stack tensors and derive LM labels from input_ids."""
            keys = batch[0].keys()
            collated = {k: torch.stack([b[k] for b in batch]) for k in keys}
            ids = collated["input_ids"].clone()
            mask = collated["attention_mask"]
            ids[mask == 0] = -100
            collated["labels"] = ids
            return collated

        test_dl = DataLoader(
            val_ds, batch_size=8, shuffle=False, num_workers=0,
            collate_fn=_collate_with_labels,
        )
        return test_dl

    except Exception as exc:
        logger.warning("Failed to build test loader: %s", exc)
        return None


def _load_faiss_index(cfg: FRLMConfig) -> Any:
    """Load the FAISS index."""
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
    logger.info("FAISS index loaded from %s (%d vectors)", faiss_path, fact_index.ntotal)
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

    # Build test dataloader from available training data
    test_loader = _build_test_loader(cfg)

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