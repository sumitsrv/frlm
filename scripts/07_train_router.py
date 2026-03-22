#!/usr/bin/env python3
"""
07_train_router.py — Phase 1: Router head pre-training.

Trains the binary router head (retrieval vs. generation classification)
using BCE loss with label smoothing. Backbone is frozen during this phase.
Includes early stopping, checkpointing, and WandB logging.

Pipeline position: Step 7 of 11
Reads from:  config.paths.labels_dir
Writes to:   config.training.output_dir / phase1_router (checkpoints)
Config used: config.training.router, config.model, config.loss, config.wandb

Usage
-----
    python scripts/07_train_router.py --config config/default.yaml
    python scripts/07_train_router.py --resume checkpoints/phase1_router/router_step_0001000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FRLMConfig, load_config, setup_logging
from src.model.frlm import FRLMModel
from src.training.dataset import RouterDataset
from src.training.router_trainer import RouterTrainer

logger = logging.getLogger(__name__)


def train_router(cfg: FRLMConfig, resume_path: str | None = None) -> Dict[str, float]:
    """Execute Phase 1: Router head pre-training.

    Parameters
    ----------
    cfg : FRLMConfig
        Loaded FRLM configuration.
    resume_path : str, optional
        Checkpoint directory to resume from.

    Returns
    -------
    dict
        Best validation metrics from training.
    """
    router_cfg = cfg.training.router

    logger.info("=" * 60)
    logger.info("Phase 1: Router Pre-training")
    logger.info("=" * 60)
    logger.info("  Epochs:          %d", router_cfg.epochs)
    logger.info("  Batch size:      %d", router_cfg.batch_size)
    logger.info("  Learning rate:   %.2e", router_cfg.learning_rate)
    logger.info("  Scheduler:       %s (warmup=%.1f%%)", router_cfg.scheduler, router_cfg.warmup_ratio * 100)
    logger.info("  Label smoothing: %.2f", router_cfg.label_smoothing)
    logger.info("  Pos weight:      %.1f", router_cfg.pos_weight)
    logger.info("  Backbone frozen: %s", router_cfg.freeze_backbone)
    logger.info("  Early stop:      %s (patience=%d)", router_cfg.early_stopping_metric, router_cfg.early_stopping_patience)

    # --- Build model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Building FRLM model...")
    model = FRLMModel.from_config(cfg)

    # --- Build datasets ---
    labels_dir = cfg.paths.resolve("labels_dir")
    # Use tokenized subdirectory (produced by 06b_prepare_training_data)
    tokenized_dir = labels_dir / "tokenized"
    if not tokenized_dir.exists() or not list(tokenized_dir.glob("*.jsonl")):
        # Fall back to labels_dir for backward compatibility
        tokenized_dir = labels_dir
    max_seq = cfg.model.backbone.max_seq_length

    logger.info("Loading router labels from %s", tokenized_dir)
    full_ds = RouterDataset(data_dir=tokenized_dir, max_seq_length=max_seq)

    if len(full_ds) == 0:
        logger.error("No training data found in %s. Run step 06 first.", labels_dir)
        sys.exit(1)

    val_size = int(len(full_ds) * cfg.training.splits.validation)
    train_size = len(full_ds) - val_size
    gen = torch.Generator().manual_seed(cfg.training.seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size], generator=gen,
    )

    logger.info("Data split: train=%d, val=%d", len(train_ds), len(val_ds))

    # --- Create trainer ---
    trainer = RouterTrainer(
        model=model,
        config=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
    )

    # --- Resume if requested ---
    if resume_path is not None:
        logger.info("Resuming from checkpoint: %s", resume_path)
        trainer.resume_from_checkpoint(resume_path)

    # --- Train ---
    t0 = time.time()
    best_metrics = trainer.train()
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("Phase 1 Complete — %.1f s", elapsed)
    for k, v in best_metrics.items():
        logger.info("  best_%s = %.4f", k, v)
    logger.info("=" * 60)

    return best_metrics


def main() -> None:
    """Parse arguments, load config, and train the router."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Train router head (binary classification).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume training from.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    logger.info("Starting 07_train_router with config: %s", args.config)

    try:
        train_router(cfg, resume_path=args.resume)
    except KeyboardInterrupt:
        logger.warning("Router training interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Router training failed.")
        sys.exit(1)

    logger.info("07_train_router completed successfully.")


if __name__ == "__main__":
    main()