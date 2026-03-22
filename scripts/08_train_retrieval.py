#!/usr/bin/env python3
"""
08_train_retrieval.py — Phase 2: Retrieval head training with InfoNCE loss.

Trains the retrieval head (semantic + granularity + temporal sub-heads)
using contrastive InfoNCE loss with hard negatives mined from FAISS.
Router is frozen during this phase; backbone may be unfrozen.

Pipeline position: Step 8 of 11
Reads from:  Phase 1 checkpoint, processed retrieval data
Writes to:   config.training.output_dir / phase2_retrieval (checkpoints)
Config used: config.training.retrieval, config.faiss, config.loss, config.wandb

Usage
-----
    python scripts/08_train_retrieval.py --config config/default.yaml
    python scripts/08_train_retrieval.py --phase1-ckpt checkpoints/phase1_router/router_step_0001000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FRLMConfig, load_config, setup_logging
from src.model.frlm import FRLMModel
from src.training.dataset import RetrievalDataset
from src.training.retrieval_trainer import RetrievalTrainer

logger = logging.getLogger(__name__)


def _find_latest_phase1_checkpoint(cfg: FRLMConfig) -> Optional[Path]:
    """Auto-discover the latest Phase 1 checkpoint."""
    phase1_dir = Path(cfg.training.output_dir) / "phase1_router"
    if not phase1_dir.exists():
        return None
    dirs = sorted(
        (d for d in phase1_dir.iterdir() if d.is_dir() and d.name.startswith("router")),
        key=lambda p: p.stat().st_mtime,
    )
    return dirs[-1] if dirs else None


def train_retrieval(
    cfg: FRLMConfig,
    phase1_checkpoint: str | None = None,
    resume_path: str | None = None,
) -> Dict[str, float]:
    """Execute Phase 2: Retrieval head training.

    Parameters
    ----------
    cfg : FRLMConfig
        Loaded configuration.
    phase1_checkpoint : str, optional
        Path to Phase 1 checkpoint. Auto-discovered if ``None``.
    resume_path : str, optional
        Path to Phase 2 checkpoint to resume from.

    Returns
    -------
    dict
        Best validation metrics.
    """
    retrieval_cfg = cfg.training.retrieval
    faiss_cfg = cfg.faiss

    logger.info("=" * 60)
    logger.info("Phase 2: Retrieval Head Training")
    logger.info("=" * 60)
    logger.info("  Epochs:      %d", retrieval_cfg.epochs)
    logger.info("  Batch size:  %d", retrieval_cfg.batch_size)
    logger.info("  LR:          %.2e", retrieval_cfg.learning_rate)
    logger.info("  Scheduler:   %s (warmup=%.1f%%)", retrieval_cfg.scheduler, retrieval_cfg.warmup_ratio * 100)
    logger.info("  Temperature: %.3f", retrieval_cfg.contrastive_temperature)
    logger.info("  Hard neg:    %d + %d random",
                faiss_cfg.hard_negatives.num_hard_negatives,
                faiss_cfg.hard_negatives.num_random_negatives)
    logger.info("  Backbone frozen: %s", retrieval_cfg.freeze_backbone)
    logger.info("  Router frozen:   %s", retrieval_cfg.freeze_router)
    logger.info("  Early stop:  %s (patience=%d)", retrieval_cfg.early_stopping_metric, retrieval_cfg.early_stopping_patience)

    # --- Resolve Phase 1 checkpoint ---
    p1_ckpt = Path(phase1_checkpoint) if phase1_checkpoint else _find_latest_phase1_checkpoint(cfg)
    if p1_ckpt is not None:
        logger.info("Phase 1 checkpoint: %s", p1_ckpt)
    else:
        logger.warning("No Phase 1 checkpoint found — training from scratch")

    # --- Build model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Building FRLM model...")
    model = FRLMModel.from_config(cfg)

    # --- Build datasets ---
    # Retrieval data lives in processed_dir/retrieval (written by 06b)
    data_dir = cfg.paths.resolve("processed_dir") / "retrieval"
    if not data_dir.exists() or not list(data_dir.glob("*.json*")):
        # Fallback to processed_dir for backward compatibility
        data_dir = cfg.paths.resolve("processed_dir")
    emb_dim = cfg.model.retrieval_head.semantic.output_dim
    num_neg = (
        faiss_cfg.hard_negatives.num_hard_negatives
        + faiss_cfg.hard_negatives.num_random_negatives
    )

    logger.info("Loading retrieval data from %s", data_dir)
    full_ds = RetrievalDataset(
        data_dir=data_dir,
        max_seq_length=cfg.model.backbone.max_seq_length,
        embedding_dim=emb_dim,
        num_negatives=num_neg,
    )

    if len(full_ds) == 0:
        logger.error("No retrieval data found in %s. Run steps 05-06 first.", data_dir)
        sys.exit(1)

    val_size = int(len(full_ds) * cfg.training.splits.validation)
    train_size = len(full_ds) - val_size
    gen = torch.Generator().manual_seed(cfg.training.seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size], generator=gen,
    )

    logger.info("Data split: train=%d, val=%d", len(train_ds), len(val_ds))

    # --- Create trainer ---
    trainer = RetrievalTrainer(
        model=model,
        config=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
    )

    # --- Resume ---
    if resume_path is not None:
        logger.info("Resuming from Phase 2 checkpoint: %s", resume_path)
        trainer.resume_from_checkpoint(resume_path)

    # --- Train ---
    t0 = time.time()
    best_metrics = trainer.train(phase1_checkpoint=p1_ckpt)
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("Phase 2 Complete — %.1f s", elapsed)
    for k, v in best_metrics.items():
        logger.info("  best_%s = %.4f", k, v)
    logger.info("=" * 60)

    return best_metrics


def main() -> None:
    """Parse arguments, load config, and train the retrieval head."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Train retrieval head with InfoNCE contrastive loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--phase1-ckpt", type=str, default=None,
        help="Path to Phase 1 router checkpoint directory.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to Phase 2 checkpoint directory to resume from.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    logger.info("Starting 08_train_retrieval with config: %s", args.config)

    try:
        train_retrieval(cfg, phase1_checkpoint=args.phase1_ckpt, resume_path=args.resume)
    except KeyboardInterrupt:
        logger.warning("Retrieval training interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Retrieval training failed.")
        sys.exit(1)

    logger.info("08_train_retrieval completed successfully.")


if __name__ == "__main__":
    main()