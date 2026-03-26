#!/usr/bin/env python3
"""
09_train_joint.py — Phase 3: Joint fine-tuning with combined loss.

Jointly trains all components (backbone, router, retrieval, generation heads)
using the combined loss: L = 1.0×L_router + 2.0×L_retrieval + 1.0×L_generation.
Integrates with DeepSpeed ZeRO Stage 2 for efficient multi-GPU training.

Pipeline position: Step 9 of 11
Reads from:  Phase 1 & 2 checkpoints, training data
Writes to:   config.training.output_dir / phase3_joint (checkpoints)
Config used: config.training.joint, config.loss, config.deepspeed, config.wandb

Usage
-----
    python scripts/09_train_joint.py --config config/default.yaml
    python scripts/09_train_joint.py --phase1-ckpt ... --phase2-ckpt ...

    # DeepSpeed multi-GPU:
    deepspeed scripts/09_train_joint.py --config config/default.yaml
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
from src.training.dataset import JointDataset
from src.training.joint_trainer import JointTrainer
from src.training.utils import resolve_device

logger = logging.getLogger(__name__)


def _find_latest_checkpoint(base_dir: Path, prefix: str) -> Optional[Path]:
    """Auto-discover the latest checkpoint in a phase directory."""
    if not base_dir.exists():
        return None
    dirs = sorted(
        (d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)),
        key=lambda p: p.stat().st_mtime,
    )
    return dirs[-1] if dirs else None


def train_joint(
    cfg: FRLMConfig,
    phase1_checkpoint: str | None = None,
    phase2_checkpoint: str | None = None,
    resume_path: str | None = None,
    no_deepspeed: bool = False,
    gpu_id: int | None = None,
) -> Dict[str, float]:
    """Execute Phase 3: Joint fine-tuning.

    Parameters
    ----------
    cfg : FRLMConfig
        Loaded configuration.
    phase1_checkpoint : str, optional
        Phase 1 router checkpoint dir. Auto-discovered if ``None``.
    phase2_checkpoint : str, optional
        Phase 2 retrieval checkpoint dir. Auto-discovered if ``None``.
    resume_path : str, optional
        Phase 3 checkpoint to resume from.
    no_deepspeed : bool
        Force disable DeepSpeed even if config says enabled.
    gpu_id : int, optional
        CUDA device ordinal override.  When ``None`` the value from
        ``cfg.training.gpu_id`` is used.

    Returns
    -------
    dict
        Best validation metrics.
    """
    joint_cfg = cfg.training.joint
    loss_cfg = cfg.loss

    logger.info("=" * 60)
    logger.info("Phase 3: Joint Fine-Tuning")
    logger.info("=" * 60)
    logger.info("  Epochs:       %d", joint_cfg.epochs)
    logger.info("  Batch size:   %d", joint_cfg.batch_size)
    logger.info("  LR:           %.2e", joint_cfg.learning_rate)
    logger.info("  Scheduler:    %s (cycles=%d)", joint_cfg.scheduler, joint_cfg.num_cycles)
    logger.info("  Warmup:       %.1f%%", joint_cfg.warmup_ratio * 100)
    logger.info("  Loss weights: router=%.1f  retrieval=%.1f  generation=%.1f",
                loss_cfg.router_weight, loss_cfg.retrieval_weight, loss_cfg.generation_weight)
    logger.info("  Temperature:  %.3f", loss_cfg.contrastive_temperature)
    logger.info("  DeepSpeed:    %s", "disabled" if no_deepspeed else cfg.deepspeed.enabled)
    logger.info("  Early stop:   %s (patience=%d)", joint_cfg.early_stopping_metric, joint_cfg.early_stopping_patience)

    # --- Auto-discover prior phase checkpoints ---
    output_base = Path(cfg.training.output_dir)
    p1_ckpt = Path(phase1_checkpoint) if phase1_checkpoint else _find_latest_checkpoint(
        output_base / "phase1_router", "router",
    )
    p2_ckpt = Path(phase2_checkpoint) if phase2_checkpoint else _find_latest_checkpoint(
        output_base / "phase2_retrieval", "retrieval",
    )

    if p1_ckpt:
        logger.info("Phase 1 checkpoint: %s", p1_ckpt)
    else:
        logger.warning("No Phase 1 checkpoint found — starting from scratch")
    if p2_ckpt:
        logger.info("Phase 2 checkpoint: %s", p2_ckpt)
    else:
        logger.warning("No Phase 2 checkpoint found — starting from scratch")

    # --- Resolve GPU device ---
    effective_gpu = gpu_id if gpu_id is not None else cfg.training.gpu_id
    device = resolve_device(effective_gpu)
    logger.info("Building FRLM model...")
    model = FRLMModel.from_config(cfg)

    # --- Build datasets ---
    # Joint data lives in processed_dir/joint (written by 06b)
    data_dir = cfg.paths.resolve("processed_dir") / "joint"
    if not data_dir.exists() or not list(data_dir.glob("*.json*")):
        # Fallback to processed_dir for backward compatibility
        data_dir = cfg.paths.resolve("processed_dir")
    emb_dim = cfg.model.retrieval_head.semantic.output_dim
    num_neg = (
        cfg.faiss.hard_negatives.num_hard_negatives
        + cfg.faiss.hard_negatives.num_random_negatives
    )

    logger.info("Loading joint training data from %s", data_dir)
    full_ds = JointDataset(
        data_dir=data_dir,
        max_seq_length=cfg.model.backbone.max_seq_length,
        embedding_dim=emb_dim,
        num_negatives=num_neg,
    )

    if len(full_ds) == 0:
        logger.error("No joint training data found in %s.", data_dir)
        sys.exit(1)

    val_size = int(len(full_ds) * cfg.training.splits.validation)
    train_size = len(full_ds) - val_size
    gen = torch.Generator().manual_seed(cfg.training.seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size], generator=gen,
    )

    logger.info("Data split: train=%d, val=%d", len(train_ds), len(val_ds))

    # --- Create trainer ---
    use_ds = None if not no_deepspeed else False
    trainer = JointTrainer(
        model=model,
        config=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        use_deepspeed=use_ds,
    )

    # --- Resume ---
    if resume_path is not None:
        logger.info("Resuming from Phase 3 checkpoint: %s", resume_path)
        trainer.resume_from_checkpoint(resume_path)

    # --- Train ---
    t0 = time.time()
    best_metrics = trainer.train(
        phase1_checkpoint=p1_ckpt,
        phase2_checkpoint=p2_ckpt,
    )
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("Phase 3 Complete — %.1f s", elapsed)
    for k, v in best_metrics.items():
        logger.info("  best_%s = %.4f", k, v)
    logger.info("=" * 60)

    return best_metrics


def main() -> None:
    """Parse arguments, load config, and run joint training."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Joint fine-tuning with combined loss.",
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
        "--phase2-ckpt", type=str, default=None,
        help="Path to Phase 2 retrieval checkpoint directory.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to Phase 3 checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--no-deepspeed", action="store_true",
        help="Force disable DeepSpeed (use standard PyTorch).",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="CUDA device ordinal to train on (0, 1, …). "
             "Overrides training.gpu_id in the config. Use -1 for CPU.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="DeepSpeed local rank (set automatically by deepspeed launcher).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    logger.info("Starting 09_train_joint with config: %s", args.config)

    try:
        train_joint(
            cfg,
            phase1_checkpoint=args.phase1_ckpt,
            phase2_checkpoint=args.phase2_ckpt,
            resume_path=args.resume,
            no_deepspeed=args.no_deepspeed,
            gpu_id=args.gpu,
        )
    except KeyboardInterrupt:
        logger.warning("Joint training interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Joint training failed.")
        sys.exit(1)

    logger.info("09_train_joint completed successfully.")


if __name__ == "__main__":
    main()