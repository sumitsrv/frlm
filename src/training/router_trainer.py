"""
Phase 1 — Router Pre-training.

Trains **only** the :class:`RouterHead` on token-level binary labels
(factual-retrieval vs linguistic-generation). The backbone is fully
frozen; only the small two-layer router MLP receives gradients.

Training loop
-------------
1. Load labelled chunks from ``data/labels/``
2. Forward through frozen backbone → router logits
3. Compute :class:`RouterLoss` (BCE with pos_weight + label smoothing)
4. Track accuracy, precision, recall, F1 per epoch
5. Early-stop on validation F1
6. Save best checkpoint via :class:`CheckpointManager`

W&B dashboard: ``router/phase1/…``
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from src.model.frlm import FRLMModel
from src.model.losses import RouterLoss
from src.training.dataset import RouterDataset
from src.training.utils import (
    CheckpointManager,
    EarlyStopping,
    GradientAccumulator,
    LearningRateScheduler,
    MetricsLogger,
    TrainingState,
    finish_wandb,
    init_wandb,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Metric helpers
# ===========================================================================


def _compute_router_metrics(
    all_logits: List[Tensor],
    all_labels: List[Tensor],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1 from collected logits/labels."""
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    tp = ((preds == 1) & (labels == 1)).float().sum().item()
    fp = ((preds == 1) & (labels == 0)).float().sum().item()
    fn = ((preds == 0) & (labels == 1)).float().sum().item()
    tn = ((preds == 0) & (labels == 0)).float().sum().item()

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / max(total, 1.0)
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ===========================================================================
# RouterTrainer
# ===========================================================================


class RouterTrainer:
    """Phase 1 trainer — router head only.

    Parameters
    ----------
    model : FRLMModel
        Full FRLM model (backbone + heads + loss).
    config : object
        ``FRLMConfig`` (or compatible) — reads ``training.router``,
        ``training.*``, ``wandb``, ``paths``, etc.
    train_dataset : RouterDataset, optional
        If ``None``, loads from ``config.paths.labels_dir``.
    val_dataset : RouterDataset, optional
        Validation split.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model: FRLMModel,
        config: Any,
        train_dataset: Optional[RouterDataset] = None,
        val_dataset: Optional[RouterDataset] = None,
        device: str = "cuda",
    ) -> None:
        self._model = model
        self._cfg = config
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Shortcut configs
        self._rcfg = config.training.router        # RouterTrainingConfig
        self._tcfg = config.training                # TrainingConfig
        self._wandb_cfg = config.wandb              # WandBConfig

        # Datasets
        self._train_ds = train_dataset
        self._val_ds = val_dataset

        # Will be initialised in ``train()``
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[LearningRateScheduler] = None
        self._loss_fn: Optional[RouterLoss] = None
        self._scaler: Optional[GradScaler] = None
        self._ckpt_mgr: Optional[CheckpointManager] = None
        self._grad_acc: Optional[GradientAccumulator] = None
        self._early_stop: Optional[EarlyStopping] = None
        self._train_logger: Optional[MetricsLogger] = None
        self._val_logger: Optional[MetricsLogger] = None
        self._state = TrainingState(best_metric_name=self._rcfg.early_stopping_metric)
        self._wandb_run: Any = None

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _build_datasets(self) -> Tuple[RouterDataset, RouterDataset]:
        """Load / split datasets."""
        if self._train_ds is not None and self._val_ds is not None:
            return self._train_ds, self._val_ds

        labels_dir = Path(self._cfg.paths.resolve("labels_dir"))
        full_ds = RouterDataset(
            data_dir=labels_dir,
            max_seq_length=self._cfg.model.backbone.max_seq_length,
        )

        val_size = int(len(full_ds) * self._tcfg.splits.validation)
        train_size = len(full_ds) - val_size
        generator = torch.Generator().manual_seed(self._tcfg.seed)
        train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
        return train_ds, val_ds  # type: ignore[return-value]

    def _build_dataloaders(
        self, train_ds: Any, val_ds: Any,
    ) -> Tuple[DataLoader, DataLoader]:
        train_dl = DataLoader(
            train_ds,
            batch_size=self._rcfg.batch_size,
            shuffle=True,
            num_workers=self._tcfg.dataloader_num_workers,
            pin_memory=self._tcfg.pin_memory,
            drop_last=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=self._rcfg.batch_size * 2,
            shuffle=False,
            num_workers=self._tcfg.dataloader_num_workers,
            pin_memory=self._tcfg.pin_memory,
        )
        return train_dl, val_dl

    # ------------------------------------------------------------------
    # Freeze / parameter groups
    # ------------------------------------------------------------------

    def _freeze_non_router(self) -> None:
        """Freeze everything except the router head."""
        # Backbone
        self._model.backbone.freeze()
        # Retrieval head
        for p in self._model.retrieval_head.parameters():
            p.requires_grad = False
        # Generation head
        for p in self._model.generation_head.parameters():
            p.requires_grad = False
        # Loss (no trainable params, but be safe)
        if self._model.loss_fn is not None:
            for p in self._model.loss_fn.parameters():
                p.requires_grad = False

        trainable = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Phase 1 freeze: %.2fM / %.2fM params trainable (router only)",
            trainable / 1e6, total / 1e6,
        )

    def _build_optimizer(self) -> torch.optim.AdamW:
        router_params = [
            p for p in self._model.router.parameters() if p.requires_grad
        ]
        return torch.optim.AdamW(
            router_params,
            lr=self._rcfg.learning_rate,
            weight_decay=self._rcfg.weight_decay,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """Run the full Phase 1 training loop.

        Returns a dict with final validation metrics.
        """
        t0 = time.time()

        # --- W&B ---
        self._wandb_run = init_wandb(
            project=self._wandb_cfg.project,
            run_name=f"phase1-router-{int(time.time())}",
            tags=self._wandb_cfg.tags + ["phase1", "router"],
            config={
                "phase": 1,
                "epochs": self._rcfg.epochs,
                "batch_size": self._rcfg.batch_size,
                "lr": self._rcfg.learning_rate,
                "scheduler": self._rcfg.scheduler,
                "warmup_ratio": self._rcfg.warmup_ratio,
                "label_smoothing": self._rcfg.label_smoothing,
                "pos_weight": self._rcfg.pos_weight,
                "freeze_backbone": self._rcfg.freeze_backbone,
            },
            enabled=self._wandb_cfg.enabled,
            entity=self._wandb_cfg.entity,
            api_key=self._wandb_cfg.api_key,
        )

        # --- Data ---
        train_ds, val_ds = self._build_datasets()
        train_dl, val_dl = self._build_dataloaders(train_ds, val_ds)

        # --- Model setup ---
        self._model.to(self._device)
        self._freeze_non_router()

        # --- Loss ---
        self._loss_fn = RouterLoss(
            pos_weight=self._rcfg.pos_weight,
            label_smoothing=self._rcfg.label_smoothing,
        )

        # --- Optimizer / scheduler ---
        self._optimizer = self._build_optimizer()
        total_steps = len(train_dl) * self._rcfg.epochs
        warmup_steps = int(total_steps * self._rcfg.warmup_ratio)
        self._scheduler = LearningRateScheduler(
            optimizer=self._optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            schedule_type=self._rcfg.scheduler,
        )

        # --- AMP ---
        use_amp = self._tcfg.fp16 and self._device.type == "cuda"
        self._scaler = GradScaler(enabled=use_amp)

        # --- Gradient accumulation ---
        self._grad_acc = GradientAccumulator(
            accumulation_steps=self._tcfg.gradient_accumulation_steps,
            max_grad_norm=self._tcfg.max_grad_norm,
        )

        # --- Checkpoint manager ---
        ckpt_dir = Path(self._tcfg.output_dir) / "phase1_router"
        self._ckpt_mgr = CheckpointManager(
            output_dir=ckpt_dir,
            max_checkpoints=self._tcfg.max_checkpoints,
            prefix="router",
        )

        # --- Early stopping ---
        es_mode = "max" if self._rcfg.early_stopping_metric != "loss" else "min"
        self._early_stop = EarlyStopping(
            patience=self._rcfg.early_stopping_patience,
            metric_name=self._rcfg.early_stopping_metric,
            mode=es_mode,
        )

        # --- Metrics loggers ---
        self._train_logger = MetricsLogger(
            wandb_run=self._wandb_run,
            log_frequency=self._tcfg.log_every_n_steps,
            prefix="router/phase1/train",
        )
        self._val_logger = MetricsLogger(
            wandb_run=self._wandb_run,
            log_frequency=1,
            prefix="router/phase1/val",
        )

        # --- Main loop ---
        best_metrics: Dict[str, float] = {}

        for epoch in range(self._rcfg.epochs):
            self._state.epoch = epoch

            # Train
            train_metrics = self._train_epoch(train_dl, use_amp)
            self._train_logger.log_epoch(train_metrics, epoch)

            # Validate
            val_metrics = self._evaluate(val_dl, use_amp)
            self._val_logger.log_epoch(val_metrics, epoch)

            # Checkpoint on improvement
            es_value = val_metrics.get(self._rcfg.early_stopping_metric, val_metrics.get("loss", 0.0))
            is_best = self._early_stop.best_value is None or (
                (es_mode == "max" and es_value > self._early_stop.best_value)
                or (es_mode == "min" and es_value < self._early_stop.best_value)
            )

            if is_best:
                self._state.best_metric = es_value
                best_metrics = val_metrics.copy()
                self._ckpt_mgr.save(
                    model=self._model,
                    optimizer=self._optimizer,
                    scheduler=self._scheduler,
                    state=self._state,
                    metrics=val_metrics,
                )

            # Early stopping
            if self._early_stop.step(es_value):
                logger.info("Early stopping at epoch %d", epoch)
                break

            logger.info(
                "[Phase 1] Epoch %d/%d — train_loss=%.4f  val_f1=%.4f  val_loss=%.4f",
                epoch + 1, self._rcfg.epochs,
                train_metrics.get("loss", 0.0),
                val_metrics.get("f1", 0.0),
                val_metrics.get("loss", 0.0),
            )

        # --- Finalise ---
        self._state.wall_time_seconds = time.time() - t0
        logger.info(
            "Phase 1 complete in %.1f s — best %s = %.4f",
            self._state.wall_time_seconds,
            self._rcfg.early_stopping_metric,
            self._state.best_metric,
        )
        finish_wandb(self._wandb_run)

        return best_metrics

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, dataloader: DataLoader, use_amp: bool) -> Dict[str, float]:
        """Run one training epoch. Returns aggregated metrics."""
        self._model.train()
        # Backbone stays in eval (frozen)
        self._model.backbone.eval()

        epoch_loss = 0.0
        all_logits: List[Tensor] = []
        all_labels: List[Tensor] = []
        n_batches = 0

        self._grad_acc.reset()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self._device)
            attention_mask = batch["attention_mask"].to(self._device)
            router_labels = batch["router_labels"].to(self._device)

            with autocast(enabled=use_amp):
                # Forward through frozen backbone
                with torch.no_grad():
                    backbone_out = self._model.backbone(input_ids, attention_mask)
                hidden = backbone_out.last_hidden_state

                # Router forward (trainable)
                router_logits = self._model.router(hidden).squeeze(-1)  # (B, seq)

                # Loss
                loss = self._loss_fn(router_logits, router_labels, mask=attention_mask)

            # Scale for accumulation
            scaled = self._grad_acc.scale_loss(loss)
            self._scaler.scale(scaled).backward()

            if self._grad_acc.should_step():
                self._grad_acc.step(
                    optimizer=self._optimizer,
                    model=self._model.router,
                    scheduler=self._scheduler,
                    scaler=self._scaler,
                )
                self._state.global_step += 1

            # Collect for epoch metrics
            epoch_loss += loss.item()
            n_batches += 1
            all_logits.append(router_logits.detach().cpu())
            all_labels.append(router_labels.detach().cpu())

            self._state.samples_seen += input_ids.size(0)

            # Step-level logging
            self._train_logger.log_step(
                {"loss": loss.item(), "lr": self._optimizer.param_groups[0]["lr"]},
                step=self._state.global_step,
            )

        # Flush any remaining metrics
        self._train_logger.flush(step=self._state.global_step)

        # Epoch metrics
        metrics = _compute_router_metrics(all_logits, all_labels)
        metrics["loss"] = epoch_loss / max(n_batches, 1)
        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader, use_amp: bool) -> Dict[str, float]:
        """Evaluate on validation set. Returns metrics dict."""
        self._model.eval()

        epoch_loss = 0.0
        all_logits: List[Tensor] = []
        all_labels: List[Tensor] = []
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self._device)
            attention_mask = batch["attention_mask"].to(self._device)
            router_labels = batch["router_labels"].to(self._device)

            with autocast(enabled=use_amp):
                backbone_out = self._model.backbone(input_ids, attention_mask)
                hidden = backbone_out.last_hidden_state
                router_logits = self._model.router(hidden).squeeze(-1)
                loss = self._loss_fn(router_logits, router_labels, mask=attention_mask)

            epoch_loss += loss.item()
            n_batches += 1
            all_logits.append(router_logits.cpu())
            all_labels.append(router_labels.cpu())

        metrics = _compute_router_metrics(all_logits, all_labels)
        metrics["loss"] = epoch_loss / max(n_batches, 1)
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------

    def resume_from_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        """Restore training state from a checkpoint."""
        if self._ckpt_mgr is None:
            ckpt_dir = Path(self._tcfg.output_dir) / "phase1_router"
            self._ckpt_mgr = CheckpointManager(output_dir=ckpt_dir)

        self._state = self._ckpt_mgr.load(
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            checkpoint_path=checkpoint_path,
            device=str(self._device),
        )
        logger.info("Resumed Phase 1 from step %d", self._state.global_step)

    # ------------------------------------------------------------------
    # Convenience: get best checkpoint path
    # ------------------------------------------------------------------

    @property
    def best_checkpoint_dir(self) -> Optional[Path]:
        """Return the directory of the best checkpoint."""
        if self._ckpt_mgr is None:
            return None
        return self._ckpt_mgr.latest()
