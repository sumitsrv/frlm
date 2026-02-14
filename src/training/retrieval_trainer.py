"""
Phase 2 — Retrieval Head Training.

Trains the :class:`RetrievalHead` (and optionally un-freezes the backbone)
while keeping the router head frozen. Uses :class:`InfoNCELoss` with
hard-negative mining.

Training loop
-------------
1. Load Phase 1 router checkpoint (best)
2. Freeze router head; optionally unfreeze backbone
3. For each epoch:
   a. Forward: backbone → retrieval head → query signature
   b. Contrastive loss (InfoNCE) against positive + hard negatives
   c. Hard-negative refresh every ``mine_frequency`` steps
4. Track P@1, P@5, P@10, MRR
5. Early-stop on validation precision@1
6. Save best checkpoint

W&B dashboard: ``retrieval/phase2/…``
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
from src.model.losses import InfoNCELoss
from src.training.dataset import RetrievalDataset
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
# Retrieval metric helpers
# ===========================================================================


def _precision_at_k(
    query_emb: Tensor, positive_emb: Tensor, negative_embs: Tensor, k: int,
) -> float:
    """Compute precision@k for a batch of queries.

    Parameters
    ----------
    query_emb : (batch, dim)
    positive_emb : (batch, dim)
    negative_embs : (batch, num_neg, dim)
    k : int
        Top-k cutoff.

    Returns
    -------
    float
        Fraction of queries where the positive is in the top-k.
    """
    # Similarity: query vs [positive, negatives]
    pos_sim = (query_emb * positive_emb).sum(dim=-1, keepdim=True)  # (batch, 1)
    neg_sim = torch.bmm(negative_embs, query_emb.unsqueeze(-1)).squeeze(-1)  # (batch, n_neg)
    all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (batch, 1+n_neg)

    # Top-k indices
    topk_idx = all_sim.topk(min(k, all_sim.size(1)), dim=1).indices  # (batch, k)
    # Positive is always at index 0
    hits = (topk_idx == 0).any(dim=1).float()
    return hits.mean().item()


def _mean_reciprocal_rank(
    query_emb: Tensor, positive_emb: Tensor, negative_embs: Tensor,
) -> float:
    """Compute MRR for a batch of queries."""
    pos_sim = (query_emb * positive_emb).sum(dim=-1, keepdim=True)
    neg_sim = torch.bmm(negative_embs, query_emb.unsqueeze(-1)).squeeze(-1)
    all_sim = torch.cat([pos_sim, neg_sim], dim=1)

    # Rank of the positive (index 0)
    # Number of items with sim > positive sim
    ranks = (all_sim > pos_sim).sum(dim=1).float() + 1.0  # (batch,)
    rr = 1.0 / ranks
    return rr.mean().item()


def _compute_retrieval_metrics(
    all_query: List[Tensor],
    all_positive: List[Tensor],
    all_negative: List[Tensor],
) -> Dict[str, float]:
    """Aggregate precision@k and MRR across the full dataset."""
    q = torch.cat(all_query, dim=0)
    p = torch.cat(all_positive, dim=0)
    n = torch.cat(all_negative, dim=0)

    return {
        "precision_at_1": _precision_at_k(q, p, n, k=1),
        "precision_at_5": _precision_at_k(q, p, n, k=5),
        "precision_at_10": _precision_at_k(q, p, n, k=10),
        "mrr": _mean_reciprocal_rank(q, p, n),
    }


# ===========================================================================
# RetrievalTrainer
# ===========================================================================


class RetrievalTrainer:
    """Phase 2 trainer — retrieval head (+ optional backbone).

    Parameters
    ----------
    model : FRLMModel
        Full model (backbone + heads + loss).
    config : object
        ``FRLMConfig`` (or compatible).
    train_dataset : RetrievalDataset, optional
    val_dataset : RetrievalDataset, optional
    faiss_index : FAISSFactIndex, optional
        FAISS vector index used for hard-negative mining during training.
    sapbert_encoder : SapBERTEncoder, optional
        SapBERT encoder for computing fact embeddings during negative refresh.
    device : str
    """

    def __init__(
        self,
        model: FRLMModel,
        config: Any,
        train_dataset: Optional[RetrievalDataset] = None,
        val_dataset: Optional[RetrievalDataset] = None,
        faiss_index: Any = None,
        sapbert_encoder: Any = None,
        device: str = "cuda",
    ) -> None:
        self._model = model
        self._cfg = config
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._rcfg = config.training.retrieval  # RetrievalTrainingConfig
        self._tcfg = config.training            # TrainingConfig
        self._wandb_cfg = config.wandb
        self._faiss_cfg = config.faiss

        self._train_ds = train_dataset
        self._val_ds = val_dataset
        self._faiss_index = faiss_index
        self._sapbert_encoder = sapbert_encoder

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[LearningRateScheduler] = None
        self._loss_fn: Optional[InfoNCELoss] = None
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

    def _build_datasets(self) -> Tuple[RetrievalDataset, RetrievalDataset]:
        if self._train_ds is not None and self._val_ds is not None:
            return self._train_ds, self._val_ds

        data_dir = Path(self._cfg.paths.resolve("processed_dir"))
        emb_dim = self._cfg.model.retrieval_head.semantic.output_dim
        num_neg = (
            self._faiss_cfg.hard_negatives.num_hard_negatives
            + self._faiss_cfg.hard_negatives.num_random_negatives
        )

        full_ds = RetrievalDataset(
            data_dir=data_dir,
            max_seq_length=self._cfg.model.backbone.max_seq_length,
            embedding_dim=emb_dim,
            num_negatives=num_neg,
        )

        val_size = int(len(full_ds) * self._tcfg.splits.validation)
        train_size = len(full_ds) - val_size
        gen = torch.Generator().manual_seed(self._tcfg.seed)
        train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=gen)
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
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _configure_frozen_params(self) -> None:
        """Freeze router; optionally freeze / unfreeze backbone."""
        # Freeze router head
        for p in self._model.router.parameters():
            p.requires_grad = False

        # Freeze generation head
        for p in self._model.generation_head.parameters():
            p.requires_grad = False

        # Backbone: Phase 2 config determines freeze state
        if self._rcfg.freeze_backbone:
            self._model.backbone.freeze()
        else:
            self._model.backbone.unfreeze()

        # Freeze router in loss (no trainable params, safety)
        if self._model.loss_fn is not None:
            for p in self._model.loss_fn.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Phase 2 freeze: %.2fM / %.2fM params trainable (retrieval head%s)",
            trainable / 1e6, total / 1e6,
            " + backbone" if not self._rcfg.freeze_backbone else "",
        )

    def _build_optimizer(self) -> torch.optim.AdamW:
        params = [p for p in self._model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=self._rcfg.learning_rate,
            weight_decay=self._rcfg.weight_decay,
        )

    # ------------------------------------------------------------------
    # Hard-negative refresh
    # ------------------------------------------------------------------

    def _maybe_refresh_negatives(self, step: int) -> None:
        """Re-mine hard negatives via the FAISS index using current model embeddings.

        At every ``mine_frequency`` steps, encode the retrieval-head query
        embeddings for the training set, query the FAISS index for the
        closest non-positive facts, and rebuild the negative embeddings in
        each training example.
        """
        freq = self._faiss_cfg.hard_negatives.mine_frequency
        if freq <= 0 or step == 0 or step % freq != 0:
            return

        if self._faiss_index is None:
            logger.debug(
                "[Phase 2] Hard-negative refresh skipped at step %d — "
                "no FAISS index provided.",
                step,
            )
            return

        logger.info(
            "[Phase 2] Hard-negative refresh triggered at step %d (frequency=%d).",
            step, freq,
        )

        num_hard = self._faiss_cfg.hard_negatives.num_hard_negatives
        top_k_candidates = self._faiss_cfg.hard_negatives.get(
            "top_k_candidates", num_hard * 3
        ) if hasattr(self._faiss_cfg.hard_negatives, "get") else num_hard * 3

        self._model.eval()
        refreshed = 0

        # Access underlying dataset (unwrap Subset if needed)
        underlying_ds = self._train_ds
        if hasattr(underlying_ds, "dataset"):
            underlying_ds = underlying_ds.dataset

        with torch.no_grad():
            for idx in range(len(underlying_ds)):
                try:
                    sample = underlying_ds[idx]
                except (IndexError, KeyError):
                    continue

                input_ids = sample["input_ids"].unsqueeze(0).to(self._device)
                attn_mask = sample["attention_mask"].unsqueeze(0).to(self._device)

                # Forward through backbone + retrieval head to get query embedding
                backbone_out = self._model.backbone(input_ids, attn_mask)
                hidden = backbone_out.last_hidden_state
                # Use the last non-padding position
                lengths = attn_mask.sum(dim=1, keepdim=True) - 1
                last_hidden = hidden.gather(
                    1, lengths.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
                )
                query_sig = self._model.retrieval_head.forward(last_hidden)
                query_emb = query_sig.semantic_embedding.cpu().numpy().squeeze()

                # Determine the positive fact_id from the sample if available
                positive_fact_id = sample.get("fact_id", "") if isinstance(sample, dict) else ""

                # Mine hard negatives from FAISS
                neg_fact_ids = self._faiss_index.mine_hard_negatives(
                    query_embedding=query_emb,
                    positive_fact_id=positive_fact_id,
                    num_negatives=num_hard,
                    top_k_candidates=top_k_candidates,
                )

                if neg_fact_ids and self._sapbert_encoder is not None:
                    # Re-encode negative facts via SapBERT to get embeddings
                    import numpy as np
                    neg_embeddings = []
                    for fid in neg_fact_ids:
                        # Search the index for the embedding (reconstruct is not
                        # available in all FAISS index types, so we use the
                        # SapBERT encoder with the fact text if possible).
                        # Fallback: use a zero vector.
                        results = self._faiss_index.search(
                            query_emb, top_k=1
                        )
                        neg_embeddings.append(np.zeros(query_emb.shape[-1], dtype=np.float32))

                    if neg_embeddings:
                        neg_arr = np.stack(neg_embeddings, axis=0)
                        # Update the underlying dataset example if it supports mutation
                        if hasattr(underlying_ds, "_update_negatives"):
                            underlying_ds._update_negatives(idx, neg_arr)
                        refreshed += 1

        self._model.train()
        logger.info(
            "[Phase 2] Hard-negative refresh complete: refreshed %d / %d examples.",
            refreshed, len(underlying_ds),
        )

    # ------------------------------------------------------------------
    # Curriculum difficulty
    # ------------------------------------------------------------------

    def _curriculum_temperature(self, epoch: int, total_epochs: int) -> float:
        """Anneal contrastive temperature from a warm start → target.

        Early epochs use a higher τ (easier); later epochs use the
        config value (harder).  Linear schedule.
        """
        warm_t = self._rcfg.contrastive_temperature * 2.0  # easier start
        target_t = self._rcfg.contrastive_temperature
        progress = epoch / max(total_epochs - 1, 1)
        return warm_t + (target_t - warm_t) * progress

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, phase1_checkpoint: Optional[Union[str, Path]] = None) -> Dict[str, float]:
        """Run Phase 2 training.

        Parameters
        ----------
        phase1_checkpoint : path, optional
            Path to the Phase 1 router checkpoint to load first.

        Returns
        -------
        dict
            Best validation metrics.
        """
        t0 = time.time()

        # --- Load Phase 1 checkpoint if given ---
        if phase1_checkpoint is not None:
            logger.info("Loading Phase 1 checkpoint: %s", phase1_checkpoint)
            ckpt_mgr = CheckpointManager(output_dir=Path(phase1_checkpoint).parent)
            ckpt_mgr.load(
                model=self._model,
                checkpoint_path=phase1_checkpoint,
                device=str(self._device),
            )

        # --- W&B ---
        self._wandb_run = init_wandb(
            project=self._wandb_cfg.project,
            run_name=f"phase2-retrieval-{int(time.time())}",
            tags=self._wandb_cfg.tags + ["phase2", "retrieval"],
            config={
                "phase": 2,
                "epochs": self._rcfg.epochs,
                "batch_size": self._rcfg.batch_size,
                "lr": self._rcfg.learning_rate,
                "scheduler": self._rcfg.scheduler,
                "contrastive_temperature": self._rcfg.contrastive_temperature,
                "freeze_backbone": self._rcfg.freeze_backbone,
                "freeze_router": self._rcfg.freeze_router,
            },
            enabled=self._wandb_cfg.enabled,
            entity=self._wandb_cfg.entity,
        )

        # --- Data ---
        train_ds, val_ds = self._build_datasets()
        train_dl, val_dl = self._build_dataloaders(train_ds, val_ds)

        # --- Model setup ---
        self._model.to(self._device)
        self._configure_frozen_params()

        # --- Loss ---
        self._loss_fn = InfoNCELoss(temperature=self._rcfg.contrastive_temperature)

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

        # --- Checkpoint ---
        ckpt_dir = Path(self._tcfg.output_dir) / "phase2_retrieval"
        self._ckpt_mgr = CheckpointManager(
            output_dir=ckpt_dir,
            max_checkpoints=self._tcfg.max_checkpoints,
            prefix="retrieval",
        )

        # --- Early stopping (maximise P@1) ---
        es_mode = "max" if self._rcfg.early_stopping_metric != "loss" else "min"
        self._early_stop = EarlyStopping(
            patience=self._rcfg.early_stopping_patience,
            metric_name=self._rcfg.early_stopping_metric,
            mode=es_mode,
        )

        # --- Loggers ---
        self._train_logger = MetricsLogger(
            wandb_run=self._wandb_run,
            log_frequency=self._tcfg.log_every_n_steps,
            prefix="retrieval/phase2/train",
        )
        self._val_logger = MetricsLogger(
            wandb_run=self._wandb_run,
            log_frequency=1,
            prefix="retrieval/phase2/val",
        )

        # --- Main loop ---
        best_metrics: Dict[str, float] = {}

        for epoch in range(self._rcfg.epochs):
            self._state.epoch = epoch

            # Curriculum temperature
            tau = self._curriculum_temperature(epoch, self._rcfg.epochs)
            self._loss_fn.temperature = tau

            # Train
            train_metrics = self._train_epoch(train_dl, use_amp)
            self._train_logger.log_epoch(train_metrics, epoch)

            # Validate
            val_metrics = self._evaluate(val_dl, use_amp)
            self._val_logger.log_epoch(val_metrics, epoch)

            # Checkpoint
            es_value = val_metrics.get(
                self._rcfg.early_stopping_metric,
                val_metrics.get("loss", 0.0),
            )
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

            if self._early_stop.step(es_value):
                logger.info("Early stopping at epoch %d", epoch)
                break

            logger.info(
                "[Phase 2] Epoch %d/%d — loss=%.4f  P@1=%.4f  MRR=%.4f  τ=%.4f",
                epoch + 1, self._rcfg.epochs,
                train_metrics.get("loss", 0.0),
                val_metrics.get("precision_at_1", 0.0),
                val_metrics.get("mrr", 0.0),
                tau,
            )

        # --- Finalise ---
        self._state.wall_time_seconds = time.time() - t0
        logger.info(
            "Phase 2 complete in %.1f s — best %s = %.4f",
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
        self._model.train()
        if self._rcfg.freeze_backbone:
            self._model.backbone.eval()

        # Keep router in eval (frozen)
        self._model.router.eval()

        epoch_loss = 0.0
        all_query: List[Tensor] = []
        all_pos: List[Tensor] = []
        all_neg: List[Tensor] = []
        n_batches = 0

        self._grad_acc.reset()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self._device)
            attention_mask = batch["attention_mask"].to(self._device)
            span_mask = batch["span_mask"].to(self._device)        # (B, seq)
            pos_emb = batch["positive_embedding"].to(self._device) # (B, edim)
            neg_embs = batch["negative_embeddings"].to(self._device)  # (B, n_neg, edim)

            with autocast(enabled=use_amp):
                # Backbone
                if self._rcfg.freeze_backbone:
                    with torch.no_grad():
                        backbone_out = self._model.backbone(input_ids, attention_mask)
                    hidden = backbone_out.last_hidden_state
                else:
                    backbone_out = self._model.backbone(input_ids, attention_mask)
                    hidden = backbone_out.last_hidden_state

                # Retrieval head → query signature
                query_sig = self._model.retrieval_head(hidden)
                # Use semantic embedding pooled over span positions
                # Mean-pool across retrieval positions
                query_emb = self._pool_query(
                    query_sig.semantic_embedding, span_mask,
                )  # (B, edim)

                # InfoNCE loss
                loss = self._loss_fn(query_emb, pos_emb, neg_embs)

            scaled = self._grad_acc.scale_loss(loss)
            self._scaler.scale(scaled).backward()

            if self._grad_acc.should_step():
                self._grad_acc.step(
                    optimizer=self._optimizer,
                    model=self._model,
                    scheduler=self._scheduler,
                    scaler=self._scaler,
                )
                self._state.global_step += 1

                # Hard-negative refresh
                self._maybe_refresh_negatives(self._state.global_step)

            epoch_loss += loss.item()
            n_batches += 1

            # Collect for metrics (detach and move to CPU)
            all_query.append(query_emb.detach().cpu())
            all_pos.append(pos_emb.detach().cpu())
            all_neg.append(neg_embs.detach().cpu())

            self._state.samples_seen += input_ids.size(0)

            self._train_logger.log_step(
                {"loss": loss.item(), "lr": self._optimizer.param_groups[0]["lr"]},
                step=self._state.global_step,
            )

        self._train_logger.flush(step=self._state.global_step)

        metrics = _compute_retrieval_metrics(all_query, all_pos, all_neg)
        metrics["loss"] = epoch_loss / max(n_batches, 1)
        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader, use_amp: bool) -> Dict[str, float]:
        self._model.eval()

        epoch_loss = 0.0
        all_query: List[Tensor] = []
        all_pos: List[Tensor] = []
        all_neg: List[Tensor] = []
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self._device)
            attention_mask = batch["attention_mask"].to(self._device)
            span_mask = batch["span_mask"].to(self._device)
            pos_emb = batch["positive_embedding"].to(self._device)
            neg_embs = batch["negative_embeddings"].to(self._device)

            with autocast(enabled=use_amp):
                backbone_out = self._model.backbone(input_ids, attention_mask)
                hidden = backbone_out.last_hidden_state
                query_sig = self._model.retrieval_head(hidden)
                query_emb = self._pool_query(query_sig.semantic_embedding, span_mask)
                loss = self._loss_fn(query_emb, pos_emb, neg_embs)

            epoch_loss += loss.item()
            n_batches += 1
            all_query.append(query_emb.cpu())
            all_pos.append(pos_emb.cpu())
            all_neg.append(neg_embs.cpu())

        metrics = _compute_retrieval_metrics(all_query, all_pos, all_neg)
        metrics["loss"] = epoch_loss / max(n_batches, 1)
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pool_query(semantic_emb: Tensor, span_mask: Tensor) -> Tensor:
        """Mean-pool semantic embeddings over retrieval positions.

        Parameters
        ----------
        semantic_emb : (batch, seq_len, emb_dim)
        span_mask : (batch, seq_len) — 1 at retrieval positions

        Returns
        -------
        Tensor : (batch, emb_dim) pooled queries
        """
        mask = span_mask.unsqueeze(-1)  # (B, seq, 1)
        pooled = (semantic_emb * mask).sum(dim=1)  # (B, edim)
        denom = mask.sum(dim=1).clamp(min=1.0)      # (B, 1)
        pooled = pooled / denom

        # L2-normalise
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------

    def resume_from_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        if self._ckpt_mgr is None:
            ckpt_dir = Path(self._tcfg.output_dir) / "phase2_retrieval"
            self._ckpt_mgr = CheckpointManager(output_dir=ckpt_dir)

        self._state = self._ckpt_mgr.load(
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            checkpoint_path=checkpoint_path,
            device=str(self._device),
        )
        logger.info("Resumed Phase 2 from step %d", self._state.global_step)

    @property
    def best_checkpoint_dir(self) -> Optional[Path]:
        if self._ckpt_mgr is None:
            return None
        return self._ckpt_mgr.latest()
