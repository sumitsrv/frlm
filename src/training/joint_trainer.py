"""
Phase 3 — Joint Fine-Tuning.

Unfreezes the entire model (backbone + router + retrieval + generation)
and trains with the combined loss:

    L = λ_router · L_router + λ_retrieval · L_retrieval + λ_gen · L_generation

Uses DeepSpeed ZeRO Stage 2 for multi-GPU training on 2-4× A100.

Training loop
-------------
1. Load Phase 1 (router) + Phase 2 (retrieval) checkpoints
2. Unfreeze all parameters (except SapBERT encoder which is external)
3. For each epoch:
   a. Forward full model → FRLMOutput with all three losses
   b. Gradient accumulation (effective batch = 8 × micro-batch)
   c. Cosine-with-restarts scheduler (3 cycles)
4. Log all three loss components + combined loss
5. Early-stop on combined_loss (minimise)
6. Save best checkpoint

W&B dashboard: ``joint/phase3/…``
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from src.model.frlm import FRLMModel
from src.training.dataset import JointDataset
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
# DeepSpeed helpers
# ===========================================================================


def _ensure_deepspeed_env() -> None:
    """Guarantee the distributed environment variables that DeepSpeed expects.

    Must be called **before** ``_build_deepspeed_config`` (which reads
    ``WORLD_SIZE``) and ``_init_deepspeed`` (which triggers
    ``dist.init_distributed``).

    * When a launcher (``deepspeed``, ``torchrun``, ``mpirun``) is used it
      will have already set ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``,
      ``MASTER_ADDR`` and ``MASTER_PORT`` — this function is a no-op.
    * When running plain ``python scripts/09_train_joint.py`` (single-GPU),
      we populate the env with safe single-process defaults so that
      DeepSpeed initialises NCCL/gloo directly instead of falling back to
      MPI discovery (which requires ``libmpi.so``).
    * ``DS_SKIP_CUDA_CHECK=1`` allows minor CUDA toolkit version mismatches
      (e.g. system 12.5 vs PyTorch-bundled 12.1) that are ABI-compatible
      within the same major version.
    """
    # Bypass strict CUDA version matching in DeepSpeed JIT builder
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        logger.info(
            "No distributed launcher detected — set RANK=0, WORLD_SIZE=1 "
            "to avoid MPI fallback."
        )


def _build_deepspeed_config(
    config: Any,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    total_steps: int,
    warmup_steps: int,
) -> Dict[str, Any]:
    """Build a DeepSpeed JSON config dict from the FRLM config.

    Fills ``"auto"`` placeholders with concrete values.

    .. note:: Call :func:`_ensure_deepspeed_env` before this function so
       that ``WORLD_SIZE`` is guaranteed to be in ``os.environ``.
    """
    ds_cfg = config.deepspeed.config

    # Convert pydantic model to dict
    ds_dict = json.loads(ds_cfg.model_dump_json())

    # Fill auto values
    ds_dict["train_micro_batch_size_per_gpu"] = micro_batch_size
    ds_dict["gradient_accumulation_steps"] = gradient_accumulation_steps

    # Calculate train_batch_size — use the distributed world size (which
    # DeepSpeed will validate against), NOT torch.cuda.device_count().
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ds_dict["train_batch_size"] = (
        micro_batch_size * gradient_accumulation_steps * world_size
    )

    # When running single-GPU, disable CPU optimizer offload.
    # DeepSpeedCPUAdam requires JIT-compiling a CUDA C++ extension which
    # fails when the system CUDA toolkit version doesn't exactly match the
    # version PyTorch was compiled against.  On a single GPU there is no
    # distributed-memory benefit anyway — FusedAdam on-GPU is faster.
    if world_size == 1:
        zero_cfg = ds_dict.get("zero_optimization", {})
        offload_opt = zero_cfg.get("offload_optimizer", {})
        if offload_opt.get("device", "none") != "none":
            logger.info(
                "Single-GPU run (world_size=1): switching "
                "offload_optimizer.device from '%s' → 'none' to avoid "
                "CPUAdam JIT build.",
                offload_opt["device"],
            )
            offload_opt["device"] = "none"

        # Shrink communication buffers — there is no cross-GPU traffic on a
        # single device, so the large default (5e8 ≈ 1 GB in FP16) wastes
        # VRAM and can cause illegal-memory-access errors when the
        # allocation silently overflows.
        _SINGLE_GPU_BUCKET = int(5e7)  # 50 M elements ≈ 100 MB in FP16
        for buf_key in ("reduce_bucket_size", "allgather_bucket_size"):
            cur = zero_cfg.get(buf_key, 0)
            if isinstance(cur, (int, float)) and cur > _SINGLE_GPU_BUCKET:
                logger.info(
                    "Single-GPU run: reducing %s from %g → %d",
                    buf_key, cur, _SINGLE_GPU_BUCKET,
                )
                zero_cfg[buf_key] = _SINGLE_GPU_BUCKET
        # No overlap benefit without actual distributed communication.
        zero_cfg["overlap_comm"] = False

    # Optimizer LR / weight decay
    jcfg = config.training.joint
    opt_params = ds_dict.get("optimizer", {}).get("params", {})
    if opt_params.get("lr") == "auto":
        opt_params["lr"] = jcfg.learning_rate
    if opt_params.get("weight_decay") == "auto":
        opt_params["weight_decay"] = jcfg.weight_decay

    # Scheduler
    sched_params = ds_dict.get("scheduler", {}).get("params", {})
    if sched_params.get("warmup_max_lr") == "auto":
        sched_params["warmup_max_lr"] = jcfg.learning_rate
    if sched_params.get("warmup_num_steps") == "auto":
        sched_params["warmup_num_steps"] = warmup_steps
    if sched_params.get("total_num_steps") == "auto":
        sched_params["total_num_steps"] = total_steps

    return ds_dict


def _init_deepspeed(
    model: nn.Module,
    ds_config: Dict[str, Any],
) -> Tuple[Any, Any, Any, Any]:
    """Initialise DeepSpeed engine.

    Returns (engine, optimizer, _, lr_scheduler).

    .. note:: Call :func:`_ensure_deepspeed_env` before this function.
    """
    try:
        import deepspeed
    except ImportError:
        raise ImportError(
            "DeepSpeed is required for Phase 3 joint training. "
            "Install with: pip install deepspeed"
        )

    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    logger.info(
        "DeepSpeed engine initialised — ZeRO stage %d, world_size=%d",
        ds_config.get("zero_optimization", {}).get("stage", 2),
        engine.world_size,
    )
    return engine, optimizer, None, lr_scheduler


# ===========================================================================
# JointTrainer
# ===========================================================================


class JointTrainer:
    """Phase 3 trainer — joint fine-tuning of all components.

    Parameters
    ----------
    model : FRLMModel
        Full model with all heads and :class:`FRLMCombinedLoss`.
    config : object
        ``FRLMConfig``.
    train_dataset : JointDataset, optional
    val_dataset : JointDataset, optional
    device : str
    use_deepspeed : bool
        Whether to use DeepSpeed ZeRO Stage 2. Defaults to the config
        value ``config.deepspeed.enabled``.
    """

    def __init__(
        self,
        model: FRLMModel,
        config: Any,
        train_dataset: Optional[JointDataset] = None,
        val_dataset: Optional[JointDataset] = None,
        device: str = "cuda",
        use_deepspeed: Optional[bool] = None,
    ) -> None:
        self._model = model
        self._cfg = config
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._jcfg = config.training.joint     # JointTrainingConfig
        self._tcfg = config.training           # TrainingConfig
        self._wandb_cfg = config.wandb
        self._loss_cfg = config.loss

        self._use_deepspeed = (
            use_deepspeed if use_deepspeed is not None
            else config.deepspeed.enabled
        )

        self._train_ds = train_dataset
        self._val_ds = val_dataset

        # Runtime state (initialised in train())
        self._optimizer: Optional[Any] = None
        self._scheduler: Optional[Any] = None
        self._scaler: Optional[GradScaler] = None
        self._ckpt_mgr: Optional[CheckpointManager] = None
        self._grad_acc: Optional[GradientAccumulator] = None
        self._early_stop: Optional[EarlyStopping] = None
        self._train_logger: Optional[MetricsLogger] = None
        self._val_logger: Optional[MetricsLogger] = None
        self._state = TrainingState(best_metric_name=self._jcfg.early_stopping_metric)
        self._wandb_run: Any = None
        self._ds_engine: Optional[Any] = None  # DeepSpeed engine

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def _build_datasets(self) -> Tuple[JointDataset, JointDataset]:
        if self._train_ds is not None and self._val_ds is not None:
            return self._train_ds, self._val_ds

        data_dir = Path(self._cfg.paths.resolve("processed_dir"))
        emb_dim = self._cfg.model.retrieval_head.semantic.output_dim
        num_neg = (
            self._cfg.faiss.hard_negatives.num_hard_negatives
            + self._cfg.faiss.hard_negatives.num_random_negatives
        )

        full_ds = JointDataset(
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
            batch_size=self._jcfg.batch_size,
            shuffle=True,
            num_workers=self._tcfg.dataloader_num_workers,
            pin_memory=self._tcfg.pin_memory,
            drop_last=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=self._jcfg.batch_size * 2,
            shuffle=False,
            num_workers=self._tcfg.dataloader_num_workers,
            pin_memory=self._tcfg.pin_memory,
        )
        return train_dl, val_dl

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _configure_params(self) -> None:
        """Unfreeze all parameters for joint training."""
        # Unfreeze backbone
        if not self._jcfg.freeze_backbone:
            self._model.backbone.unfreeze()
        else:
            self._model.backbone.freeze()

        # Unfreeze router
        if not self._jcfg.freeze_router:
            for p in self._model.router.parameters():
                p.requires_grad = True
        else:
            for p in self._model.router.parameters():
                p.requires_grad = False

        # Unfreeze retrieval head
        if not self._jcfg.freeze_retrieval:
            for p in self._model.retrieval_head.parameters():
                p.requires_grad = True
        else:
            for p in self._model.retrieval_head.parameters():
                p.requires_grad = False

        # Unfreeze generation head
        for p in self._model.generation_head.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Phase 3 params: %.2fM / %.2fM trainable",
            trainable / 1e6, total / 1e6,
        )

    # ------------------------------------------------------------------
    # Load prior phase checkpoints
    # ------------------------------------------------------------------

    def _load_phase_checkpoints(
        self,
        phase1_checkpoint: Optional[Union[str, Path]] = None,
        phase2_checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        """Load checkpoints from Phase 1 and/or Phase 2.

        Parameters
        ----------
        device : str, optional
            Override the map-location for ``torch.load``.  When ``None``
            the trainer's own ``self._device`` is used.  Pass ``"cpu"``
            when DeepSpeed will handle device placement later.
        """
        load_device = device if device is not None else str(self._device)

        if phase1_checkpoint is not None:
            logger.info("Loading Phase 1 router checkpoint: %s", phase1_checkpoint)
            ckpt = CheckpointManager(output_dir=Path(phase1_checkpoint).parent)
            ckpt.load(
                model=self._model,
                checkpoint_path=phase1_checkpoint,
                device=load_device,
            )

        if phase2_checkpoint is not None:
            logger.info("Loading Phase 2 retrieval checkpoint: %s", phase2_checkpoint)
            ckpt = CheckpointManager(output_dir=Path(phase2_checkpoint).parent)
            ckpt.load(
                model=self._model,
                checkpoint_path=phase2_checkpoint,
                device=load_device,
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        phase1_checkpoint: Optional[Union[str, Path]] = None,
        phase2_checkpoint: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """Run Phase 3 joint training.

        Parameters
        ----------
        phase1_checkpoint : path, optional
            Best Phase 1 checkpoint directory.
        phase2_checkpoint : path, optional
            Best Phase 2 checkpoint directory.

        Returns
        -------
        dict
            Best validation metrics.
        """
        t0 = time.time()

        # --- Determine whether DeepSpeed will drive training ---
        use_ds = self._use_deepspeed and torch.cuda.is_available()

        # --- Load prior phases ---
        # When DeepSpeed is active, keep the model on CPU during
        # checkpoint loading so that ``deepspeed.initialize()`` can
        # manage device placement and FP16 conversion in one pass.
        # Pre-moving to CUDA and *then* handing to DeepSpeed doubles
        # peak VRAM (model + FP16 copy + comm buffers) and can cause
        # ``CUDA error: an illegal memory access was encountered``.
        if use_ds:
            self._load_phase_checkpoints(
                phase1_checkpoint, phase2_checkpoint, device="cpu",
            )
        else:
            self._model.to(self._device)
            self._load_phase_checkpoints(phase1_checkpoint, phase2_checkpoint)

        # --- Configure parameters ---
        self._configure_params()

        # --- W&B ---
        self._wandb_run = init_wandb(
            project=self._wandb_cfg.project,
            run_name=f"phase3-joint-{int(time.time())}",
            tags=self._wandb_cfg.tags + ["phase3", "joint"],
            config={
                "phase": 3,
                "epochs": self._jcfg.epochs,
                "batch_size": self._jcfg.batch_size,
                "lr": self._jcfg.learning_rate,
                "scheduler": self._jcfg.scheduler,
                "num_cycles": self._jcfg.num_cycles,
                "use_deepspeed": self._use_deepspeed,
                "loss_weights": {
                    "router": self._loss_cfg.router_weight,
                    "retrieval": self._loss_cfg.retrieval_weight,
                    "generation": self._loss_cfg.generation_weight,
                },
            },
            enabled=self._wandb_cfg.enabled,
            entity=self._wandb_cfg.entity,
            api_key=self._wandb_cfg.api_key,
        )

        # --- Data ---
        train_ds, val_ds = self._build_datasets()
        train_dl, val_dl = self._build_dataloaders(train_ds, val_ds)

        total_steps = len(train_dl) * self._jcfg.epochs
        warmup_steps = int(total_steps * self._jcfg.warmup_ratio)

        # --- DeepSpeed or standard training ---
        use_amp: bool
        if self._use_deepspeed and torch.cuda.is_available():
            _ensure_deepspeed_env()
            ds_config = _build_deepspeed_config(
                config=self._cfg,
                micro_batch_size=self._jcfg.batch_size,
                gradient_accumulation_steps=self._tcfg.gradient_accumulation_steps,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )
            self._ds_engine, self._optimizer, _, self._scheduler = _init_deepspeed(
                model=self._model,
                ds_config=ds_config,
            )
            # Align self._device with where DeepSpeed actually placed the
            # model (cuda:<local_rank>) and flush any deferred CUDA errors
            # from initialisation so they surface here, not at the first
            # innocent-looking ``.to()`` call inside the training loop.
            self._device = self._ds_engine.device
            torch.cuda.synchronize()
            use_amp = False  # DeepSpeed handles FP16 internally
            self._grad_acc = None
            self._scaler = None
        else:
            # Standard PyTorch training
            self._optimizer = torch.optim.AdamW(
                [p for p in self._model.parameters() if p.requires_grad],
                lr=self._jcfg.learning_rate,
                weight_decay=self._jcfg.weight_decay,
            )
            self._scheduler = LearningRateScheduler(
                optimizer=self._optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                schedule_type=self._jcfg.scheduler,
                num_cycles=self._jcfg.num_cycles,
            )
            use_amp = self._tcfg.fp16 and self._device.type == "cuda"
            self._scaler = GradScaler(enabled=use_amp)
            self._grad_acc = GradientAccumulator(
                accumulation_steps=self._tcfg.gradient_accumulation_steps,
                max_grad_norm=self._tcfg.max_grad_norm,
            )

        # --- Checkpoint ---
        ckpt_dir = Path(self._tcfg.output_dir) / "phase3_joint"
        self._ckpt_mgr = CheckpointManager(
            output_dir=ckpt_dir,
            max_checkpoints=self._tcfg.max_checkpoints,
            prefix="joint",
        )

        # --- Early stopping (minimise combined_loss) ---
        es_mode = "min"  # combined_loss → lower is better
        self._early_stop = EarlyStopping(
            patience=self._jcfg.early_stopping_patience,
            metric_name=self._jcfg.early_stopping_metric,
            mode=es_mode,
        )

        # --- Loggers ---
        self._train_logger = MetricsLogger(
            wandb_run=self._wandb_run,
            log_frequency=self._tcfg.log_every_n_steps,
            prefix="joint/phase3/train",
        )
        self._val_logger = MetricsLogger(
            wandb_run=self._wandb_run,
            log_frequency=1,
            prefix="joint/phase3/val",
        )

        # --- Main loop ---
        best_metrics: Dict[str, float] = {}

        for epoch in range(self._jcfg.epochs):
            self._state.epoch = epoch

            # Train
            train_metrics = self._train_epoch(train_dl, use_amp)
            self._train_logger.log_epoch(train_metrics, epoch)

            # Validate
            val_metrics = self._evaluate(val_dl, use_amp)
            self._val_logger.log_epoch(val_metrics, epoch)

            # Checkpoint
            es_value = val_metrics.get(
                self._jcfg.early_stopping_metric,
                val_metrics.get("combined_loss", val_metrics.get("total_loss", 0.0)),
            )
            is_best = self._early_stop.best_value is None or (
                es_value < self._early_stop.best_value
            )

            if is_best:
                self._state.best_metric = es_value
                best_metrics = val_metrics.copy()
                self._ckpt_mgr.save(
                    model=self._model,
                    optimizer=None,  # DeepSpeed manages optimizer state
                    scheduler=None,
                    state=self._state,
                    metrics=val_metrics,
                )

            if self._early_stop.step(es_value):
                logger.info("Early stopping at epoch %d", epoch)
                break

            logger.info(
                "[Phase 3] Epoch %d/%d — total=%.4f  router=%.4f  "
                "retrieval=%.4f  generation=%.4f",
                epoch + 1, self._jcfg.epochs,
                train_metrics.get("total_loss", 0.0),
                train_metrics.get("router_loss", 0.0),
                train_metrics.get("retrieval_loss", 0.0),
                train_metrics.get("generation_loss", 0.0),
            )

        # --- Finalise ---
        self._state.wall_time_seconds = time.time() - t0
        logger.info(
            "Phase 3 complete in %.1f s — best %s = %.4f",
            self._state.wall_time_seconds,
            self._jcfg.early_stopping_metric,
            self._state.best_metric,
        )
        finish_wandb(self._wandb_run)

        return best_metrics

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, dataloader: DataLoader, use_amp: bool) -> Dict[str, float]:
        model = self._ds_engine if self._ds_engine is not None else self._model
        model.train()

        # Resolve the target device — when DeepSpeed is active use the
        # engine's device to guarantee data lands on the correct GPU.
        device = self._ds_engine.device if self._ds_engine is not None else self._device

        epoch_losses: Dict[str, float] = {}
        n_batches = 0

        if self._grad_acc is not None:
            self._grad_acc.reset()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            router_labels = batch["router_labels"].to(device)
            span_mask = batch["span_mask"].to(device)
            pos_emb = batch["positive_embedding"].to(device)
            neg_embs = batch["negative_embeddings"].to(device)
            token_labels = batch["token_labels"].to(device)

            # Reshape embeddings for the full model forward:
            #   fact_embeddings: (B, seq, emb_dim) — broadcast positive per position
            #   negative_embeddings: (B, seq, n_neg, emb_dim)
            B, seq_len = input_ids.shape
            edim = pos_emb.size(-1)
            n_neg = neg_embs.size(1)

            # Expand per-example embeddings to per-position
            fact_emb_expanded = pos_emb.unsqueeze(1).expand(B, seq_len, edim)
            neg_emb_expanded = neg_embs.unsqueeze(1).expand(B, seq_len, n_neg, edim)

            if self._ds_engine is not None:
                # --- DeepSpeed forward / backward ---
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    router_labels=router_labels,
                    fact_embeddings=fact_emb_expanded,
                    negative_embeddings=neg_emb_expanded,
                    token_labels=token_labels,
                )
                loss = output.total_loss
                self._ds_engine.backward(loss)
                self._ds_engine.step()
            else:
                # --- Standard PyTorch ---
                with autocast(enabled=use_amp):
                    output = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        router_labels=router_labels,
                        fact_embeddings=fact_emb_expanded,
                        negative_embeddings=neg_emb_expanded,
                        token_labels=token_labels,
                    )
                    loss = output.total_loss

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
            self._state.samples_seen += B
            n_batches += 1

            # Accumulate per-component losses
            if output.loss_dict is not None:
                for k, v in output.loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            # Step-level logging
            step_metrics = {"total_loss": loss.item()}
            if output.loss_dict:
                for k, v in output.loss_dict.items():
                    step_metrics[k] = v.item()
            lr = (
                self._optimizer.param_groups[0]["lr"]
                if self._optimizer is not None and hasattr(self._optimizer, "param_groups")
                else 0.0
            )
            step_metrics["lr"] = lr
            self._train_logger.log_step(step_metrics, step=self._state.global_step)

        self._train_logger.flush(step=self._state.global_step)

        # Average epoch losses
        metrics = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        # Alias for early stopping
        if "total_loss" in metrics:
            metrics["combined_loss"] = metrics["total_loss"]
        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader, use_amp: bool) -> Dict[str, float]:
        model = self._ds_engine if self._ds_engine is not None else self._model
        model.eval()

        # Resolve the target device (same rationale as _train_epoch).
        device = self._ds_engine.device if self._ds_engine is not None else self._device

        epoch_losses: Dict[str, float] = {}
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            router_labels = batch["router_labels"].to(device)
            span_mask = batch["span_mask"].to(device)
            pos_emb = batch["positive_embedding"].to(device)
            neg_embs = batch["negative_embeddings"].to(device)
            token_labels = batch["token_labels"].to(device)

            B, seq_len = input_ids.shape
            edim = pos_emb.size(-1)
            n_neg = neg_embs.size(1)
            fact_emb_expanded = pos_emb.unsqueeze(1).expand(B, seq_len, edim)
            neg_emb_expanded = neg_embs.unsqueeze(1).expand(B, seq_len, n_neg, edim)

            # DeepSpeed handles FP16 internally; only use autocast for
            # standard PyTorch path to avoid double-casting.
            with autocast(enabled=use_amp and self._ds_engine is None):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    router_labels=router_labels,
                    fact_embeddings=fact_emb_expanded,
                    negative_embeddings=neg_emb_expanded,
                    token_labels=token_labels,
                )

            n_batches += 1
            if output.loss_dict is not None:
                for k, v in output.loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

        metrics = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        if "total_loss" in metrics:
            metrics["combined_loss"] = metrics["total_loss"]
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------

    def resume_from_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        if self._ckpt_mgr is None:
            ckpt_dir = Path(self._tcfg.output_dir) / "phase3_joint"
            self._ckpt_mgr = CheckpointManager(output_dir=ckpt_dir)

        self._state = self._ckpt_mgr.load(
            model=self._model,
            optimizer=None,
            scheduler=None,
            checkpoint_path=checkpoint_path,
            device=str(self._device),
        )
        logger.info("Resumed Phase 3 from step %d", self._state.global_step)

    @property
    def best_checkpoint_dir(self) -> Optional[Path]:
        if self._ckpt_mgr is None:
            return None
        return self._ckpt_mgr.latest()
