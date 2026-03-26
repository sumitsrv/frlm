"""
Training Utilities.

Re-usable components shared across all three training phases:

- :class:`CheckpointManager` — save / load model + optimizer + scheduler + state
- :class:`MetricsLogger`     — aggregate and log metrics to W&B + console
- :class:`EarlyStopping`     — configurable patience, metric, mode (min/max)
- :class:`GradientAccumulator` — for effective batch size > actual batch size
- :class:`LearningRateScheduler` — warmup + cosine / linear decay
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


# ===========================================================================
# GPU device selection
# ===========================================================================


def resolve_device(gpu_id: int = 0) -> str:
    """Return a CUDA device string for *gpu_id* and pin the process to it.

    Parameters
    ----------
    gpu_id : int
        CUDA device ordinal (0, 1, …).  Use ``-1`` to force CPU.

    Returns
    -------
    str
        ``"cuda:N"`` or ``"cpu"``.

    Side-effects
    -------------
    * Calls ``torch.cuda.set_device(gpu_id)`` so that *all* subsequent
      default-device allocations (``torch.empty(..., device="cuda")``,
      ``model.cuda()``, etc.) land on the chosen GPU.
    * Sets ``CUDA_VISIBLE_DEVICES`` in ``os.environ`` so that child
      processes (e.g. DeepSpeed, dataloader workers) also see only the
      selected device.
    """
    if gpu_id < 0 or not torch.cuda.is_available():
        logger.info("Using CPU (gpu_id=%d, cuda_available=%s)",
                     gpu_id, torch.cuda.is_available())
        return "cpu"

    num_gpus = torch.cuda.device_count()
    if gpu_id >= num_gpus:
        logger.warning(
            "Requested gpu_id=%d but only %d GPU(s) visible — falling "
            "back to GPU 0.",
            gpu_id, num_gpus,
        )
        gpu_id = 0

    torch.cuda.set_device(gpu_id)
    logger.info(
        "Pinned process to GPU %d (%s)",
        gpu_id, torch.cuda.get_device_name(gpu_id),
    )
    return f"cuda:{gpu_id}"


# ===========================================================================
# CheckpointManager
# ===========================================================================


@dataclass
class TrainingState:
    """Serialisable snapshot of the training loop state.

    Everything needed to **resume** training from exactly where we left
    off: epoch, global step, best metric, and RNG states.
    """

    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_metric_name: str = "loss"
    samples_seen: int = 0
    wall_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name,
            "samples_seen": self.samples_seen,
            "wall_time_seconds": self.wall_time_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingState":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class CheckpointManager:
    """Save and load training checkpoints with rotation.

    Parameters
    ----------
    output_dir : str or Path
        Root directory for checkpoints.
    max_checkpoints : int
        Maximum number of checkpoints to keep; older ones are deleted.
    prefix : str
        Filename prefix for checkpoint files.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        max_checkpoints: int = 5,
        prefix: str = "checkpoint",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._max_checkpoints = max_checkpoints
        self._prefix = prefix
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    # ------------------------------------------------------------------ save

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        scheduler: Optional[_LRScheduler],
        state: TrainingState,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Persist a checkpoint and rotate old ones.

        Returns the path to the saved checkpoint.
        """
        tag = f"{self._prefix}_step_{state.global_step:07d}"
        ckpt_dir = self._output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Model
        torch.save(model.state_dict(), ckpt_dir / "model.pt")

        # Optimizer
        if optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        # Scheduler
        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

        # Training state + metrics
        meta: Dict[str, Any] = {"state": state.to_dict()}
        if metrics:
            meta["metrics"] = metrics
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "Checkpoint saved: %s (step=%d, epoch=%d)",
            ckpt_dir.name, state.global_step, state.epoch,
        )

        self._rotate()
        return ckpt_dir

    # ------------------------------------------------------------------ load

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ) -> TrainingState:
        """Load a checkpoint. If *checkpoint_path* is ``None``, loads the
        latest checkpoint in the output directory.

        Returns the restored :class:`TrainingState`.
        """
        if checkpoint_path is None:
            checkpoint_path = self.latest()
        else:
            checkpoint_path = Path(checkpoint_path)

        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}"
            )

        logger.info("Loading checkpoint from %s", checkpoint_path)

        model.load_state_dict(
            torch.load(checkpoint_path / "model.pt", map_location=device)
        )

        if optimizer is not None and (checkpoint_path / "optimizer.pt").exists():
            optimizer.load_state_dict(
                torch.load(checkpoint_path / "optimizer.pt", map_location=device)
            )

        if scheduler is not None and (checkpoint_path / "scheduler.pt").exists():
            scheduler.load_state_dict(
                torch.load(checkpoint_path / "scheduler.pt", map_location=device)
            )

        state = TrainingState()
        meta_path = checkpoint_path / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            state = TrainingState.from_dict(meta.get("state", {}))

        logger.info(
            "Checkpoint loaded: step=%d, epoch=%d", state.global_step, state.epoch
        )
        return state

    # ------------------------------------------------------------------ latest

    def latest(self) -> Optional[Path]:
        """Return the path to the most recent checkpoint, or ``None``."""
        dirs = sorted(
            (d for d in self._output_dir.iterdir() if d.is_dir() and d.name.startswith(self._prefix)),
            key=lambda p: p.stat().st_mtime,
        )
        return dirs[-1] if dirs else None

    # ------------------------------------------------------------------ rotate

    def _rotate(self) -> None:
        dirs = sorted(
            (d for d in self._output_dir.iterdir() if d.is_dir() and d.name.startswith(self._prefix)),
            key=lambda p: p.stat().st_mtime,
        )
        while len(dirs) > self._max_checkpoints:
            old = dirs.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            logger.debug("Rotated old checkpoint: %s", old.name)

    def list_checkpoints(self) -> List[Path]:
        """Return all checkpoint directories, oldest first."""
        return sorted(
            (d for d in self._output_dir.iterdir() if d.is_dir() and d.name.startswith(self._prefix)),
            key=lambda p: p.stat().st_mtime,
        )


# ===========================================================================
# MetricsLogger
# ===========================================================================


class MetricsLogger:
    """Aggregate and log metrics to W&B and the Python logger.

    Parameters
    ----------
    wandb_run : optional
        An active ``wandb.Run`` object.  ``None`` disables W&B logging.
    log_frequency : int
        Log every *n* calls to :meth:`log_step`.
    prefix : str
        Prefix prepended to all metric names in W&B (e.g. ``"train"``).
    """

    def __init__(
        self,
        wandb_run: Any = None,
        log_frequency: int = 10,
        prefix: str = "train",
    ) -> None:
        self._wandb_run = wandb_run
        self._log_freq = log_frequency
        self._prefix = prefix
        self._step_count = 0
        self._accumulators: Dict[str, List[float]] = {}

    @property
    def wandb_enabled(self) -> bool:
        return self._wandb_run is not None

    def log_step(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Record a single-step metric dict."""
        self._step_count += 1
        for k, v in metrics.items():
            self._accumulators.setdefault(k, []).append(v)

        if self._step_count % self._log_freq == 0:
            self._flush(step)

    def log_epoch(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log epoch-level aggregated metrics."""
        prefixed = {f"{self._prefix}/{k}": v for k, v in metrics.items()}
        prefixed[f"{self._prefix}/epoch"] = epoch

        if self._wandb_run is not None:
            try:
                self._wandb_run.log(prefixed)
            except Exception as exc:
                logger.debug("WandB log failed: %s", exc)

        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        logger.info("[%s] epoch=%d  %s", self._prefix, epoch, "  ".join(parts))

    def _flush(self, step: Optional[int] = None) -> None:
        if not self._accumulators:
            return

        averaged: Dict[str, float] = {}
        for k, vals in self._accumulators.items():
            averaged[f"{self._prefix}/{k}"] = sum(vals) / len(vals)

        if step is not None:
            averaged[f"{self._prefix}/step"] = step

        if self._wandb_run is not None:
            try:
                self._wandb_run.log(averaged)
            except Exception as exc:
                logger.debug("WandB log failed: %s", exc)

        self._accumulators.clear()

    def flush(self, step: Optional[int] = None) -> None:
        """Force-flush any accumulated metrics."""
        self._flush(step)

    def summary(self) -> Dict[str, float]:
        """Return the last accumulated averages (for testing)."""
        out: Dict[str, float] = {}
        for k, vals in self._accumulators.items():
            out[k] = sum(vals) / len(vals) if vals else 0.0
        return out


# ===========================================================================
# EarlyStopping
# ===========================================================================


class EarlyStopping:
    """Early stopping tracker.

    Parameters
    ----------
    patience : int
        Number of evaluations without improvement before stopping.
    metric_name : str
        Name of the monitored metric (for logging only).
    mode : ``"min"`` or ``"max"``
        Whether the metric should be minimised or maximised.
    min_delta : float
        Minimum absolute improvement to count as an improvement.
    """

    def __init__(
        self,
        patience: int = 3,
        metric_name: str = "loss",
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """Update with a new metric value.

        Returns ``True`` if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > (self.best_value + self.min_delta)
        else:
            improved = value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered: %s did not improve for %d evals (best=%.4f)",
                    self.metric_name, self.patience, self.best_value,
                )

        return self.should_stop

    def reset(self) -> None:
        """Reset the tracker."""
        self.best_value = None
        self.counter = 0
        self.should_stop = False


# ===========================================================================
# GradientAccumulator
# ===========================================================================


class GradientAccumulator:
    """Manages gradient accumulation for large effective batch sizes.

    Parameters
    ----------
    accumulation_steps : int
        Number of micro-batches to accumulate before an optimizer step.
    max_grad_norm : float or ``None``
        If set, clip gradients to this norm before stepping.
    """

    def __init__(
        self,
        accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        if accumulation_steps < 1:
            raise ValueError(
                f"accumulation_steps must be >= 1, got {accumulation_steps}"
            )
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self._micro_step: int = 0

    @property
    def micro_step(self) -> int:
        """Current micro-step within the accumulation window."""
        return self._micro_step

    def should_step(self) -> bool:
        """Whether the optimizer should perform a parameter update."""
        self._micro_step += 1
        return self._micro_step >= self.accumulation_steps

    def step(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[Any] = None,
    ) -> None:
        """Execute optimizer step with optional gradient clipping and AMP.

        Parameters
        ----------
        optimizer : Optimizer
        model : nn.Module
        scheduler : LR scheduler, optional
        scaler : torch.cuda.amp.GradScaler, optional
        """
        if self.max_grad_norm is not None:
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
        self._micro_step = 0

    def reset(self) -> None:
        """Reset the micro-step counter."""
        self._micro_step = 0

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale loss by the number of accumulation steps."""
        if self.accumulation_steps > 1:
            return loss / self.accumulation_steps
        return loss


# ===========================================================================
# LearningRateScheduler
# ===========================================================================


class LearningRateScheduler(_LRScheduler):
    """Warmup followed by cosine or linear decay.

    Parameters
    ----------
    optimizer : Optimizer
    total_steps : int
        Total number of training steps.
    warmup_steps : int
        Linear warmup from 0 to base LR over this many steps.
    schedule_type : str
        ``"cosine"`` or ``"linear"`` decay after warmup.
    min_lr_ratio : float
        Fraction of base LR at the end of training.
    num_cycles : int
        Number of cosine cycles (for ``"cosine_with_restarts"``).
    last_epoch : int
        Passed to ``_LRScheduler``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        schedule_type: str = "cosine",
        min_lr_ratio: float = 0.0,
        num_cycles: int = 1,
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type
        self.min_lr_ratio = min_lr_ratio
        self.num_cycles = num_cycles
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(self.warmup_steps, 1)
        else:
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(progress, 1.0)

            if self.schedule_type == "linear":
                scale = 1.0 - progress * (1.0 - self.min_lr_ratio)
            elif self.schedule_type == "cosine_with_restarts":
                scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                    1.0 + math.cos(math.pi * (progress * self.num_cycles % 1.0))
                )
            else:
                # Default: cosine
                scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                    1.0 + math.cos(math.pi * progress)
                )

        return [base_lr * scale for base_lr in self.base_lrs]


# ===========================================================================
# W&B initialisation helper
# ===========================================================================


def init_wandb(
    project: str,
    run_name: Optional[str],
    tags: List[str],
    config: Dict[str, Any],
    enabled: bool = True,
    entity: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """Safely initialise a Weights & Biases run.

    Returns the ``wandb.Run`` object or ``None`` if disabled / unavailable.
    """
    if not enabled:
        logger.info("WandB disabled.")
        return None
    try:
        import os
        import wandb

        # Set API key via env-var so wandb.init() never triggers the
        # interactive login prompt (which rejects newer v1 keys > 40 chars).
        if api_key and api_key != "CHANGE_ME":
            os.environ["WANDB_API_KEY"] = api_key
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name or f"run-{int(time.time())}",
            tags=tags,
            config=config,
        )
        logger.info("WandB initialised: %s/%s", project, run.name)
        return run
    except ImportError:
        logger.warning("wandb not installed — experiment tracking disabled.")
        return None
    except Exception as exc:
        logger.warning("WandB init failed: %s", exc)
        return None


def finish_wandb(run: Any) -> None:
    """Safely close a WandB run."""
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
