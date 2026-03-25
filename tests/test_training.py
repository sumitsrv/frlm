"""
Tests for Phase 7 — Training Pipeline.

Tests cover:
- TrainingState serialisation round-trip
- CheckpointManager save / load / rotation / latest
- MetricsLogger accumulation, flushing, prefix propagation
- EarlyStopping min/max modes, patience, min_delta, reset
- GradientAccumulator micro-step counting, should_step, scale_loss, clipping
- LearningRateScheduler warmup, cosine, linear, cosine_with_restarts
- RouterDataset: JSON + JSONL loading, padding, truncation, empty dir
- RetrievalDataset: embedding loading, negative padding/truncation
- JointDataset: full field set, token_labels padding with -100
- _compute_router_metrics: accuracy, precision, recall, F1
- _precision_at_k, _mean_reciprocal_rank: retrieval metric helpers
- _build_deepspeed_config: auto-fill of "auto" placeholders
- RouterTrainer: instantiation, freeze logic, build_optimizer
- RetrievalTrainer: instantiation, pool_query, curriculum_temperature
- JointTrainer: instantiation, configure_params
- init_wandb / finish_wandb: safe fallbacks when wandb unavailable
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, load_config
from src.model.frlm import FRLMModel

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
from src.training.dataset import JointDataset, RetrievalDataset, RouterDataset
from src.training.router_trainer import RouterTrainer, _compute_router_metrics
from src.training.retrieval_trainer import (
    RetrievalTrainer,
    _compute_retrieval_metrics,
    _mean_reciprocal_rank,
    _precision_at_k,
)
from src.training.joint_trainer import JointTrainer, _build_deepspeed_config, _ensure_deepspeed_env


# ====================================================================
# Fixtures
# ====================================================================

SEQ_LEN = 16
EMB_DIM = 32
NUM_NEG = 5
BATCH = 4


@pytest.fixture(scope="module")
def default_config() -> FRLMConfig:
    return load_config()


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def tiny_model() -> nn.Module:
    """A trivial model for checkpoint tests."""
    return nn.Linear(8, 4)


@pytest.fixture()
def router_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with router training data."""
    d = tmp_path / "labels"
    d.mkdir()

    # Single JSON file
    example = {
        "input_ids": list(range(SEQ_LEN)),
        "attention_mask": [1] * SEQ_LEN,
        "router_labels": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    }
    with open(d / "sample_001.json", "w") as f:
        json.dump(example, f)

    # JSONL file with 3 examples
    with open(d / "samples.jsonl", "w") as f:
        for i in range(3):
            ex = {
                "input_ids": list(range(i, i + SEQ_LEN)),
                "attention_mask": [1] * SEQ_LEN,
                "router_labels": [i % 2] * SEQ_LEN,
            }
            f.write(json.dumps(ex) + "\n")

    return d


@pytest.fixture()
def retrieval_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with retrieval training data."""
    d = tmp_path / "processed"
    d.mkdir()

    for i in range(3):
        example = {
            "input_ids": list(range(SEQ_LEN)),
            "attention_mask": [1] * SEQ_LEN,
            "span_mask": [1 if j % 3 == 0 else 0 for j in range(SEQ_LEN)],
            "positive_embedding": np.random.randn(EMB_DIM).tolist(),
            "negative_embeddings": np.random.randn(NUM_NEG, EMB_DIM).tolist(),
        }
        with open(d / f"ret_{i:03d}.json", "w") as f:
            json.dump(example, f)

    return d


@pytest.fixture()
def joint_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with joint training data."""
    d = tmp_path / "joint"
    d.mkdir()

    for i in range(4):
        example = {
            "input_ids": list(range(SEQ_LEN)),
            "attention_mask": [1] * SEQ_LEN,
            "router_labels": [1 if j % 2 == 0 else 0 for j in range(SEQ_LEN)],
            "span_mask": [1 if j % 2 == 0 else 0 for j in range(SEQ_LEN)],
            "positive_embedding": np.random.randn(EMB_DIM).tolist(),
            "negative_embeddings": np.random.randn(NUM_NEG, EMB_DIM).tolist(),
            "token_labels": list(range(1, SEQ_LEN + 1)),
        }
        with open(d / f"joint_{i:03d}.json", "w") as f:
            json.dump(example, f)

    return d


# ====================================================================
# SECTION 1 — TrainingState
# ====================================================================


class TestTrainingState:
    """Serialisation / deserialisation of TrainingState."""

    def test_default_values(self) -> None:
        state = TrainingState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_metric == 0.0
        assert state.best_metric_name == "loss"
        assert state.samples_seen == 0
        assert state.wall_time_seconds == 0.0

    def test_to_dict(self) -> None:
        state = TrainingState(epoch=5, global_step=100, best_metric=0.95)
        d = state.to_dict()
        assert d["epoch"] == 5
        assert d["global_step"] == 100
        assert d["best_metric"] == 0.95

    def test_round_trip(self) -> None:
        state = TrainingState(
            epoch=3, global_step=500, best_metric=0.88,
            best_metric_name="f1", samples_seen=1000, wall_time_seconds=123.4,
        )
        d = state.to_dict()
        restored = TrainingState.from_dict(d)
        assert restored.epoch == state.epoch
        assert restored.global_step == state.global_step
        assert restored.best_metric == state.best_metric
        assert restored.best_metric_name == state.best_metric_name
        assert restored.samples_seen == state.samples_seen
        assert restored.wall_time_seconds == state.wall_time_seconds

    def test_from_dict_ignores_extra_keys(self) -> None:
        d = {"epoch": 2, "global_step": 50, "unknown_field": "ignored"}
        state = TrainingState.from_dict(d)
        assert state.epoch == 2
        assert state.global_step == 50

    def test_from_dict_missing_keys_uses_defaults(self) -> None:
        state = TrainingState.from_dict({"epoch": 7})
        assert state.epoch == 7
        assert state.global_step == 0
        assert state.best_metric == 0.0


# ====================================================================
# SECTION 2 — CheckpointManager
# ====================================================================


class TestCheckpointManager:
    """Tests for saving, loading, rotating checkpoints."""

    def test_save_creates_directory(self, tmp_dir: Path, tiny_model: nn.Module) -> None:
        mgr = CheckpointManager(tmp_dir / "ckpts", max_checkpoints=3)
        state = TrainingState(global_step=10)
        path = mgr.save(tiny_model, None, None, state)
        assert path.exists()
        assert (path / "model.pt").exists()
        assert (path / "meta.json").exists()

    def test_save_with_optimizer(self, tmp_dir: Path, tiny_model: nn.Module) -> None:
        optim = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        mgr = CheckpointManager(tmp_dir / "ckpts")
        state = TrainingState(global_step=20)
        path = mgr.save(tiny_model, optim, None, state)
        assert (path / "optimizer.pt").exists()

    def test_save_with_metrics(self, tmp_dir: Path, tiny_model: nn.Module) -> None:
        mgr = CheckpointManager(tmp_dir / "ckpts")
        state = TrainingState(global_step=30)
        metrics = {"loss": 0.5, "accuracy": 0.9}
        path = mgr.save(tiny_model, None, None, state, metrics=metrics)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        assert meta["metrics"]["loss"] == 0.5
        assert meta["metrics"]["accuracy"] == 0.9

    def test_load_restores_model(self, tmp_dir: Path, tiny_model: nn.Module) -> None:
        mgr = CheckpointManager(tmp_dir / "ckpts")
        state = TrainingState(global_step=40, epoch=2)
        mgr.save(tiny_model, None, None, state)

        # Create a new model and load
        new_model = nn.Linear(8, 4)
        restored_state = mgr.load(new_model)
        assert restored_state.global_step == 40
        assert restored_state.epoch == 2

        # Weights should match
        for p1, p2 in zip(tiny_model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_latest(self, tmp_dir: Path, tiny_model: nn.Module) -> None:
        mgr = CheckpointManager(tmp_dir / "ckpts")
        assert mgr.latest() is None

        mgr.save(tiny_model, None, None, TrainingState(global_step=1))
        import time; time.sleep(0.05)
        mgr.save(tiny_model, None, None, TrainingState(global_step=2))

        latest = mgr.latest()
        assert latest is not None
        assert "0000002" in latest.name

    def test_rotation(self, tmp_dir: Path, tiny_model: nn.Module) -> None:
        mgr = CheckpointManager(tmp_dir / "ckpts", max_checkpoints=2)
        import time

        for i in range(4):
            mgr.save(tiny_model, None, None, TrainingState(global_step=i))
            time.sleep(0.05)

        dirs = mgr.list_checkpoints()
        assert len(dirs) == 2
        # Only the last two should remain
        names = [d.name for d in dirs]
        assert "checkpoint_step_0000002" in names
        assert "checkpoint_step_0000003" in names

    def test_load_raises_when_no_checkpoint(self, tmp_dir: Path, tiny_model: nn.Module) -> None:
        mgr = CheckpointManager(tmp_dir / "empty_ckpts")
        with pytest.raises(FileNotFoundError):
            mgr.load(tiny_model)


# ====================================================================
# SECTION 3 — MetricsLogger
# ====================================================================


class TestMetricsLogger:
    """Tests for metric accumulation and logging."""

    def test_accumulation(self) -> None:
        logger = MetricsLogger(wandb_run=None, log_frequency=5, prefix="train")
        for i in range(5):
            logger.log_step({"loss": float(i)})
        # After 5 steps, accumulators should be flushed
        s = logger.summary()
        # After flush, accumulators are cleared
        assert len(s) == 0

    def test_prefix_propagation(self) -> None:
        mock_wandb = MagicMock()
        logger = MetricsLogger(wandb_run=mock_wandb, log_frequency=2, prefix="val")
        logger.log_step({"loss": 1.0})
        logger.log_step({"loss": 2.0})

        # Should have called wandb.log with prefixed keys
        mock_wandb.log.assert_called()
        call_args = mock_wandb.log.call_args[0][0]
        assert "val/loss" in call_args

    def test_log_epoch(self) -> None:
        mock_wandb = MagicMock()
        logger = MetricsLogger(wandb_run=mock_wandb, prefix="router/phase1")
        logger.log_epoch({"f1": 0.85, "loss": 0.3}, epoch=2)
        call_args = mock_wandb.log.call_args[0][0]
        assert "router/phase1/f1" in call_args
        assert "router/phase1/loss" in call_args
        assert call_args["router/phase1/epoch"] == 2

    def test_wandb_disabled(self) -> None:
        logger = MetricsLogger(wandb_run=None, prefix="test")
        assert not logger.wandb_enabled
        # Should not raise
        logger.log_step({"loss": 0.5})
        logger.log_epoch({"loss": 0.3}, epoch=0)
        logger.flush()

    def test_manual_flush(self) -> None:
        mock_wandb = MagicMock()
        logger = MetricsLogger(wandb_run=mock_wandb, log_frequency=100, prefix="x")
        logger.log_step({"a": 1.0})
        logger.log_step({"a": 3.0})
        # Not auto-flushed yet (freq=100)
        s = logger.summary()
        assert abs(s["a"] - 2.0) < 1e-6

        logger.flush(step=99)
        mock_wandb.log.assert_called_once()
        # After flush, summary should be empty
        assert len(logger.summary()) == 0


# ====================================================================
# SECTION 4 — EarlyStopping
# ====================================================================


class TestEarlyStopping:
    """Tests for early stopping tracker."""

    def test_max_mode_no_stop(self) -> None:
        es = EarlyStopping(patience=3, mode="max", metric_name="f1")
        assert not es.step(0.5)
        assert not es.step(0.6)
        assert not es.step(0.7)
        assert es.counter == 0

    def test_max_mode_stops(self) -> None:
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.8)
        es.step(0.7)  # no improvement, counter=1
        assert not es.should_stop
        result = es.step(0.6)  # counter=2 → stop
        assert result is True
        assert es.should_stop

    def test_min_mode_no_stop(self) -> None:
        es = EarlyStopping(patience=3, mode="min", metric_name="loss")
        assert not es.step(1.0)
        assert not es.step(0.8)
        assert not es.step(0.5)
        assert es.counter == 0

    def test_min_mode_stops(self) -> None:
        es = EarlyStopping(patience=2, mode="min")
        es.step(0.5)
        es.step(0.6)  # worse
        result = es.step(0.6)  # worse again
        assert result is True

    def test_min_delta(self) -> None:
        es = EarlyStopping(patience=2, mode="max", min_delta=0.1)
        es.step(0.5)
        assert not es.step(0.55)  # improvement < delta, counter=1
        assert es.counter == 1
        assert not es.step(0.61)  # 0.61 > 0.5 + 0.1, improvement
        assert es.counter == 0

    def test_reset(self) -> None:
        es = EarlyStopping(patience=1, mode="min")
        es.step(0.5)
        es.step(0.6)  # triggers stop
        assert es.should_stop
        es.reset()
        assert not es.should_stop
        assert es.best_value is None
        assert es.counter == 0

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(mode="invalid")

    def test_first_step_never_stops(self) -> None:
        es = EarlyStopping(patience=1, mode="min")
        assert not es.step(999.0)
        assert es.best_value == 999.0


# ====================================================================
# SECTION 5 — GradientAccumulator
# ====================================================================


class TestGradientAccumulator:
    """Tests for gradient accumulation."""

    def test_should_step_period(self) -> None:
        ga = GradientAccumulator(accumulation_steps=4)
        results = []
        for _ in range(8):
            ready = ga.should_step()
            results.append(ready)
            if ready:
                ga.reset()
        # Should be True at steps 4 and 8 (1-indexed micro steps)
        assert results == [False, False, False, True, False, False, False, True]

    def test_scale_loss(self) -> None:
        ga = GradientAccumulator(accumulation_steps=4)
        loss = torch.tensor(2.0)
        scaled = ga.scale_loss(loss)
        assert torch.allclose(scaled, torch.tensor(0.5))

    def test_scale_loss_no_accumulation(self) -> None:
        ga = GradientAccumulator(accumulation_steps=1)
        loss = torch.tensor(2.0)
        scaled = ga.scale_loss(loss)
        assert torch.allclose(scaled, loss)

    def test_reset(self) -> None:
        ga = GradientAccumulator(accumulation_steps=4)
        ga.should_step()
        ga.should_step()
        assert ga.micro_step == 2
        ga.reset()
        assert ga.micro_step == 0

    def test_step_with_clipping(self) -> None:
        model = nn.Linear(4, 2)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()

        ga = GradientAccumulator(accumulation_steps=1, max_grad_norm=0.1)
        ga.should_step()
        ga.step(optim, model)

        # After step, gradients should be zeroed
        for p in model.parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, torch.zeros_like(p.grad))

    def test_step_with_scheduler(self) -> None:
        model = nn.Linear(4, 2)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5)

        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()

        ga = GradientAccumulator(accumulation_steps=1)
        ga.should_step()
        ga.step(optim, model, scheduler=scheduler)

        # Scheduler should have stepped
        assert optim.param_groups[0]["lr"] == pytest.approx(0.05)

    def test_invalid_accumulation_steps(self) -> None:
        with pytest.raises(ValueError, match="accumulation_steps must be >= 1"):
            GradientAccumulator(accumulation_steps=0)


# ====================================================================
# SECTION 6 — LearningRateScheduler
# ====================================================================


class TestLearningRateScheduler:
    """Tests for warmup + decay schedules."""

    def test_warmup_phase(self) -> None:
        optim = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = LearningRateScheduler(
            optim, total_steps=100, warmup_steps=10, schedule_type="cosine",
        )
        # At step 0, LR should be 0.0
        assert scheduler.get_lr()[0] == pytest.approx(0.0, abs=1e-6)

        # Simulate steps during warmup
        for _ in range(5):
            scheduler.step()
        assert scheduler.get_lr()[0] == pytest.approx(0.5, abs=0.1)

    def test_cosine_decay(self) -> None:
        optim = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = LearningRateScheduler(
            optim, total_steps=100, warmup_steps=0, schedule_type="cosine",
        )
        # At step 0, LR should be base_lr
        assert scheduler.get_lr()[0] == pytest.approx(1.0, abs=0.01)

        # At end, should approach min_lr_ratio (default 0.0)
        for _ in range(100):
            scheduler.step()
        assert scheduler.get_lr()[0] == pytest.approx(0.0, abs=0.01)

    def test_linear_decay(self) -> None:
        optim = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = LearningRateScheduler(
            optim, total_steps=100, warmup_steps=0, schedule_type="linear",
        )
        # Halfway through, LR should be ~0.5
        for _ in range(50):
            scheduler.step()
        assert scheduler.get_lr()[0] == pytest.approx(0.5, abs=0.05)

    def test_cosine_with_restarts(self) -> None:
        optim = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = LearningRateScheduler(
            optim, total_steps=100, warmup_steps=0,
            schedule_type="cosine_with_restarts", num_cycles=2,
        )
        # LR should cycle — at 25% and 75% it should be near 0
        for _ in range(25):
            scheduler.step()
        lr_at_quarter = scheduler.get_lr()[0]

        for _ in range(25):
            scheduler.step()
        lr_at_half = scheduler.get_lr()[0]
        # At each restart point, should bounce back up
        assert lr_at_half > lr_at_quarter

    def test_min_lr_ratio(self) -> None:
        optim = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = LearningRateScheduler(
            optim, total_steps=100, warmup_steps=0,
            schedule_type="cosine", min_lr_ratio=0.1,
        )
        for _ in range(100):
            scheduler.step()
        assert scheduler.get_lr()[0] >= 0.09  # Should not go below min_lr_ratio * base_lr


# ====================================================================
# SECTION 7 — RouterDataset
# ====================================================================


class TestRouterDataset:
    """Tests for Phase 1 router dataset."""

    def test_loads_json_and_jsonl(self, router_data_dir: Path) -> None:
        ds = RouterDataset(data_dir=router_data_dir, max_seq_length=SEQ_LEN)
        # 1 JSON + 3 JSONL = 4 examples
        assert len(ds) == 4

    def test_example_shape(self, router_data_dir: Path) -> None:
        ds = RouterDataset(data_dir=router_data_dir, max_seq_length=SEQ_LEN)
        example = ds[0]
        assert example["input_ids"].shape == (SEQ_LEN,)
        assert example["attention_mask"].shape == (SEQ_LEN,)
        assert example["router_labels"].shape == (SEQ_LEN,)

    def test_dtypes(self, router_data_dir: Path) -> None:
        ds = RouterDataset(data_dir=router_data_dir, max_seq_length=SEQ_LEN)
        example = ds[0]
        assert example["input_ids"].dtype == torch.long
        assert example["attention_mask"].dtype == torch.long
        assert example["router_labels"].dtype == torch.float

    def test_padding(self, router_data_dir: Path) -> None:
        ds = RouterDataset(data_dir=router_data_dir, max_seq_length=SEQ_LEN + 10)
        example = ds[0]
        # Last 10 positions should be padded with 0
        assert example["input_ids"].shape == (SEQ_LEN + 10,)
        assert example["attention_mask"][-1].item() == 0

    def test_truncation(self, router_data_dir: Path) -> None:
        ds = RouterDataset(data_dir=router_data_dir, max_seq_length=8)
        example = ds[0]
        assert example["input_ids"].shape == (8,)

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        ds = RouterDataset(data_dir=empty, max_seq_length=SEQ_LEN)
        assert len(ds) == 0

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        ds = RouterDataset(data_dir=tmp_path / "does_not_exist")
        assert len(ds) == 0

    def test_custom_files_list(self, router_data_dir: Path) -> None:
        files = list(router_data_dir.glob("*.json"))
        ds = RouterDataset(data_dir=router_data_dir, max_seq_length=SEQ_LEN, files=files)
        # Only JSON files, not JSONL
        assert len(ds) == 1


# ====================================================================
# SECTION 8 — RetrievalDataset
# ====================================================================


class TestRetrievalDataset:
    """Tests for Phase 2 retrieval dataset."""

    def test_loads_examples(self, retrieval_data_dir: Path) -> None:
        ds = RetrievalDataset(
            data_dir=retrieval_data_dir, max_seq_length=SEQ_LEN,
            embedding_dim=EMB_DIM, num_negatives=NUM_NEG,
        )
        assert len(ds) == 3

    def test_example_shape(self, retrieval_data_dir: Path) -> None:
        ds = RetrievalDataset(
            data_dir=retrieval_data_dir, max_seq_length=SEQ_LEN,
            embedding_dim=EMB_DIM, num_negatives=NUM_NEG,
        )
        example = ds[0]
        assert example["input_ids"].shape == (SEQ_LEN,)
        assert example["span_mask"].shape == (SEQ_LEN,)
        assert example["positive_embedding"].shape == (EMB_DIM,)
        assert example["negative_embeddings"].shape == (NUM_NEG, EMB_DIM)

    def test_dtypes(self, retrieval_data_dir: Path) -> None:
        ds = RetrievalDataset(
            data_dir=retrieval_data_dir, max_seq_length=SEQ_LEN,
            embedding_dim=EMB_DIM, num_negatives=NUM_NEG,
        )
        example = ds[0]
        assert example["span_mask"].dtype == torch.float
        assert example["positive_embedding"].dtype == torch.float32
        assert example["negative_embeddings"].dtype == torch.float32

    def test_negative_padding(self, tmp_path: Path) -> None:
        """When fewer negatives provided, should pad with zeros."""
        d = tmp_path / "ret"
        d.mkdir()
        example = {
            "input_ids": list(range(SEQ_LEN)),
            "span_mask": [1] * SEQ_LEN,
            "positive_embedding": np.random.randn(EMB_DIM).tolist(),
            "negative_embeddings": np.random.randn(2, EMB_DIM).tolist(),  # Only 2 negatives
        }
        with open(d / "test.json", "w") as f:
            json.dump(example, f)

        ds = RetrievalDataset(data_dir=d, max_seq_length=SEQ_LEN,
                              embedding_dim=EMB_DIM, num_negatives=NUM_NEG)
        item = ds[0]
        assert item["negative_embeddings"].shape == (NUM_NEG, EMB_DIM)
        # Last 3 should be zero-padded
        assert torch.allclose(item["negative_embeddings"][2:], torch.zeros(3, EMB_DIM))

    def test_negative_truncation(self, tmp_path: Path) -> None:
        """When more negatives provided, should truncate."""
        d = tmp_path / "ret"
        d.mkdir()
        example = {
            "input_ids": list(range(SEQ_LEN)),
            "span_mask": [1] * SEQ_LEN,
            "positive_embedding": np.random.randn(EMB_DIM).tolist(),
            "negative_embeddings": np.random.randn(10, EMB_DIM).tolist(),
        }
        with open(d / "test.json", "w") as f:
            json.dump(example, f)

        ds = RetrievalDataset(data_dir=d, max_seq_length=SEQ_LEN,
                              embedding_dim=EMB_DIM, num_negatives=NUM_NEG)
        item = ds[0]
        assert item["negative_embeddings"].shape == (NUM_NEG, EMB_DIM)


# ====================================================================
# SECTION 9 — JointDataset
# ====================================================================


class TestJointDataset:
    """Tests for Phase 3 joint dataset."""

    def test_loads_examples(self, joint_data_dir: Path) -> None:
        ds = JointDataset(
            data_dir=joint_data_dir, max_seq_length=SEQ_LEN,
            embedding_dim=EMB_DIM, num_negatives=NUM_NEG,
        )
        assert len(ds) == 4

    def test_all_fields_present(self, joint_data_dir: Path) -> None:
        ds = JointDataset(
            data_dir=joint_data_dir, max_seq_length=SEQ_LEN,
            embedding_dim=EMB_DIM, num_negatives=NUM_NEG,
        )
        example = ds[0]
        assert "input_ids" in example
        assert "attention_mask" in example
        assert "router_labels" in example
        assert "span_mask" in example
        assert "positive_embedding" in example
        assert "negative_embeddings" in example
        assert "token_labels" in example

    def test_shapes(self, joint_data_dir: Path) -> None:
        ds = JointDataset(
            data_dir=joint_data_dir, max_seq_length=SEQ_LEN,
            embedding_dim=EMB_DIM, num_negatives=NUM_NEG,
        )
        example = ds[0]
        assert example["input_ids"].shape == (SEQ_LEN,)
        assert example["router_labels"].shape == (SEQ_LEN,)
        assert example["span_mask"].shape == (SEQ_LEN,)
        assert example["positive_embedding"].shape == (EMB_DIM,)
        assert example["negative_embeddings"].shape == (NUM_NEG, EMB_DIM)
        assert example["token_labels"].shape == (SEQ_LEN,)

    def test_token_labels_dtype(self, joint_data_dir: Path) -> None:
        ds = JointDataset(
            data_dir=joint_data_dir, max_seq_length=SEQ_LEN,
            embedding_dim=EMB_DIM, num_negatives=NUM_NEG,
        )
        example = ds[0]
        assert example["token_labels"].dtype == torch.long

    def test_token_labels_padding(self, tmp_path: Path) -> None:
        """Token labels should pad with -100 (ignore_index)."""
        d = tmp_path / "joint_pad"
        d.mkdir()
        example = {
            "input_ids": list(range(8)),  # shorter than max_seq_length
            "router_labels": [0] * 8,
            "token_labels": list(range(8)),
            "positive_embedding": np.zeros(EMB_DIM).tolist(),
            "negative_embeddings": np.zeros((NUM_NEG, EMB_DIM)).tolist(),
        }
        with open(d / "test.json", "w") as f:
            json.dump(example, f)

        ds = JointDataset(data_dir=d, max_seq_length=16,
                          embedding_dim=EMB_DIM, num_negatives=NUM_NEG)
        item = ds[0]
        # Last 8 positions should be -100
        assert (item["token_labels"][8:] == -100).all()
        # First 8 should be actual labels
        assert (item["token_labels"][:8] != -100).all()

    def test_missing_token_labels_defaults_to_ignore(self, tmp_path: Path) -> None:
        """If token_labels is missing, should default to all -100."""
        d = tmp_path / "joint_no_tl"
        d.mkdir()
        example = {
            "input_ids": list(range(SEQ_LEN)),
            "router_labels": [0] * SEQ_LEN,
            "positive_embedding": np.zeros(EMB_DIM).tolist(),
            "negative_embeddings": np.zeros((NUM_NEG, EMB_DIM)).tolist(),
        }
        with open(d / "test.json", "w") as f:
            json.dump(example, f)

        ds = JointDataset(data_dir=d, max_seq_length=SEQ_LEN,
                          embedding_dim=EMB_DIM, num_negatives=NUM_NEG)
        item = ds[0]
        assert (item["token_labels"] == -100).all()


# ====================================================================
# SECTION 10 — Router Metrics
# ====================================================================


class TestComputeRouterMetrics:
    """Tests for _compute_router_metrics helper."""

    def test_perfect_predictions(self) -> None:
        logits = [torch.tensor([10.0, 10.0, -10.0, -10.0])]
        labels = [torch.tensor([1.0, 1.0, 0.0, 0.0])]
        m = _compute_router_metrics(logits, labels)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self) -> None:
        logits = [torch.tensor([-10.0, -10.0, 10.0, 10.0])]
        labels = [torch.tensor([1.0, 1.0, 0.0, 0.0])]
        m = _compute_router_metrics(logits, labels)
        assert m["accuracy"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)

    def test_mixed_predictions(self) -> None:
        # 2 correct out of 4
        logits = [torch.tensor([10.0, -10.0, -10.0, 10.0])]
        labels = [torch.tensor([1.0, 1.0, 0.0, 0.0])]
        m = _compute_router_metrics(logits, labels)
        assert m["accuracy"] == pytest.approx(0.5)
        # tp=1, fp=1, fn=1, tn=1
        assert m["tp"] == 1.0
        assert m["fp"] == 1.0
        assert m["fn"] == 1.0
        assert m["tn"] == 1.0

    def test_multiple_batches(self) -> None:
        logits = [
            torch.tensor([10.0, 10.0]),
            torch.tensor([10.0, -10.0]),
        ]
        labels = [
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, 0.0]),
        ]
        m = _compute_router_metrics(logits, labels)
        # tp=3, fp=0, fn=0, tn=1
        assert m["tp"] == 3.0


# ====================================================================
# SECTION 11 — Retrieval Metrics
# ====================================================================


class TestRetrievalMetrics:
    """Tests for _precision_at_k, _mean_reciprocal_rank, _compute_retrieval_metrics."""

    def test_precision_at_1_perfect(self) -> None:
        # Positive is most similar to query
        query = torch.randn(BATCH, EMB_DIM)
        positive = query + 0.01 * torch.randn(BATCH, EMB_DIM)  # Very similar
        negatives = torch.randn(BATCH, NUM_NEG, EMB_DIM)  # Random
        # Normalise so dot product = cosine sim
        query = torch.nn.functional.normalize(query, dim=-1)
        positive = torch.nn.functional.normalize(positive, dim=-1)
        negatives = torch.nn.functional.normalize(negatives, dim=-1)

        p1 = _precision_at_k(query, positive, negatives, k=1)
        assert p1 > 0.5  # Should often be 1.0 with near-identical pos

    def test_precision_at_k_range(self) -> None:
        query = torch.randn(BATCH, EMB_DIM)
        positive = torch.randn(BATCH, EMB_DIM)
        negatives = torch.randn(BATCH, NUM_NEG, EMB_DIM)

        p1 = _precision_at_k(query, positive, negatives, k=1)
        p5 = _precision_at_k(query, positive, negatives, k=5)
        assert 0.0 <= p1 <= 1.0
        assert 0.0 <= p5 <= 1.0
        assert p5 >= p1  # Larger k should give >= hit rate

    def test_mrr_range(self) -> None:
        query = torch.randn(BATCH, EMB_DIM)
        positive = torch.randn(BATCH, EMB_DIM)
        negatives = torch.randn(BATCH, NUM_NEG, EMB_DIM)

        mrr = _mean_reciprocal_rank(query, positive, negatives)
        assert 0.0 < mrr <= 1.0

    def test_compute_retrieval_metrics_keys(self) -> None:
        q = [torch.randn(BATCH, EMB_DIM)]
        p = [torch.randn(BATCH, EMB_DIM)]
        n = [torch.randn(BATCH, NUM_NEG, EMB_DIM)]

        m = _compute_retrieval_metrics(q, p, n)
        assert "precision_at_1" in m
        assert "precision_at_5" in m
        assert "precision_at_10" in m
        assert "mrr" in m

    def test_perfect_retrieval_mrr(self) -> None:
        query = torch.eye(BATCH, EMB_DIM)[:BATCH]
        positive = query.clone()  # Identical
        negatives = -query.unsqueeze(1).expand(-1, NUM_NEG, -1)  # Opposite
        mrr = _mean_reciprocal_rank(query, positive, negatives)
        assert mrr == pytest.approx(1.0, abs=1e-6)


# ====================================================================
# SECTION 12 — _build_deepspeed_config
# ====================================================================


class TestBuildDeepSpeedConfig:
    """Tests for DeepSpeed config auto-fill."""

    def test_auto_values_filled(self, default_config: FRLMConfig, monkeypatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "1")
        ds_dict = _build_deepspeed_config(
            config=default_config,
            micro_batch_size=4,
            gradient_accumulation_steps=8,
            total_steps=1000,
            warmup_steps=100,
        )

        assert ds_dict["train_micro_batch_size_per_gpu"] == 4
        assert ds_dict["gradient_accumulation_steps"] == 8
        assert isinstance(ds_dict["train_batch_size"], int)
        # world_size=1 → 4 * 8 * 1 = 32
        assert ds_dict["train_batch_size"] == 4 * 8 * 1

    def test_batch_size_scales_with_world_size(self, default_config: FRLMConfig, monkeypatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "4")
        ds_dict = _build_deepspeed_config(
            config=default_config,
            micro_batch_size=4,
            gradient_accumulation_steps=8,
            total_steps=1000,
            warmup_steps=100,
        )
        assert ds_dict["train_batch_size"] == 4 * 8 * 4

    def test_single_gpu_keeps_cpu_offload(self, default_config: FRLMConfig, monkeypatch) -> None:
        """Single-GPU run must keep optimizer on CPU to stay within VRAM budget."""
        monkeypatch.setenv("WORLD_SIZE", "1")
        ds_dict = _build_deepspeed_config(
            config=default_config,
            micro_batch_size=4,
            gradient_accumulation_steps=8,
            total_steps=1000,
            warmup_steps=100,
        )
        offload_dev = ds_dict["zero_optimization"]["offload_optimizer"]["device"]
        assert offload_dev == "cpu"

    def test_single_gpu_disables_pin_memory(self, default_config: FRLMConfig, monkeypatch) -> None:
        """Single-GPU run must disable pin_memory to avoid huge cudaHostRegister."""
        monkeypatch.setenv("WORLD_SIZE", "1")
        ds_dict = _build_deepspeed_config(
            config=default_config,
            micro_batch_size=4,
            gradient_accumulation_steps=8,
            total_steps=1000,
            warmup_steps=100,
        )
        pin_mem = ds_dict["zero_optimization"]["offload_optimizer"]["pin_memory"]
        assert pin_mem is False

    def test_activation_checkpointing_removed(self, default_config: FRLMConfig, monkeypatch) -> None:
        """activation_checkpointing section must be stripped (HuggingFace handles it)."""
        monkeypatch.setenv("WORLD_SIZE", "1")
        ds_dict = _build_deepspeed_config(
            config=default_config,
            micro_batch_size=4,
            gradient_accumulation_steps=8,
            total_steps=1000,
            warmup_steps=100,
        )
        assert "activation_checkpointing" not in ds_dict

    def test_multi_gpu_keeps_cpu_offload(self, default_config: FRLMConfig, monkeypatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "2")
        ds_dict = _build_deepspeed_config(
            config=default_config,
            micro_batch_size=4,
            gradient_accumulation_steps=8,
            total_steps=1000,
            warmup_steps=100,
        )
        offload_dev = ds_dict["zero_optimization"]["offload_optimizer"]["device"]
        assert offload_dev == "cpu"

    def test_optimizer_params_filled(self, default_config: FRLMConfig, monkeypatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "1")
        ds_dict = _build_deepspeed_config(
            config=default_config,
            micro_batch_size=4,
            gradient_accumulation_steps=8,
            total_steps=1000,
            warmup_steps=100,
        )

        opt_params = ds_dict["optimizer"]["params"]
        assert isinstance(opt_params["lr"], float)
        assert isinstance(opt_params["weight_decay"], float)
        assert opt_params["lr"] == default_config.training.joint.learning_rate

    def test_scheduler_params_filled(self, default_config: FRLMConfig, monkeypatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "1")
        ds_dict = _build_deepspeed_config(
            config=default_config,
            micro_batch_size=4,
            gradient_accumulation_steps=8,
            total_steps=1000,
            warmup_steps=100,
        )

        sched_params = ds_dict["scheduler"]["params"]
        assert sched_params["warmup_num_steps"] == 100
        assert sched_params["total_num_steps"] == 1000
        assert sched_params["warmup_max_lr"] == default_config.training.joint.learning_rate

    def test_zero_stage(self, default_config: FRLMConfig, monkeypatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "1")
        ds_dict = _build_deepspeed_config(
            config=default_config,
            micro_batch_size=4, gradient_accumulation_steps=8,
            total_steps=1000, warmup_steps=100,
        )
        assert ds_dict["zero_optimization"]["stage"] == 2

    def test_ensure_deepspeed_env_sets_defaults(self, monkeypatch) -> None:
        """_ensure_deepspeed_env populates single-process defaults."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("MASTER_ADDR", raising=False)
        monkeypatch.delenv("MASTER_PORT", raising=False)

        _ensure_deepspeed_env()

        assert os.environ["RANK"] == "0"
        assert os.environ["LOCAL_RANK"] == "0"
        assert os.environ["WORLD_SIZE"] == "1"
        assert os.environ["MASTER_ADDR"] == "localhost"
        assert os.environ["MASTER_PORT"] == "29500"
        assert os.environ.get("DS_SKIP_CUDA_CHECK") == "1"

    def test_ensure_deepspeed_env_preserves_existing(self, monkeypatch) -> None:
        """_ensure_deepspeed_env does not overwrite launcher-provided values."""
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")

        _ensure_deepspeed_env()

        assert os.environ["RANK"] == "3"
        assert os.environ["WORLD_SIZE"] == "8"
        assert os.environ["MASTER_ADDR"] == "10.0.0.1"


# ====================================================================
# SECTION 13 — RouterTrainer (unit-level)
# ====================================================================


class TestRouterTrainer:
    """Unit tests for RouterTrainer (no actual training)."""

    def _make_mock_model(self) -> MagicMock:
        """Create a mock FRLMModel with the needed attributes."""
        model = MagicMock(spec=FRLMModel)
        model.backbone = MagicMock()
        model.backbone.freeze = MagicMock()
        model.backbone.unfreeze = MagicMock()
        model.router = MagicMock()
        model.retrieval_head = MagicMock()
        model.generation_head = MagicMock()
        model.loss_fn = MagicMock()

        # Make parameters() return real parameters for optimizer
        real_param = nn.Linear(4, 2)
        model.router.parameters = real_param.parameters
        model.backbone.parameters = lambda: iter([])
        model.retrieval_head.parameters = lambda: iter([])
        model.generation_head.parameters = lambda: iter([])
        model.loss_fn.parameters = lambda: iter([])
        model.parameters = real_param.parameters

        return model

    def test_instantiation(self, default_config: FRLMConfig) -> None:
        model = self._make_mock_model()
        trainer = RouterTrainer(
            model=model, config=default_config, device="cpu",
        )
        assert trainer._device == torch.device("cpu")
        assert trainer._rcfg == default_config.training.router

    def test_freeze_non_router(self, default_config: FRLMConfig) -> None:
        model = self._make_mock_model()
        trainer = RouterTrainer(
            model=model, config=default_config, device="cpu",
        )
        trainer._freeze_non_router()
        # backbone should have freeze() called
        model.backbone.freeze.assert_called()

    def test_build_optimizer(self, default_config: FRLMConfig) -> None:
        model = self._make_mock_model()
        trainer = RouterTrainer(
            model=model, config=default_config, device="cpu",
        )
        optim = trainer._build_optimizer()
        assert optim is not None
        assert len(optim.param_groups) >= 1


# ====================================================================
# SECTION 14 — RetrievalTrainer (unit-level)
# ====================================================================


class TestRetrievalTrainer:
    """Unit tests for RetrievalTrainer."""

    def _make_mock_model(self) -> MagicMock:
        model = MagicMock(spec=FRLMModel)
        model.backbone = MagicMock()
        model.backbone.freeze = MagicMock()
        model.backbone.unfreeze = MagicMock()
        model.router = MagicMock()
        model.retrieval_head = MagicMock()
        model.generation_head = MagicMock()
        model.loss_fn = MagicMock()

        real_param = nn.Linear(4, 2)
        model.retrieval_head.parameters = real_param.parameters
        model.backbone.parameters = lambda: iter([])
        model.router.parameters = lambda: iter([])
        model.generation_head.parameters = lambda: iter([])
        model.loss_fn.parameters = lambda: iter([])
        model.parameters = real_param.parameters

        return model

    def test_instantiation(self, default_config: FRLMConfig) -> None:
        model = self._make_mock_model()
        trainer = RetrievalTrainer(
            model=model, config=default_config, device="cpu",
        )
        assert trainer._rcfg == default_config.training.retrieval

    def test_pool_query_l2_normalized(self) -> None:
        """_pool_query should return L2-normalised embeddings."""
        # hidden: (batch, seq, dim)
        hidden = torch.randn(BATCH, SEQ_LEN, EMB_DIM)
        span_mask = torch.zeros(BATCH, SEQ_LEN)
        span_mask[:, 0:4] = 1.0  # First 4 positions are retrieval

        pooled = RetrievalTrainer._pool_query(hidden, span_mask)
        norms = pooled.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)

    def test_pool_query_shape(self) -> None:
        hidden = torch.randn(BATCH, SEQ_LEN, EMB_DIM)
        span_mask = torch.ones(BATCH, SEQ_LEN)
        pooled = RetrievalTrainer._pool_query(hidden, span_mask)
        assert pooled.shape == (BATCH, EMB_DIM)

    def test_curriculum_temperature(self, default_config: FRLMConfig) -> None:
        model = self._make_mock_model()
        trainer = RetrievalTrainer(
            model=model, config=default_config, device="cpu",
        )
        target = default_config.loss.contrastive_temperature
        # At start (progress=0), temperature should be 2× target
        t0 = trainer._curriculum_temperature(0, 100)
        assert t0 == pytest.approx(2 * target, abs=1e-4)

        # At end (progress=total), temperature should be close to target
        t1 = trainer._curriculum_temperature(100, 100)
        assert t1 == pytest.approx(target, abs=1e-2)

        # Midway should be between
        t_mid = trainer._curriculum_temperature(50, 100)
        assert target < t_mid < 2 * target


# ====================================================================
# SECTION 15 — JointTrainer (unit-level)
# ====================================================================


class TestJointTrainer:
    """Unit tests for JointTrainer."""

    def _make_mock_model(self) -> MagicMock:
        model = MagicMock(spec=FRLMModel)
        model.backbone = MagicMock()
        model.backbone.freeze = MagicMock()
        model.backbone.unfreeze = MagicMock()
        model.router = MagicMock()
        model.retrieval_head = MagicMock()
        model.generation_head = MagicMock()
        model.loss_fn = MagicMock()

        real_param = nn.Linear(4, 2)
        model.parameters = real_param.parameters
        model.backbone.parameters = lambda: iter([])
        model.router.parameters = lambda: iter([])
        model.retrieval_head.parameters = lambda: iter([])
        model.generation_head.parameters = lambda: iter([])
        model.loss_fn.parameters = lambda: iter([])

        return model

    def test_instantiation(self, default_config: FRLMConfig) -> None:
        model = self._make_mock_model()
        trainer = JointTrainer(
            model=model, config=default_config, device="cpu",
            use_deepspeed=False,
        )
        assert trainer._jcfg == default_config.training.joint
        assert not trainer._use_deepspeed

    def test_deepspeed_default_from_config(self, default_config: FRLMConfig) -> None:
        model = self._make_mock_model()
        trainer = JointTrainer(
            model=model, config=default_config, device="cpu",
        )
        # Should match config value
        assert trainer._use_deepspeed == default_config.deepspeed.enabled

    def test_deepspeed_override(self, default_config: FRLMConfig) -> None:
        model = self._make_mock_model()
        trainer = JointTrainer(
            model=model, config=default_config, device="cpu",
            use_deepspeed=False,
        )
        assert not trainer._use_deepspeed

    def test_configure_params(self, default_config: FRLMConfig) -> None:
        model = self._make_mock_model()
        trainer = JointTrainer(
            model=model, config=default_config, device="cpu",
            use_deepspeed=False,
        )
        trainer._configure_params()
        # Backbone should be unfrozen by default
        model.backbone.unfreeze.assert_called()


# ====================================================================
# SECTION 16 — init_wandb / finish_wandb
# ====================================================================


class TestWandBHelpers:
    """Tests for safe WandB init/finish."""

    def test_disabled(self) -> None:
        run = init_wandb(
            project="test", run_name="test", tags=[], config={}, enabled=False,
        )
        assert run is None

    def test_import_error(self) -> None:
        """When wandb is not installed, should return None."""
        with patch.dict("sys.modules", {"wandb": None}):
            # Force re-import to fail
            original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
            def _mock_import(name, *args, **kwargs):
                if name == "wandb":
                    raise ImportError("No module named 'wandb'")
                return original_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=_mock_import):
                run = init_wandb(
                    project="test", run_name="test", tags=[], config={}, enabled=True,
                )
                assert run is None

    def test_finish_wandb_none(self) -> None:
        """Should not raise when run is None."""
        finish_wandb(None)

    def test_finish_wandb_mock(self) -> None:
        mock_run = MagicMock()
        finish_wandb(mock_run)
        mock_run.finish.assert_called_once()


# ====================================================================
# SECTION 17 — DeepSpeed Config JSON
# ====================================================================


class TestDeepSpeedConfigJSON:
    """Validate the config/deepspeed_config.json file."""

    def test_file_exists(self) -> None:
        path = Path(__file__).resolve().parent.parent / "config" / "deepspeed_config.json"
        assert path.exists(), "config/deepspeed_config.json should exist"

    def test_valid_json(self) -> None:
        path = Path(__file__).resolve().parent.parent / "config" / "deepspeed_config.json"
        with open(path) as f:
            cfg = json.load(f)
        assert isinstance(cfg, dict)

    def test_zero_stage_2(self) -> None:
        path = Path(__file__).resolve().parent.parent / "config" / "deepspeed_config.json"
        with open(path) as f:
            cfg = json.load(f)
        assert cfg["zero_optimization"]["stage"] == 2

    def test_fp16_enabled(self) -> None:
        path = Path(__file__).resolve().parent.parent / "config" / "deepspeed_config.json"
        with open(path) as f:
            cfg = json.load(f)
        assert cfg["fp16"]["enabled"] is True

    def test_auto_placeholders(self) -> None:
        path = Path(__file__).resolve().parent.parent / "config" / "deepspeed_config.json"
        with open(path) as f:
            cfg = json.load(f)
        # These should be "auto" for dynamic filling
        assert cfg["train_batch_size"] == "auto"
        assert cfg["train_micro_batch_size_per_gpu"] == "auto"
        assert cfg["gradient_accumulation_steps"] == "auto"
        assert cfg["optimizer"]["params"]["lr"] == "auto"
        assert cfg["scheduler"]["params"]["warmup_num_steps"] == "auto"
        assert cfg["scheduler"]["params"]["total_num_steps"] == "auto"

    def test_overlap_comm(self) -> None:
        path = Path(__file__).resolve().parent.parent / "config" / "deepspeed_config.json"
        with open(path) as f:
            cfg = json.load(f)
        assert cfg["zero_optimization"]["overlap_comm"] is True

    def test_activation_checkpointing(self) -> None:
        path = Path(__file__).resolve().parent.parent / "config" / "deepspeed_config.json"
        with open(path) as f:
            cfg = json.load(f)
        assert cfg["activation_checkpointing"]["partition_activations"] is False


# ====================================================================
# SECTION 18 — Training __init__ exports
# ====================================================================


class TestTrainingModuleExports:
    """Verify that all expected symbols are exported from src.training."""

    def test_trainer_exports(self) -> None:
        from src.training import RouterTrainer, RetrievalTrainer, JointTrainer
        assert RouterTrainer is not None
        assert RetrievalTrainer is not None
        assert JointTrainer is not None

    def test_dataset_exports(self) -> None:
        from src.training import RouterDataset, RetrievalDataset, JointDataset
        assert RouterDataset is not None
        assert RetrievalDataset is not None
        assert JointDataset is not None

    def test_utils_exports(self) -> None:
        from src.training import (
            TrainingState, CheckpointManager, MetricsLogger,
            EarlyStopping, GradientAccumulator, LearningRateScheduler,
            init_wandb, finish_wandb,
        )
        assert TrainingState is not None
        assert CheckpointManager is not None
        assert MetricsLogger is not None
        assert EarlyStopping is not None
        assert GradientAccumulator is not None
        assert LearningRateScheduler is not None
        assert init_wandb is not None
        assert finish_wandb is not None

    def test_all_list(self) -> None:
        import src.training as training_mod
        expected = {
            "RouterTrainer", "RetrievalTrainer", "JointTrainer",
            "RouterDataset", "RetrievalDataset", "JointDataset",
            "TrainingState", "CheckpointManager", "MetricsLogger",
            "EarlyStopping", "GradientAccumulator", "LearningRateScheduler",
            "init_wandb", "finish_wandb",
        }
        assert expected.issubset(set(training_mod.__all__))


# ====================================================================
# SECTION 19 — Integration: Checkpoint save/load round-trip
# ====================================================================


class TestCheckpointIntegration:
    """End-to-end checkpoint save → load → verify."""

    def test_full_round_trip(self, tmp_dir: Path) -> None:
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9)

        # Take a few optimizer steps
        x = torch.randn(2, 8)
        for _ in range(3):
            loss = model(x).sum()
            loss.backward()
            optim.step()
            optim.zero_grad()
            sched.step()

        state = TrainingState(epoch=3, global_step=300, best_metric=0.92)
        mgr = CheckpointManager(tmp_dir / "integration_ckpt")
        path = mgr.save(model, optim, sched, state, metrics={"acc": 0.92})

        # Restore into new instances
        model2 = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        optim2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        sched2 = torch.optim.lr_scheduler.StepLR(optim2, step_size=1, gamma=0.9)

        restored_state = mgr.load(model2, optim2, sched2, checkpoint_path=path)
        assert restored_state.epoch == 3
        assert restored_state.global_step == 300
        assert restored_state.best_metric == 0.92

        # Weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


# ====================================================================
# SECTION 20 — Config validation for training
# ====================================================================


class TestTrainingConfigValidation:
    """Config-level checks for training settings."""

    def test_router_training_defaults(self, default_config: FRLMConfig) -> None:
        r = default_config.training.router
        assert r.learning_rate > 0
        assert r.epochs > 0
        assert r.batch_size > 0
        assert r.scheduler in ("linear", "cosine", "cosine_with_restarts")
        assert 0 < r.warmup_ratio < 1

    def test_retrieval_training_defaults(self, default_config: FRLMConfig) -> None:
        r = default_config.training.retrieval
        assert r.learning_rate > 0
        assert r.epochs > 0
        assert r.scheduler in ("linear", "cosine", "cosine_with_restarts")

    def test_joint_training_defaults(self, default_config: FRLMConfig) -> None:
        j = default_config.training.joint
        assert j.learning_rate > 0
        assert j.epochs > 0
        assert j.scheduler in ("linear", "cosine", "cosine_with_restarts")
        assert j.num_cycles >= 1

    def test_loss_config(self, default_config: FRLMConfig) -> None:
        l = default_config.loss
        assert l.router_weight > 0
        assert l.retrieval_weight > 0
        assert l.generation_weight > 0
        assert l.contrastive_temperature > 0

    def test_deepspeed_config(self, default_config: FRLMConfig) -> None:
        ds = default_config.deepspeed
        assert isinstance(ds.enabled, bool)
        assert ds.config.zero_optimization.stage == 2

    def test_wandb_config(self, default_config: FRLMConfig) -> None:
        w = default_config.wandb
        assert w.project == "frlm"
        assert len(w.tags) > 0

    def test_splits_sum_to_one(self, default_config: FRLMConfig) -> None:
        s = default_config.training.splits
        assert abs(s.train + s.validation + s.test - 1.0) < 1e-6


# ====================================================================
# SECTION 21 — Edge cases
# ====================================================================


class TestEdgeCases:
    """Boundary and edge case tests."""

    def test_gradient_accumulator_steps_one(self) -> None:
        ga = GradientAccumulator(accumulation_steps=1)
        assert ga.should_step()  # Always steps immediately
        assert ga.micro_step == 1  # Not reset until step() is called
        ga.reset()
        assert ga.micro_step == 0

    def test_early_stopping_patience_zero_behavior(self) -> None:
        """With patience=1, should stop after one non-improvement."""
        es = EarlyStopping(patience=1, mode="min")
        es.step(0.5)
        result = es.step(0.6)
        assert result is True

    def test_lr_scheduler_total_steps_one(self) -> None:
        optim = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = LearningRateScheduler(optim, total_steps=1, warmup_steps=0)
        lrs = scheduler.get_lr()
        assert all(lr >= 0 for lr in lrs)

    def test_checkpoint_manager_max_one(self, tmp_dir: Path) -> None:
        """Max 1 checkpoint should still work."""
        model = nn.Linear(4, 2)
        mgr = CheckpointManager(tmp_dir / "ckpts", max_checkpoints=1)
        import time
        for i in range(3):
            mgr.save(model, None, None, TrainingState(global_step=i))
            time.sleep(0.05)
        assert len(mgr.list_checkpoints()) == 1

    def test_router_dataset_with_long_sequence(self, router_data_dir: Path) -> None:
        """Max seq length smaller than data should truncate."""
        ds = RouterDataset(data_dir=router_data_dir, max_seq_length=4)
        item = ds[0]
        assert item["input_ids"].shape == (4,)

    def test_metrics_logger_no_steps_summary(self) -> None:
        logger = MetricsLogger(wandb_run=None, prefix="test")
        s = logger.summary()
        assert s == {}

    def test_training_state_from_empty_dict(self) -> None:
        state = TrainingState.from_dict({})
        assert state.epoch == 0
        assert state.global_step == 0
