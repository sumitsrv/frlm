"""
Tests for the full pipeline configuration.

Tests config loading end-to-end, path resolution, training config,
evaluation config, and serving config.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, PathsConfig, load_config


@pytest.fixture(scope="module")
def default_config() -> FRLMConfig:
    return load_config()


# ===========================================================================
# Config loading E2E
# ===========================================================================


class TestConfigLoadingE2E:
    """Full config loading end-to-end tests."""

    def test_load_returns_complete_config(self, default_config: FRLMConfig) -> None:
        assert default_config is not None
        assert isinstance(default_config, FRLMConfig)

    def test_config_is_frozen_after_load(self, default_config: FRLMConfig) -> None:
        """Config values should be stable across accesses."""
        seed1 = default_config.project.seed
        seed2 = default_config.project.seed
        assert seed1 == seed2

    def test_nested_access_chain(self, default_config: FRLMConfig) -> None:
        """Deeply nested access should work without errors."""
        tau = default_config.loss.contrastive_temperature
        assert isinstance(tau, float)
        assert tau > 0

        mode_names = default_config.model.retrieval_head.temporal.mode_names
        assert isinstance(mode_names, list)
        assert len(mode_names) == 3


# ===========================================================================
# Path resolution
# ===========================================================================


class TestPathResolution:
    """Test that all configured paths can be resolved."""

    def test_all_path_fields_resolve(self, default_config: FRLMConfig) -> None:
        paths = default_config.paths
        for field_name in type(paths).model_fields:
            resolved = paths.resolve(field_name)
            assert isinstance(resolved, Path), f"Failed to resolve {field_name}"
            assert resolved.is_absolute(), f"{field_name} did not resolve to absolute path"

    def test_corpus_dir_is_under_data(self, default_config: FRLMConfig) -> None:
        corpus = default_config.paths.resolve("corpus_dir")
        data = default_config.paths.resolve("data_dir")
        assert str(corpus).startswith(str(data))

    def test_processed_dir_is_under_data(self, default_config: FRLMConfig) -> None:
        processed = default_config.paths.resolve("processed_dir")
        data = default_config.paths.resolve("data_dir")
        assert str(processed).startswith(str(data))

    def test_labels_dir_is_under_data(self, default_config: FRLMConfig) -> None:
        labels = default_config.paths.resolve("labels_dir")
        data = default_config.paths.resolve("data_dir")
        assert str(labels).startswith(str(data))


# ===========================================================================
# Training config
# ===========================================================================


class TestTrainingConfig:
    """Validate training pipeline configuration."""

    def test_three_phases(self, default_config: FRLMConfig) -> None:
        t = default_config.training
        assert hasattr(t, "router")
        assert hasattr(t, "retrieval")
        assert hasattr(t, "joint")

    def test_router_phase_config(self, default_config: FRLMConfig) -> None:
        r = default_config.training.router
        assert r.epochs > 0
        assert r.batch_size > 0
        assert r.learning_rate > 0
        assert r.freeze_backbone is True
        assert r.early_stopping_metric == "f1"

    def test_retrieval_phase_config(self, default_config: FRLMConfig) -> None:
        r = default_config.training.retrieval
        assert r.epochs > 0
        assert r.contrastive_temperature == 0.07
        assert r.freeze_router is True
        assert r.early_stopping_metric == "precision_at_1"

    def test_joint_phase_config(self, default_config: FRLMConfig) -> None:
        j = default_config.training.joint
        assert j.epochs > 0
        assert j.freeze_backbone is False
        assert j.freeze_router is False
        assert j.freeze_retrieval is False
        assert j.early_stopping_metric == "combined_loss"

    def test_data_splits(self, default_config: FRLMConfig) -> None:
        s = default_config.training.splits
        assert abs(s.train + s.validation + s.test - 1.0) < 1e-6
        assert s.train == 0.8
        assert s.validation == 0.1
        assert s.test == 0.1

    def test_gradient_accumulation(self, default_config: FRLMConfig) -> None:
        assert default_config.training.gradient_accumulation_steps == 8

    def test_gpu_id_default(self, default_config: FRLMConfig) -> None:
        """training.gpu_id should exist and default to None (auto-select)."""
        assert hasattr(default_config.training, "gpu_id")
        assert default_config.training.gpu_id is None

    def test_max_grad_norm(self, default_config: FRLMConfig) -> None:
        assert default_config.training.max_grad_norm == 1.0

    def test_checkpoint_rotation(self, default_config: FRLMConfig) -> None:
        assert default_config.training.max_checkpoints == 5


# ===========================================================================
# Evaluation config
# ===========================================================================


class TestEvaluationConfig:
    """Validate evaluation configuration."""

    def test_retrieval_k_values(self, default_config: FRLMConfig) -> None:
        k_values = default_config.evaluation.retrieval.k_values
        assert k_values == [1, 5, 10, 20]
        assert all(k > 0 for k in k_values)
        assert k_values == sorted(k_values)

    def test_router_threshold_sweep(self, default_config: FRLMConfig) -> None:
        thresholds = default_config.evaluation.router.threshold_sweep
        assert len(thresholds) > 0
        assert all(0.0 < t < 1.0 for t in thresholds)

    def test_eval_sample_sizes(self, default_config: FRLMConfig) -> None:
        assert default_config.evaluation.retrieval.num_eval_samples > 0
        assert default_config.evaluation.generation.num_eval_samples > 0
        assert default_config.evaluation.router.num_eval_samples > 0
        assert default_config.evaluation.end_to_end.num_eval_samples > 0

    def test_perplexity_enabled(self, default_config: FRLMConfig) -> None:
        assert default_config.evaluation.generation.compute_perplexity is True


# ===========================================================================
# Serving config
# ===========================================================================


class TestServingConfig:
    """Validate FastAPI serving configuration."""

    def test_host_and_port(self, default_config: FRLMConfig) -> None:
        assert default_config.serving.host == "0.0.0.0"
        assert default_config.serving.port == 8000

    def test_cors_origins(self, default_config: FRLMConfig) -> None:
        assert "*" in default_config.serving.cors_origins

    def test_timeout(self, default_config: FRLMConfig) -> None:
        assert default_config.serving.request_timeout == 60

    def test_warmup(self, default_config: FRLMConfig) -> None:
        assert default_config.serving.model_warmup is True

    def test_workers(self, default_config: FRLMConfig) -> None:
        assert default_config.serving.workers >= 1


# ===========================================================================
# DeepSpeed config
# ===========================================================================


class TestDeepSpeedConfig:
    """Validate DeepSpeed configuration."""

    def test_enabled(self, default_config: FRLMConfig) -> None:
        assert default_config.deepspeed.enabled is True

    def test_zero_stage(self, default_config: FRLMConfig) -> None:
        assert default_config.deepspeed.config.zero_optimization.stage == 2

    def test_fp16(self, default_config: FRLMConfig) -> None:
        assert default_config.deepspeed.config.fp16.enabled is True

    def test_optimizer_type(self, default_config: FRLMConfig) -> None:
        assert default_config.deepspeed.config.optimizer.type == "AdamW"

    def test_gradient_clipping(self, default_config: FRLMConfig) -> None:
        assert default_config.deepspeed.config.gradient_clipping == 1.0