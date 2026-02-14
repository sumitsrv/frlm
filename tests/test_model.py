"""
Tests for model architecture configuration.

Tests backbone, router head, retrieval head sub-heads, generation head,
and loss weight configuration consistency.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, load_config


@pytest.fixture(scope="module")
def default_config() -> FRLMConfig:
    return load_config()


# ===========================================================================
# Backbone config
# ===========================================================================


class TestBackboneConfig:
    """Validate BioMedLM backbone configuration."""

    def test_model_name(self, default_config: FRLMConfig) -> None:
        assert default_config.model.backbone.name == "stanford-crfm/BioMedLM"

    def test_hidden_dim(self, default_config: FRLMConfig) -> None:
        assert default_config.model.backbone.hidden_dim == 2560

    def test_num_layers_and_heads(self, default_config: FRLMConfig) -> None:
        bb = default_config.model.backbone
        assert bb.num_layers == 32
        assert bb.num_heads == 32
        # hidden_dim should be divisible by num_heads
        assert bb.hidden_dim % bb.num_heads == 0

    def test_vocab_size(self, default_config: FRLMConfig) -> None:
        assert default_config.model.backbone.vocab_size == 50257

    def test_max_seq_length(self, default_config: FRLMConfig) -> None:
        assert default_config.model.backbone.max_seq_length == 1024

    def test_gradient_checkpointing(self, default_config: FRLMConfig) -> None:
        assert default_config.model.backbone.gradient_checkpointing is True


# ===========================================================================
# Router head config
# ===========================================================================


class TestRouterHeadConfig:
    """Validate router head configuration."""

    def test_architecture(self, default_config: FRLMConfig) -> None:
        rh = default_config.model.router_head
        # Linear(hidden_dim, 256) -> ReLU -> Linear(256, 1)
        assert rh.input_dim == 2560
        assert rh.hidden_dim == 256
        assert rh.output_dim == 1

    def test_activation(self, default_config: FRLMConfig) -> None:
        assert default_config.model.router_head.activation == "relu"

    def test_dropout(self, default_config: FRLMConfig) -> None:
        assert 0.0 <= default_config.model.router_head.dropout < 1.0

    def test_threshold(self, default_config: FRLMConfig) -> None:
        assert default_config.model.router_head.threshold == 0.5


# ===========================================================================
# Retrieval head sub-heads
# ===========================================================================


class TestRetrievalHeadConfig:
    """Validate retrieval head sub-head configurations."""

    def test_semantic_projects_to_sapbert_space(self, default_config: FRLMConfig) -> None:
        sem = default_config.model.retrieval_head.semantic
        assert sem.input_dim == default_config.model.backbone.hidden_dim
        assert sem.output_dim == default_config.sapbert.embedding_dim
        assert sem.normalize is True

    def test_granularity_four_levels(self, default_config: FRLMConfig) -> None:
        gran = default_config.model.retrieval_head.granularity
        assert gran.input_dim == default_config.model.backbone.hidden_dim
        assert gran.num_levels == 4
        assert gran.level_names == ["atomic", "relation", "entity", "cluster"]

    def test_temporal_three_modes(self, default_config: FRLMConfig) -> None:
        temp = default_config.model.retrieval_head.temporal
        assert temp.input_dim == default_config.model.backbone.hidden_dim
        assert temp.num_modes == 3
        assert temp.mode_names == ["CURRENT", "AT_TIMESTAMP", "HISTORY"]

    def test_all_input_dims_match_backbone(self, default_config: FRLMConfig) -> None:
        hd = default_config.model.backbone.hidden_dim
        rh = default_config.model.retrieval_head
        assert rh.semantic.input_dim == hd
        assert rh.granularity.input_dim == hd
        assert rh.temporal.input_dim == hd


# ===========================================================================
# Generation head config
# ===========================================================================


class TestGenerationHeadConfig:
    """Validate generation head configuration."""

    def test_input_dim(self, default_config: FRLMConfig) -> None:
        assert default_config.model.generation_head.input_dim == default_config.model.backbone.hidden_dim

    def test_output_dim_matches_vocab(self, default_config: FRLMConfig) -> None:
        assert default_config.model.generation_head.output_dim == default_config.model.backbone.vocab_size

    def test_tie_weights(self, default_config: FRLMConfig) -> None:
        assert default_config.model.generation_head.tie_weights is True


# ===========================================================================
# Loss config
# ===========================================================================


class TestLossConfig:
    """Validate combined loss configuration."""

    def test_weights(self, default_config: FRLMConfig) -> None:
        loss = default_config.loss
        assert loss.router_weight == 1.0
        assert loss.retrieval_weight == 2.0
        assert loss.generation_weight == 1.0

    def test_temperature(self, default_config: FRLMConfig) -> None:
        assert default_config.loss.contrastive_temperature == 0.07
        assert default_config.loss.contrastive_temperature > 0

    def test_combined_formula(self, default_config: FRLMConfig) -> None:
        """L_total = 1.0*L_r + 2.0*L_ret + 1.0*L_gen."""
        loss = default_config.loss
        l_r, l_ret, l_gen = 0.5, 1.0, 2.0
        expected = loss.router_weight * l_r + loss.retrieval_weight * l_ret + loss.generation_weight * l_gen
        assert expected == 1.0 * 0.5 + 2.0 * 1.0 + 1.0 * 2.0
        assert expected == 4.5

    def test_retrieval_has_highest_weight(self, default_config: FRLMConfig) -> None:
        loss = default_config.loss
        assert loss.retrieval_weight > loss.router_weight
        assert loss.retrieval_weight > loss.generation_weight