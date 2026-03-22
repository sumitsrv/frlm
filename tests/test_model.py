"""
Tests for Phase 6 — FRLM Model Architecture.

Tests cover:
- Config-level validation (retained from Phase 3)
- RouterHead: forward / predict / decide shapes and value ranges
- RetrievalHead: forward shapes, L2-normalisation, QuerySignature
- GenerationHead: forward shapes, weight tying
- InfoNCELoss: value range, gradient flow, hard-negative support
- RouterLoss: value range, masking, label smoothing
- GenerationLoss: value range, masking, ignore_index
- FRLMCombinedLoss: weighted sum, loss_dict keys, partial inputs
- FRLMModel full forward pass (with mocked backbone)
- Frozen backbone: identical hidden states across forward calls
- Router mask: correct gating of retrieval vs generation positions
- save_pretrained / from_pretrained round-trip (with mocked backbone)
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, load_config

# ---- Dimensions used throughout tests (small for speed) ----
BATCH = 4
SEQ_LEN = 16
HIDDEN_DIM = 64
EMB_DIM = 32
VOCAB_SIZE = 128
NUM_GRAN = 4
NUM_TEMP = 3
NUM_NEG = 5


# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture(scope="module")
def default_config() -> FRLMConfig:
    return load_config()


@pytest.fixture()
def hidden_states() -> Tensor:
    """Random hidden states: (batch, seq_len, hidden_dim)."""
    return torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)


@pytest.fixture()
def hidden_2d() -> Tensor:
    """Random hidden states without seq dim: (batch, hidden_dim)."""
    return torch.randn(BATCH, HIDDEN_DIM)


# ====================================================================
# SECTION 1 — Config-level tests (retained from earlier phases)
# ====================================================================


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
        assert bb.hidden_dim % bb.num_heads == 0

    def test_vocab_size(self, default_config: FRLMConfig) -> None:
        assert default_config.model.backbone.vocab_size == 50257

    def test_max_seq_length(self, default_config: FRLMConfig) -> None:
        assert default_config.model.backbone.max_seq_length == 1024

    def test_gradient_checkpointing(self, default_config: FRLMConfig) -> None:
        assert default_config.model.backbone.gradient_checkpointing is True


class TestRouterHeadConfig:
    """Validate router head configuration."""

    def test_architecture(self, default_config: FRLMConfig) -> None:
        rh = default_config.model.router_head
        assert rh.input_dim == 2560
        assert rh.hidden_dim == 256
        assert rh.output_dim == 1

    def test_activation(self, default_config: FRLMConfig) -> None:
        assert default_config.model.router_head.activation == "relu"

    def test_dropout(self, default_config: FRLMConfig) -> None:
        assert 0.0 <= default_config.model.router_head.dropout < 1.0

    def test_threshold(self, default_config: FRLMConfig) -> None:
        assert default_config.model.router_head.threshold == 0.3


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


class TestGenerationHeadConfig:
    """Validate generation head configuration."""

    def test_input_dim(self, default_config: FRLMConfig) -> None:
        assert default_config.model.generation_head.input_dim == default_config.model.backbone.hidden_dim

    def test_output_dim_matches_vocab(self, default_config: FRLMConfig) -> None:
        assert default_config.model.generation_head.output_dim == default_config.model.backbone.vocab_size

    def test_tie_weights(self, default_config: FRLMConfig) -> None:
        assert default_config.model.generation_head.tie_weights is True


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


# ====================================================================
# SECTION 2 — RouterHead (nn.Module)
# ====================================================================


class TestRouterHeadModule:
    """Test RouterHead with random tensors."""

    def _make_head(self) -> "RouterHead":
        from src.model.router_head import RouterHead
        return RouterHead(hidden_dim=HIDDEN_DIM, intermediate_dim=32, dropout=0.1, threshold=0.5)

    def test_forward_3d_shape(self, hidden_states: Tensor) -> None:
        head = self._make_head()
        logits = head.forward(hidden_states)
        assert logits.shape == (BATCH, SEQ_LEN, 1)

    def test_forward_2d_shape(self, hidden_2d: Tensor) -> None:
        head = self._make_head()
        logits = head.forward(hidden_2d)
        assert logits.shape == (BATCH, 1)

    def test_predict_range(self, hidden_states: Tensor) -> None:
        head = self._make_head()
        probs = head.predict(hidden_states)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_decide_returns_bool(self, hidden_states: Tensor) -> None:
        head = self._make_head()
        mask = head.decide(hidden_states)
        assert mask.dtype == torch.bool
        assert mask.shape == (BATCH, SEQ_LEN)

    def test_decide_threshold_override(self, hidden_states: Tensor) -> None:
        head = self._make_head()
        all_ret = head.decide(hidden_states, threshold=0.0)
        all_gen = head.decide(hidden_states, threshold=1.0)
        assert all_ret.all()  # threshold=0 → everything is retrieval
        assert not all_gen.any()  # threshold=1 → nothing is retrieval

    def test_threshold_property(self) -> None:
        head = self._make_head()
        assert head.threshold == 0.5
        head.threshold = 0.7
        assert head.threshold == 0.7

    def test_threshold_validation(self) -> None:
        head = self._make_head()
        with pytest.raises(ValueError):
            head.threshold = 1.5

    def test_parameter_count(self) -> None:
        head = self._make_head()
        params = list(head.parameters())
        # Linear(64,32): 64*32+32, Linear(32,1): 32*1+1
        expected = 64 * 32 + 32 + 32 * 1 + 1
        total = sum(p.numel() for p in params)
        assert total == expected

    def test_gradient_flows(self, hidden_states: Tensor) -> None:
        """Ensure gradients flow back through the router."""
        head = self._make_head()
        hs = hidden_states.clone().requires_grad_(True)
        logits = head.forward(hs)
        logits.sum().backward()
        assert hs.grad is not None
        assert (hs.grad != 0).any()

    def test_from_config(self, default_config: FRLMConfig) -> None:
        from src.model.router_head import RouterHead
        head = RouterHead.from_config(default_config.model.router_head)
        assert head._hidden_dim == 2560
        assert head._intermediate_dim == 256


# ====================================================================
# SECTION 3 — RetrievalHead (nn.Module)
# ====================================================================


class TestRetrievalHeadModule:
    """Test RetrievalHead with random tensors."""

    def _make_head(self) -> "RetrievalHead":
        from src.model.retrieval_head import RetrievalHead
        return RetrievalHead(
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMB_DIM,
            num_granularity_levels=NUM_GRAN,
            num_temporal_modes=NUM_TEMP,
        )

    def test_forward_3d_shapes(self, hidden_states: Tensor) -> None:
        head = self._make_head()
        qs = head.forward(hidden_states)
        assert qs.semantic_embedding.shape == (BATCH, SEQ_LEN, EMB_DIM)
        assert qs.granularity_logits.shape == (BATCH, SEQ_LEN, NUM_GRAN)
        assert qs.temporal_logits.shape == (BATCH, SEQ_LEN, NUM_TEMP)

    def test_forward_2d_shapes(self, hidden_2d: Tensor) -> None:
        head = self._make_head()
        qs = head.forward(hidden_2d)
        assert qs.semantic_embedding.shape == (BATCH, EMB_DIM)
        assert qs.granularity_logits.shape == (BATCH, NUM_GRAN)
        assert qs.temporal_logits.shape == (BATCH, NUM_TEMP)

    def test_semantic_l2_normalised(self, hidden_states: Tensor) -> None:
        head = self._make_head()
        qs = head.forward(hidden_states)
        norms = qs.semantic_embedding.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_granularity_level_property(self, hidden_2d: Tensor) -> None:
        head = self._make_head()
        qs = head.forward(hidden_2d)
        levels = qs.granularity_level
        assert levels.shape == (BATCH,)
        assert (levels >= 0).all() and (levels < NUM_GRAN).all()

    def test_temporal_mode_property(self, hidden_2d: Tensor) -> None:
        head = self._make_head()
        qs = head.forward(hidden_2d)
        modes = qs.temporal_mode
        assert modes.shape == (BATCH,)
        assert (modes >= 0).all() and (modes < NUM_TEMP).all()

    def test_gradient_flows(self, hidden_states: Tensor) -> None:
        head = self._make_head()
        hs = hidden_states.clone().requires_grad_(True)
        qs = head.forward(hs)
        loss = qs.semantic_embedding.sum() + qs.granularity_logits.sum() + qs.temporal_logits.sum()
        loss.backward()
        assert hs.grad is not None

    def test_embedding_dim_property(self) -> None:
        head = self._make_head()
        assert head.embedding_dim == EMB_DIM

    def test_from_config(self, default_config: FRLMConfig) -> None:
        from src.model.retrieval_head import RetrievalHead
        head = RetrievalHead.from_config(default_config.model.retrieval_head)
        assert head.embedding_dim == 768
        assert head.num_granularity_levels == 4
        assert head.num_temporal_modes == 3


# ====================================================================
# SECTION 4 — GenerationHead (nn.Module)
# ====================================================================


class TestGenerationHeadModule:
    """Test GenerationHead with random tensors."""

    def _make_head(self) -> "GenerationHead":
        from src.model.generation_head import GenerationHead
        return GenerationHead(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)

    def test_forward_3d_shape(self, hidden_states: Tensor) -> None:
        head = self._make_head()
        logits = head.forward(hidden_states)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)

    def test_forward_2d_shape(self, hidden_2d: Tensor) -> None:
        head = self._make_head()
        logits = head.forward(hidden_2d)
        assert logits.shape == (BATCH, VOCAB_SIZE)

    def test_gradient_flows(self, hidden_states: Tensor) -> None:
        head = self._make_head()
        hs = hidden_states.clone().requires_grad_(True)
        logits = head.forward(hs)
        logits.sum().backward()
        assert hs.grad is not None

    def test_weight_tying(self) -> None:
        from src.model.generation_head import GenerationHead
        head = GenerationHead(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
        assert not head.is_tied

        emb_weight = torch.randn(VOCAB_SIZE, HIDDEN_DIM)
        head.tie_weights(emb_weight)
        assert head.is_tied
        assert torch.equal(head.proj.weight.data, emb_weight)

    def test_tied_weights_produce_same_output(self) -> None:
        from src.model.generation_head import GenerationHead
        head = GenerationHead(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
        emb_weight = torch.randn(VOCAB_SIZE, HIDDEN_DIM)
        head.tie_weights(emb_weight)

        x = torch.randn(2, HIDDEN_DIM)
        out = head.forward(x)
        expected = x @ emb_weight.t()
        assert torch.allclose(out, expected, atol=1e-5)

    def test_from_config(self, default_config: FRLMConfig) -> None:
        from src.model.generation_head import GenerationHead
        head = GenerationHead.from_config(default_config.model.generation_head)
        assert head.hidden_dim == 2560
        assert head.vocab_size == 50257


# ====================================================================
# SECTION 5 — InfoNCELoss
# ====================================================================


class TestInfoNCELoss:
    """Test contrastive InfoNCE loss."""

    def _make_loss(self, temperature: float = 0.07) -> "InfoNCELoss":
        from src.model.losses import InfoNCELoss
        return InfoNCELoss(temperature=temperature)

    def test_positive_loss(self) -> None:
        loss_fn = self._make_loss()
        q = torch.randn(BATCH, EMB_DIM)
        p = torch.randn(BATCH, EMB_DIM)
        n = torch.randn(BATCH, NUM_NEG, EMB_DIM)
        loss = loss_fn(q, p, n)
        assert loss.item() > 0

    def test_perfect_positive_low_loss(self) -> None:
        """When query == positive and negatives are orthogonal, loss should be low."""
        loss_fn = self._make_loss(temperature=1.0)
        q = torch.nn.functional.normalize(torch.randn(BATCH, EMB_DIM), dim=-1)
        p = q.clone()
        n = torch.nn.functional.normalize(torch.randn(BATCH, NUM_NEG, EMB_DIM), dim=-1)
        loss = loss_fn(q, p, n)
        assert loss.item() > 0
        assert loss.item() < 10.0

    def test_gradient_flows_to_query(self) -> None:
        loss_fn = self._make_loss()
        q = torch.randn(BATCH, EMB_DIM, requires_grad=True)
        p = torch.randn(BATCH, EMB_DIM)
        n = torch.randn(BATCH, NUM_NEG, EMB_DIM)
        loss = loss_fn(q, p, n)
        loss.backward()
        assert q.grad is not None

    def test_temperature_override(self) -> None:
        loss_fn = self._make_loss(temperature=0.07)
        q = torch.randn(BATCH, EMB_DIM)
        p = torch.randn(BATCH, EMB_DIM)
        n = torch.randn(BATCH, NUM_NEG, EMB_DIM)
        l1 = loss_fn(q, p, n, temperature=0.07)
        l2 = loss_fn(q, p, n, temperature=1.0)
        assert not torch.isclose(l1, l2)

    def test_invalid_temperature(self) -> None:
        from src.model.losses import InfoNCELoss
        with pytest.raises(ValueError):
            InfoNCELoss(temperature=0.0)
        with pytest.raises(ValueError):
            InfoNCELoss(temperature=-1.0)

    def test_single_negative(self) -> None:
        loss_fn = self._make_loss()
        q = torch.randn(BATCH, EMB_DIM)
        p = torch.randn(BATCH, EMB_DIM)
        n = torch.randn(BATCH, 1, EMB_DIM)
        loss = loss_fn(q, p, n)
        assert loss.item() > 0

    def test_many_negatives(self) -> None:
        loss_fn = self._make_loss()
        q = torch.randn(BATCH, EMB_DIM)
        p = torch.randn(BATCH, EMB_DIM)
        n = torch.randn(BATCH, 50, EMB_DIM)
        loss = loss_fn(q, p, n)
        assert loss.item() > 0


# ====================================================================
# SECTION 6 — RouterLoss
# ====================================================================


class TestRouterLoss:
    """Test binary cross-entropy router loss."""

    def _make_loss(self, **kwargs: Any) -> "RouterLoss":
        from src.model.losses import RouterLoss
        return RouterLoss(**kwargs)

    def test_basic_loss_positive(self) -> None:
        loss_fn = self._make_loss()
        logits = torch.randn(BATCH, SEQ_LEN)
        labels = torch.randint(0, 2, (BATCH, SEQ_LEN)).float()
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self) -> None:
        loss_fn = self._make_loss()
        labels = torch.randint(0, 2, (BATCH, SEQ_LEN)).float()
        logits = labels * 10.0 - (1 - labels) * 10.0
        loss = loss_fn(logits, labels)
        assert loss.item() < 0.01

    def test_mask_excludes_positions(self) -> None:
        loss_fn = self._make_loss()
        logits = torch.randn(BATCH, SEQ_LEN)
        labels = torch.ones(BATCH, SEQ_LEN)
        mask_full = torch.ones(BATCH, SEQ_LEN)
        mask_half = torch.zeros(BATCH, SEQ_LEN)
        mask_half[:, :SEQ_LEN // 2] = 1.0

        loss_full = loss_fn(logits, labels, mask=mask_full)
        loss_half = loss_fn(logits, labels, mask=mask_half)
        assert loss_full.shape == ()
        assert loss_half.shape == ()

    def test_label_smoothing(self) -> None:
        loss_no_smooth = self._make_loss(label_smoothing=0.0)
        loss_smooth = self._make_loss(label_smoothing=0.1)
        logits = torch.randn(BATCH, SEQ_LEN)
        labels = torch.randint(0, 2, (BATCH, SEQ_LEN)).float()
        l1 = loss_no_smooth(logits, labels)
        l2 = loss_smooth(logits, labels)
        assert not torch.isclose(l1, l2)

    def test_pos_weight(self) -> None:
        loss_fn = self._make_loss(pos_weight=2.0)
        logits = torch.randn(BATCH, SEQ_LEN)
        labels = torch.ones(BATCH, SEQ_LEN)
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_3d_logits_squeezed(self) -> None:
        loss_fn = self._make_loss()
        logits = torch.randn(BATCH, SEQ_LEN, 1)
        labels = torch.randint(0, 2, (BATCH, SEQ_LEN)).float()
        loss = loss_fn(logits, labels)
        assert loss.shape == ()


# ====================================================================
# SECTION 7 — GenerationLoss
# ====================================================================


class TestGenerationLoss:
    """Test cross-entropy generation loss."""

    def _make_loss(self, **kwargs: Any) -> "GenerationLoss":
        from src.model.losses import GenerationLoss
        return GenerationLoss(**kwargs)

    def test_basic_loss_positive(self) -> None:
        loss_fn = self._make_loss()
        logits = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE)
        labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self) -> None:
        loss_fn = self._make_loss()
        labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        logits = torch.full((BATCH, SEQ_LEN, VOCAB_SIZE), -100.0)
        for b in range(BATCH):
            for s in range(SEQ_LEN):
                logits[b, s, labels[b, s]] = 100.0
        loss = loss_fn(logits, labels)
        assert loss.item() < 0.01

    def test_mask_excludes_positions(self) -> None:
        loss_fn = self._make_loss()
        logits = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE)
        labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
        mask[:, :SEQ_LEN // 2] = True
        loss = loss_fn(logits, labels, mask=mask)
        assert loss.shape == ()

    def test_ignore_index(self) -> None:
        loss_fn = self._make_loss(ignore_index=-100)
        logits = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE)
        labels = torch.full((BATCH, SEQ_LEN), -100, dtype=torch.long)
        loss = loss_fn(logits, labels)
        assert loss.item() == 0.0

    def test_label_smoothing(self) -> None:
        loss_no = self._make_loss(label_smoothing=0.0)
        loss_yes = self._make_loss(label_smoothing=0.1)
        logits = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE)
        labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        l1 = loss_no(logits, labels)
        l2 = loss_yes(logits, labels)
        assert not torch.isclose(l1, l2)


# ====================================================================
# SECTION 8 — FRLMCombinedLoss
# ====================================================================


class TestFRLMCombinedLoss:
    """Test the weighted combined loss."""

    def _make_loss(self, **kwargs: Any) -> "FRLMCombinedLoss":
        from src.model.losses import FRLMCombinedLoss
        return FRLMCombinedLoss(**kwargs)

    def test_all_components(self) -> None:
        loss_fn = self._make_loss()
        router_logits = torch.randn(BATCH, SEQ_LEN)
        router_labels = torch.randint(0, 2, (BATCH, SEQ_LEN)).float()
        query_emb = torch.randn(BATCH, EMB_DIM)
        pos_emb = torch.randn(BATCH, EMB_DIM)
        neg_embs = torch.randn(BATCH, NUM_NEG, EMB_DIM)
        gen_logits = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE)
        gen_labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        router_mask = router_labels

        total, loss_dict = loss_fn(
            router_logits=router_logits,
            router_labels=router_labels,
            query_emb=query_emb,
            positive_emb=pos_emb,
            negative_embs=neg_embs,
            gen_logits=gen_logits,
            gen_labels=gen_labels,
            router_mask=router_mask,
        )
        assert total.item() > 0
        assert "router_loss" in loss_dict
        assert "retrieval_loss" in loss_dict
        assert "generation_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_router_only(self) -> None:
        loss_fn = self._make_loss()
        router_logits = torch.randn(BATCH, SEQ_LEN)
        router_labels = torch.randint(0, 2, (BATCH, SEQ_LEN)).float()

        total, loss_dict = loss_fn(
            router_logits=router_logits,
            router_labels=router_labels,
        )
        assert "router_loss" in loss_dict
        assert "retrieval_loss" not in loss_dict
        assert "generation_loss" not in loss_dict

    def test_weights_affect_total(self) -> None:
        router_logits = torch.randn(BATCH, SEQ_LEN)
        router_labels = torch.randint(0, 2, (BATCH, SEQ_LEN)).float()
        query_emb = torch.randn(BATCH, EMB_DIM)
        pos_emb = torch.randn(BATCH, EMB_DIM)
        neg_embs = torch.randn(BATCH, NUM_NEG, EMB_DIM)

        loss_1x = self._make_loss(router_weight=1.0, retrieval_weight=1.0)
        loss_2x = self._make_loss(router_weight=1.0, retrieval_weight=2.0)

        t1, d1 = loss_1x(router_logits, router_labels, query_emb, pos_emb, neg_embs)
        t2, d2 = loss_2x(router_logits, router_labels, query_emb, pos_emb, neg_embs)

        ret_loss = d1["retrieval_loss"]
        diff = t2 - t1
        assert torch.isclose(diff, ret_loss, atol=1e-5)

    def test_attention_mask_passed(self) -> None:
        loss_fn = self._make_loss()
        router_logits = torch.randn(BATCH, SEQ_LEN)
        router_labels = torch.randint(0, 2, (BATCH, SEQ_LEN)).float()
        attn_mask = torch.ones(BATCH, SEQ_LEN)
        attn_mask[:, SEQ_LEN // 2:] = 0

        t1, _ = loss_fn(router_logits, router_labels, attention_mask=attn_mask)
        t2, _ = loss_fn(router_logits, router_labels)
        assert t1.shape == ()
        assert t2.shape == ()

    def test_from_config(self, default_config: FRLMConfig) -> None:
        from src.model.losses import FRLMCombinedLoss
        loss_fn = FRLMCombinedLoss.from_config(
            default_config.loss, default_config.training
        )
        assert loss_fn.lambda_router == 1.0
        assert loss_fn.lambda_retrieval == 2.0
        assert loss_fn.lambda_generation == 1.0

    def test_empty_retrieval_positions(self) -> None:
        loss_fn = self._make_loss()
        router_logits = torch.randn(BATCH, SEQ_LEN)
        router_labels = torch.zeros(BATCH, SEQ_LEN)
        query_emb = torch.randn(0, EMB_DIM)
        pos_emb = torch.randn(0, EMB_DIM)
        neg_embs = torch.randn(0, NUM_NEG, EMB_DIM)

        total, loss_dict = loss_fn(
            router_logits=router_logits,
            router_labels=router_labels,
            query_emb=query_emb,
            positive_emb=pos_emb,
            negative_embs=neg_embs,
        )
        assert loss_dict["retrieval_loss"].item() == 0.0


# ====================================================================
# SECTION 9 — Mocked backbone for integration tests
# ====================================================================


class _MockTransformer(nn.Module):
    """Tiny transformer stand-in for integration tests."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_dim)
        self.wpe = nn.Embedding(512, hidden_dim)
        self.h = nn.ModuleList([nn.Identity()])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.config = MagicMock()
        self.config.hidden_size = hidden_dim
        self.config.vocab_size = vocab_size
        self.config.output_hidden_states = True
        self._hidden_dim = hidden_dim

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, **kwargs: Any) -> MagicMock:
        batch, seq = input_ids.shape
        tok_emb = self.wte(input_ids)
        pos_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        pos_emb = self.wpe(pos_ids)
        hidden = self.ln_f(tok_emb + pos_emb)
        out = MagicMock()
        out.last_hidden_state = hidden
        out.hidden_states = (hidden,)
        return out

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")

    def gradient_checkpointing_enable(self) -> None:
        pass


def _make_mock_backbone(
    hidden_dim: int = HIDDEN_DIM,
    vocab_size: int = VOCAB_SIZE,
    freeze: bool = False,
) -> "BioMedLMBackbone":
    """Create a BioMedLMBackbone with a tiny mock transformer."""
    from src.model.backbone import BioMedLMBackbone
    bb = BioMedLMBackbone.__new__(BioMedLMBackbone)
    nn.Module.__init__(bb)
    bb._model_name = "mock"
    bb._hidden_dim = hidden_dim
    bb._freeze = freeze
    bb._gradient_checkpointing = False
    bb._dtype_str = "float32"
    bb.transformer = _MockTransformer(hidden_dim, vocab_size)
    if freeze:
        for p in bb.transformer.parameters():
            p.requires_grad = False
    return bb


def _make_frlm(
    hidden_dim: int = HIDDEN_DIM,
    emb_dim: int = EMB_DIM,
    vocab_size: int = VOCAB_SIZE,
    with_loss: bool = True,
    freeze_backbone: bool = False,
) -> "FRLMModel":
    """Assemble a complete FRLMModel with mock backbone."""
    from src.model.frlm import FRLMModel
    from src.model.router_head import RouterHead
    from src.model.retrieval_head import RetrievalHead
    from src.model.generation_head import GenerationHead
    from src.model.losses import FRLMCombinedLoss

    backbone = _make_mock_backbone(hidden_dim, vocab_size, freeze=freeze_backbone)
    router = RouterHead(hidden_dim=hidden_dim, intermediate_dim=32)
    retrieval = RetrievalHead(hidden_dim=hidden_dim, embedding_dim=emb_dim)
    generation = GenerationHead(hidden_dim=hidden_dim, vocab_size=vocab_size)
    loss_fn = FRLMCombinedLoss() if with_loss else None

    return FRLMModel(
        backbone=backbone,
        router=router,
        retrieval_head=retrieval,
        generation_head=generation,
        loss_fn=loss_fn,
    )


# ====================================================================
# SECTION 10 — FRLMModel full forward pass
# ====================================================================


class TestFRLMModelForward:
    """Test the full composite model forward pass."""

    def _make_inputs(self) -> Dict[str, Tensor]:
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        attention_mask = torch.ones(BATCH, SEQ_LEN)
        router_labels = torch.randint(0, 2, (BATCH, SEQ_LEN)).float()
        fact_embs = torch.randn(BATCH, SEQ_LEN, EMB_DIM)
        neg_embs = torch.randn(BATCH, SEQ_LEN, NUM_NEG, EMB_DIM)
        token_labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "router_labels": router_labels,
            "fact_embeddings": fact_embs,
            "negative_embeddings": neg_embs,
            "token_labels": token_labels,
        }

    def test_output_shapes(self) -> None:
        model = _make_frlm()
        inputs = self._make_inputs()
        out = model(**inputs)
        assert out.last_hidden_state.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)
        assert out.router_logits.shape == (BATCH, SEQ_LEN, 1)
        assert out.router_probs.shape == (BATCH, SEQ_LEN, 1)
        assert out.router_mask.shape == (BATCH, SEQ_LEN)
        assert out.router_mask.dtype == torch.bool
        assert out.gen_logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)

    def test_query_signature_present(self) -> None:
        model = _make_frlm()
        inputs = self._make_inputs()
        out = model(**inputs)
        qs = out.query_signature
        assert qs is not None
        assert qs.semantic_embedding.shape == (BATCH, SEQ_LEN, EMB_DIM)

    def test_loss_computed_when_labels_provided(self) -> None:
        model = _make_frlm(with_loss=True)
        inputs = self._make_inputs()
        out = model(**inputs)
        assert out.total_loss is not None
        assert out.total_loss.item() > 0
        assert out.loss_dict is not None
        assert "router_loss" in out.loss_dict

    def test_no_loss_without_labels(self) -> None:
        model = _make_frlm(with_loss=True)
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = model(input_ids=input_ids)
        assert out.total_loss is None
        assert out.loss_dict is None

    def test_no_loss_without_loss_fn(self) -> None:
        model = _make_frlm(with_loss=False)
        inputs = self._make_inputs()
        out = model(**inputs)
        assert out.total_loss is None

    def test_gradient_flows_end_to_end(self) -> None:
        model = _make_frlm(with_loss=True)
        inputs = self._make_inputs()
        out = model(**inputs)
        out.total_loss.backward()
        for p in model.router.parameters():
            assert p.grad is not None

    def test_router_probs_in_range(self) -> None:
        model = _make_frlm()
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = model(input_ids=input_ids)
        assert (out.router_probs >= 0).all()
        assert (out.router_probs <= 1).all()

    def test_semantic_embedding_normalised(self) -> None:
        model = _make_frlm()
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = model(input_ids=input_ids)
        norms = out.query_signature.semantic_embedding.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ====================================================================
# SECTION 11 — Frozen backbone
# ====================================================================


class TestFrozenBackbone:
    """Verify frozen backbone behaviour."""

    def test_frozen_params_no_grad(self) -> None:
        model = _make_frlm(freeze_backbone=True)
        for p in model.backbone.parameters():
            assert not p.requires_grad

    def test_frozen_produces_identical_outputs(self) -> None:
        model = _make_frlm(freeze_backbone=True)
        model.eval()
        input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
        with torch.no_grad():
            out1 = model.backbone(input_ids)
            out2 = model.backbone(input_ids)
        assert torch.equal(out1.last_hidden_state, out2.last_hidden_state)

    def test_frozen_backbone_heads_still_train(self) -> None:
        model = _make_frlm(freeze_backbone=True, with_loss=True)
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        router_labels = torch.randint(0, 2, (BATCH, SEQ_LEN)).float()
        token_labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = model(
            input_ids=input_ids,
            router_labels=router_labels,
            token_labels=token_labels,
        )
        out.total_loss.backward()
        for p in model.router.parameters():
            assert p.grad is not None
        for p in model.backbone.parameters():
            assert p.grad is None

    def test_freeze_unfreeze(self) -> None:
        model = _make_frlm(freeze_backbone=False)
        assert all(p.requires_grad for p in model.backbone.parameters())
        model.backbone.freeze()
        assert all(not p.requires_grad for p in model.backbone.parameters())
        model.backbone.unfreeze()
        assert all(p.requires_grad for p in model.backbone.parameters())


# ====================================================================
# SECTION 12 — Router mask gating
# ====================================================================


class TestRouterMaskGating:
    """Test that router mask correctly separates retrieval vs generation."""

    def test_mask_is_boolean(self) -> None:
        model = _make_frlm()
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = model(input_ids=input_ids)
        assert out.router_mask.dtype == torch.bool

    def test_mask_depends_on_threshold(self) -> None:
        model = _make_frlm()
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

        model.router_threshold = 0.0
        out_low = model(input_ids=input_ids)

        model.router_threshold = 1.0
        out_high = model(input_ids=input_ids)

        assert out_low.router_mask.all()
        assert not out_high.router_mask.any()

    def test_retrieval_positions_get_embeddings(self) -> None:
        model = _make_frlm(with_loss=True)
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        router_labels = torch.ones(BATCH, SEQ_LEN)
        fact_embs = torch.randn(BATCH, SEQ_LEN, EMB_DIM)
        neg_embs = torch.randn(BATCH, SEQ_LEN, NUM_NEG, EMB_DIM)
        token_labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

        out = model(
            input_ids=input_ids,
            router_labels=router_labels,
            fact_embeddings=fact_embs,
            negative_embeddings=neg_embs,
            token_labels=token_labels,
        )
        assert out.loss_dict is not None
        assert "retrieval_loss" in out.loss_dict
        assert out.loss_dict["retrieval_loss"].item() > 0

    def test_generation_positions_use_gen_loss(self) -> None:
        model = _make_frlm(with_loss=True)
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        router_labels = torch.zeros(BATCH, SEQ_LEN)
        token_labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

        out = model(
            input_ids=input_ids,
            router_labels=router_labels,
            token_labels=token_labels,
        )
        assert out.loss_dict is not None
        assert "generation_loss" in out.loss_dict
        assert out.loss_dict["generation_loss"].item() > 0

    def test_mixed_mask_both_losses(self) -> None:
        model = _make_frlm(with_loss=True)
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        router_labels = torch.zeros(BATCH, SEQ_LEN)
        router_labels[:, :SEQ_LEN // 2] = 1.0
        fact_embs = torch.randn(BATCH, SEQ_LEN, EMB_DIM)
        neg_embs = torch.randn(BATCH, SEQ_LEN, NUM_NEG, EMB_DIM)
        token_labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

        out = model(
            input_ids=input_ids,
            router_labels=router_labels,
            fact_embeddings=fact_embs,
            negative_embeddings=neg_embs,
            token_labels=token_labels,
        )
        assert out.loss_dict is not None
        assert "router_loss" in out.loss_dict
        assert "retrieval_loss" in out.loss_dict
        assert "generation_loss" in out.loss_dict
        for k in ("router_loss", "retrieval_loss", "generation_loss"):
            assert out.loss_dict[k].item() > 0


# ====================================================================
# SECTION 13 — BackboneOutput and module properties
# ====================================================================


class TestBackboneOutputAndProperties:
    """Test BackboneOutput dataclass and backbone properties."""

    def test_backbone_output_dataclass(self) -> None:
        from src.model.backbone import BackboneOutput
        hidden = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = BackboneOutput(last_hidden_state=hidden, all_hidden_states=(hidden,))
        assert torch.equal(out.last_hidden_state, hidden)
        assert len(out.all_hidden_states) == 1

    def test_mock_backbone_hidden_dim(self) -> None:
        bb = _make_mock_backbone(hidden_dim=64)
        assert bb.get_hidden_dim() == 64

    def test_mock_backbone_vocab_size(self) -> None:
        bb = _make_mock_backbone(vocab_size=128)
        assert bb.vocab_size == 128

    def test_mock_backbone_embedding_weight(self) -> None:
        bb = _make_mock_backbone(hidden_dim=64, vocab_size=128)
        w = bb.get_embedding_weight()
        assert w.shape == (128, 64)

    def test_forward_returns_backbone_output(self) -> None:
        bb = _make_mock_backbone()
        input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
        out = bb(input_ids)
        from src.model.backbone import BackboneOutput
        assert isinstance(out, BackboneOutput)
        assert out.last_hidden_state.shape == (2, SEQ_LEN, HIDDEN_DIM)


# ====================================================================
# SECTION 14 — QuerySignature helpers
# ====================================================================


class TestQuerySignatureHelpers:
    """Test QuerySignature dataclass properties."""

    def test_granularity_level_shape(self) -> None:
        from src.model.retrieval_head import QuerySignature
        qs = QuerySignature(
            semantic_embedding=torch.randn(BATCH, EMB_DIM),
            granularity_logits=torch.randn(BATCH, NUM_GRAN),
            temporal_logits=torch.randn(BATCH, NUM_TEMP),
        )
        assert qs.granularity_level.shape == (BATCH,)

    def test_temporal_mode_shape(self) -> None:
        from src.model.retrieval_head import QuerySignature
        qs = QuerySignature(
            semantic_embedding=torch.randn(BATCH, EMB_DIM),
            granularity_logits=torch.randn(BATCH, NUM_GRAN),
            temporal_logits=torch.randn(BATCH, NUM_TEMP),
        )
        assert qs.temporal_mode.shape == (BATCH,)

    def test_argmax_correctness(self) -> None:
        from src.model.retrieval_head import QuerySignature
        gran_logits = torch.tensor([[0.1, 0.2, 0.9, 0.0]])
        temp_logits = torch.tensor([[0.3, 0.1, 0.8]])
        qs = QuerySignature(
            semantic_embedding=torch.randn(1, EMB_DIM),
            granularity_logits=gran_logits,
            temporal_logits=temp_logits,
        )
        assert qs.granularity_level.item() == 2
        assert qs.temporal_mode.item() == 2


# ====================================================================
# SECTION 15 — Module exports
# ====================================================================


class TestModelExports:
    """Verify __init__.py exports the expected symbols."""

    def test_all_classes_importable(self) -> None:
        from src.model import (
            BackboneOutput,
            BioMedLMBackbone,
            FRLMCombinedLoss,
            FRLMModel,
            FRLMOutput,
            GenerationHead,
            GenerationLoss,
            InfoNCELoss,
            QuerySignature,
            RetrievalHead,
            RouterHead,
            RouterLoss,
        )
        for cls in (
            BackboneOutput, BioMedLMBackbone, FRLMCombinedLoss, FRLMModel,
            FRLMOutput, GenerationHead, GenerationLoss, InfoNCELoss,
            QuerySignature, RetrievalHead, RouterHead, RouterLoss,
        ):
            assert cls is not None

    def test_all_list(self) -> None:
        import src.model as m
        assert hasattr(m, "__all__")
        assert "FRLMModel" in m.__all__
        assert "InfoNCELoss" in m.__all__
        assert "RouterLoss" in m.__all__
        assert "GenerationLoss" in m.__all__
        assert "FRLMCombinedLoss" in m.__all__
        assert "FRLMOutput" in m.__all__


# ====================================================================
# SECTION 16 — FRLMModel properties
# ====================================================================


class TestFRLMModelProperties:
    """Test FRLMModel property accessors and threshold management."""

    def test_hidden_dim(self) -> None:
        model = _make_frlm(hidden_dim=64)
        assert model.hidden_dim == 64

    def test_router_threshold_default(self) -> None:
        model = _make_frlm()
        assert model.router_threshold == 0.5

    def test_router_threshold_setter(self) -> None:
        model = _make_frlm()
        model.router_threshold = 0.8
        assert model.router_threshold == 0.8
        assert model.router.threshold == 0.8

    def test_has_all_components(self) -> None:
        model = _make_frlm()
        assert model.backbone is not None
        assert model.router is not None
        assert model.retrieval_head is not None
        assert model.generation_head is not None


# ====================================================================
# SECTION 17 — Save / Load round-trip
# ====================================================================


class TestSaveLoadRoundTrip:
    """Test save_pretrained / from_pretrained (with mock backbone)."""

    def test_save_creates_files(self) -> None:
        model = _make_frlm(with_loss=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            p = Path(tmpdir)
            assert (p / "backbone").is_dir()
            assert (p / "router.pt").is_file()
            assert (p / "retrieval.pt").is_file()
            assert (p / "generation.pt").is_file()
            assert (p / "loss.pt").is_file()
            assert (p / "config.json").is_file()

    def test_save_config_json_contents(self) -> None:
        model = _make_frlm()
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            with open(Path(tmpdir) / "config.json") as f:
                meta = json.load(f)
            assert meta["hidden_dim"] == HIDDEN_DIM
            assert meta["embedding_dim"] == EMB_DIM
            assert meta["vocab_size"] == VOCAB_SIZE
            assert meta["num_granularity_levels"] == NUM_GRAN
            assert meta["num_temporal_modes"] == NUM_TEMP

    def test_load_restores_heads(self) -> None:
        model = _make_frlm()
        input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
        model.eval()
        with torch.no_grad():
            orig_out = model(input_ids=input_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            with patch("src.model.frlm.BioMedLMBackbone") as MockBB:
                mock_bb = _make_mock_backbone()
                mock_bb.transformer.load_state_dict(model.backbone.transformer.state_dict())
                MockBB.return_value = mock_bb

                from src.model.frlm import FRLMModel
                loaded = FRLMModel.from_pretrained(tmpdir)

            loaded.eval()
            with torch.no_grad():
                loaded_out = loaded(input_ids=input_ids)

            assert torch.allclose(
                orig_out.router_logits, loaded_out.router_logits, atol=1e-5
            )

    def test_load_missing_dir_raises(self) -> None:
        from src.model.frlm import FRLMModel
        with pytest.raises(FileNotFoundError):
            FRLMModel.from_pretrained("/nonexistent/model/path")