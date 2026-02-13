"""
Tests for embedding operations configuration.

Tests SapBERT config, FAISS config, hierarchical index levels,
and hard negative mining settings.
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
# SapBERT config
# ===========================================================================


class TestSapBERTConfig:
    """Validate SapBERT encoder configuration."""

    def test_model_name(self, default_config: FRLMConfig) -> None:
        assert default_config.sapbert.model_name == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

    def test_embedding_dim(self, default_config: FRLMConfig) -> None:
        assert default_config.sapbert.embedding_dim == 768

    def test_max_length(self, default_config: FRLMConfig) -> None:
        assert default_config.sapbert.max_length == 64

    def test_batch_size(self, default_config: FRLMConfig) -> None:
        assert default_config.sapbert.batch_size == 256

    def test_pool_strategy(self, default_config: FRLMConfig) -> None:
        assert default_config.sapbert.pool_strategy == "cls"

    def test_device(self, default_config: FRLMConfig) -> None:
        assert default_config.sapbert.device in ("cuda", "cpu")


# ===========================================================================
# FAISS config
# ===========================================================================


class TestFAISSConfig:
    """Validate FAISS vector index configuration."""

    def test_index_type(self, default_config: FRLMConfig) -> None:
        assert "IVF" in default_config.faiss.index_type
        assert "PQ" in default_config.faiss.index_type

    def test_metric(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.metric in ("L2", "IP")

    def test_embedding_dim_matches_sapbert(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.embedding_dim == default_config.sapbert.embedding_dim

    def test_nlist_and_nprobe(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.nlist == 4096
        assert default_config.faiss.nprobe == 64
        assert default_config.faiss.nprobe <= default_config.faiss.nlist

    def test_pq_settings(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.pq_m == 64
        assert default_config.faiss.pq_nbits == 8
        # PQ sub-quantizers must divide embedding dimension
        assert default_config.faiss.embedding_dim % default_config.faiss.pq_m == 0

    def test_train_sample_size(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.train_sample_size == 500000

    def test_search_k(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.search_k == 100


# ===========================================================================
# Hierarchical index levels
# ===========================================================================


class TestHierarchicalIndex:
    """Validate hierarchical index level configuration."""

    def test_four_levels_defined(self, default_config: FRLMConfig) -> None:
        hier = default_config.faiss.hierarchical
        assert hier.level_0 == "atomic"
        assert hier.level_1 == "relation"
        assert hier.level_2 == "entity"
        assert hier.level_3 == "cluster"

    def test_default_level(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.hierarchical.default_level == 0

    def test_levels_match_granularity_head(self, default_config: FRLMConfig) -> None:
        """FAISS hierarchical level names should match granularity sub-head level names."""
        hier = default_config.faiss.hierarchical
        granularity_names = default_config.model.retrieval_head.granularity.level_names
        faiss_names = [hier.level_0, hier.level_1, hier.level_2, hier.level_3]
        assert faiss_names == granularity_names

    def test_num_granularity_levels_matches(self, default_config: FRLMConfig) -> None:
        assert default_config.model.retrieval_head.granularity.num_levels == 4


# ===========================================================================
# Hard negative config
# ===========================================================================


class TestHardNegatives:
    """Validate hard negative mining configuration."""

    def test_num_hard_negatives(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.hard_negatives.num_hard_negatives == 15

    def test_num_random_negatives(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.hard_negatives.num_random_negatives == 5

    def test_mine_frequency(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.hard_negatives.mine_frequency == 1000

    def test_similarity_range(self, default_config: FRLMConfig) -> None:
        sr = default_config.faiss.hard_negatives.similarity_range
        assert sr.min == 0.3
        assert sr.max == 0.8
        assert sr.min < sr.max

    def test_total_negatives_per_positive(self, default_config: FRLMConfig) -> None:
        hn = default_config.faiss.hard_negatives
        total = hn.num_hard_negatives + hn.num_random_negatives
        assert total == 20