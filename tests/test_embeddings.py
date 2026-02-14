"""
Tests for Phase 4 — Embedding Pipeline & FAISS Index.

Tests cover:
- SapBERT encoder configuration and mock-based encode behaviour
- FAISS index build, search, batch-search, hard-negative mining, save/load
- Hierarchical index: all four levels, expand_to_fact_ids, save/load
- Configuration cross-validation (SapBERT ↔ FAISS dims, etc.)

All heavy models (SapBERT transformer, GPU FAISS) are mocked or replaced
with lightweight surrogates so tests run in < 2 s on CPU without downloads.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, load_config
from src.kg.schema import (
    BiomedicalEntity,
    ClusterType,
    Fact,
    FactCluster,
    Relation,
    TemporalEnvelope,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTITY_DRUG = BiomedicalEntity(
    id="C0001", label="Gefitinib", entity_type="Drug", canonical_id="C0001"
)
_ENTITY_GENE = BiomedicalEntity(
    id="C0002", label="EGFR", entity_type="Gene", canonical_id="C0002"
)
_ENTITY_DISEASE = BiomedicalEntity(
    id="C0003", label="NSCLC", entity_type="Disease", canonical_id="C0003"
)
_ENTITY_PROTEIN = BiomedicalEntity(
    id="C0004", label="KRAS", entity_type="Protein", canonical_id="C0004"
)


def _make_fact(
    subject: BiomedicalEntity = _ENTITY_DRUG,
    obj: BiomedicalEntity = _ENTITY_GENE,
    relation: str = "INHIBITS",
    valid_from: date = date(2024, 1, 1),
    valid_to: date | None = None,
) -> Fact:
    return Fact(
        subject=subject,
        relation=Relation(type=relation),
        object=obj,
        temporal=TemporalEnvelope(valid_from=valid_from, valid_to=valid_to),
        source="PMID:12345",
        confidence=0.95,
    )


def _make_sample_facts() -> List[Fact]:
    """Create a diverse set of 6 facts across 4 entities."""
    return [
        _make_fact(_ENTITY_DRUG, _ENTITY_GENE, "INHIBITS"),
        _make_fact(_ENTITY_DRUG, _ENTITY_DISEASE, "TREATS"),
        _make_fact(_ENTITY_GENE, _ENTITY_DISEASE, "ASSOCIATED_WITH"),
        _make_fact(_ENTITY_GENE, _ENTITY_PROTEIN, "INTERACTS_WITH"),
        _make_fact(_ENTITY_DRUG, _ENTITY_PROTEIN, "BINDS_TO"),
        _make_fact(
            _ENTITY_DRUG, _ENTITY_GENE, "INHIBITS",
            valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1),
        ),
    ]


def _random_embeddings(n: int, dim: int = 768) -> np.ndarray:
    """L2-normalised random vectors."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ===========================================================================
# Config-level tests (preserved from Phase 1 test file)
# ===========================================================================


@pytest.fixture(scope="module")
def default_config() -> FRLMConfig:
    return load_config()


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
        assert default_config.faiss.embedding_dim % default_config.faiss.pq_m == 0

    def test_train_sample_size(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.train_sample_size == 500000

    def test_search_k(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.search_k == 100


class TestHierarchicalConfig:
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
        hier = default_config.faiss.hierarchical
        granularity_names = default_config.model.retrieval_head.granularity.level_names
        faiss_names = [hier.level_0, hier.level_1, hier.level_2, hier.level_3]
        assert faiss_names == granularity_names

    def test_num_granularity_levels_matches(self, default_config: FRLMConfig) -> None:
        assert default_config.model.retrieval_head.granularity.num_levels == 4


class TestHardNegativeConfig:
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


# ===========================================================================
# SapBERT encoder (mocked model)
# ===========================================================================


class TestSapBERTEncoder:
    """Test SapBERTEncoder with a mocked HuggingFace model."""

    @pytest.fixture(autouse=True)
    def _mock_encoder(self) -> None:
        """Patch AutoModel/AutoTokenizer so no model is downloaded."""
        import torch

        # Fake tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        # Fake model output
        fake_hidden = torch.randn(1, 10, 768)
        mock_output = MagicMock()
        mock_output.last_hidden_state = fake_hidden

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_model.parameters.return_value = [torch.zeros(1)]

        with patch(
            "src.embeddings.sapbert.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch(
            "src.embeddings.sapbert.AutoModel.from_pretrained",
            return_value=mock_model,
        ):
            from src.embeddings.sapbert import SapBERTEncoder
            self.encoder = SapBERTEncoder(
                model_name="test-model",
                device="cpu",
                dtype="float32",
            )

    def test_embedding_dim_property(self) -> None:
        assert self.encoder.embedding_dim == 768

    def test_device_property(self) -> None:
        import torch
        assert self.encoder.device == torch.device("cpu")

    def test_pool_strategy_default(self) -> None:
        assert self.encoder.pool_strategy == "cls"

    def test_model_name_property(self) -> None:
        assert self.encoder.model_name == "test-model"

    def test_encode_query_returns_correct_shape(self) -> None:
        vec = self.encoder.encode_query("Gefitinib inhibits EGFR")
        assert vec.shape == (768,)
        assert vec.dtype == np.float32

    def test_encode_query_is_l2_normalised(self) -> None:
        vec = self.encoder.encode_query("test query")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_encode_fact_returns_correct_shape(self) -> None:
        fact = _make_fact()
        vec = self.encoder.encode_fact(fact)
        assert vec.shape == (768,)

    def test_encode_facts_batch_shape(self) -> None:
        import torch

        # Make the mock tokenizer/model handle variable batch sizes
        def dynamic_tokenizer(texts, **kwargs):
            n = len(texts)
            return {
                "input_ids": torch.zeros(n, 10, dtype=torch.long),
                "attention_mask": torch.ones(n, 10, dtype=torch.long),
            }

        self.encoder._tokenizer.side_effect = dynamic_tokenizer

        def dynamic_model(**kwargs):
            n = kwargs["input_ids"].shape[0]
            out = MagicMock()
            out.last_hidden_state = torch.randn(n, 10, 768)
            return out

        self.encoder._model.side_effect = dynamic_model

        facts = _make_sample_facts()
        vecs = self.encoder.encode_facts_batch(facts, batch_size=2)
        assert vecs.shape == (len(facts), 768)

    def test_encode_texts_batch_empty(self) -> None:
        vecs = self.encoder.encode_texts_batch([])
        assert vecs.shape == (0, 768)

    def test_invalid_pool_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="pool_strategy"):
            from src.embeddings.sapbert import SapBERTEncoder
            with patch(
                "src.embeddings.sapbert.AutoTokenizer.from_pretrained"
            ), patch(
                "src.embeddings.sapbert.AutoModel.from_pretrained"
            ):
                SapBERTEncoder(pool_strategy="invalid_pool")


class TestSapBERTPooling:
    """Test the three pooling strategies independently."""

    def test_cls_pooling(self) -> None:
        import torch
        from src.embeddings.sapbert import SapBERTEncoder

        # Directly test _pool without a real model
        encoder = object.__new__(SapBERTEncoder)
        encoder._pool_strategy = "cls"

        hidden = torch.randn(2, 5, 768)
        mask = torch.ones(2, 5, dtype=torch.long)
        pooled = encoder._pool(hidden, mask)
        assert pooled.shape == (2, 768)
        # CLS should pick index 0
        expected_direction = hidden[:, 0, :]
        # After L2 norm, direction should match
        cos_sim = torch.nn.functional.cosine_similarity(pooled, expected_direction, dim=-1)
        assert (cos_sim > 0.99).all()

    def test_mean_pooling(self) -> None:
        import torch
        from src.embeddings.sapbert import SapBERTEncoder

        encoder = object.__new__(SapBERTEncoder)
        encoder._pool_strategy = "mean"

        hidden = torch.ones(1, 4, 768)
        mask = torch.tensor([[1, 1, 1, 0]])
        pooled = encoder._pool(hidden, mask)
        assert pooled.shape == (1, 768)
        # All-ones mean should be all-ones (then normalised)
        norm = pooled.norm(dim=-1)
        assert abs(norm.item() - 1.0) < 1e-5

    def test_max_pooling(self) -> None:
        import torch
        from src.embeddings.sapbert import SapBERTEncoder

        encoder = object.__new__(SapBERTEncoder)
        encoder._pool_strategy = "max"

        hidden = torch.tensor([[[1.0, 2.0], [3.0, 0.0], [0.0, 0.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = encoder._pool(hidden, mask)
        assert pooled.shape == (1, 2)
        # Max of [1,3,0] and [2,0,0] with mask → [3,2] normalised
        expected = torch.tensor([[3.0, 2.0]])
        expected = expected / expected.norm(dim=-1, keepdim=True)
        assert torch.allclose(pooled, expected, atol=1e-5)


# ===========================================================================
# FAISS index
# ===========================================================================


class TestFAISSFactIndex:
    """Test FAISSFactIndex build/search/save/load."""

    DIM = 64  # small dimension for fast tests

    @pytest.fixture()
    def index(self) -> "FAISSFactIndex":
        from src.embeddings.faiss_index import FAISSFactIndex
        return FAISSFactIndex(
            embedding_dim=self.DIM,
            index_type="Flat",
            metric="L2",
            nprobe=1,
            use_gpu=False,
        )

    @pytest.fixture()
    def populated_index(self, index: "FAISSFactIndex") -> "FAISSFactIndex":
        n = 100
        rng = np.random.default_rng(0)
        embs = rng.standard_normal((n, self.DIM)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms
        ids = [f"fact_{i:04d}" for i in range(n)]
        index.build_index(embs, ids)
        return index

    def test_build_index_properties(self, populated_index: "FAISSFactIndex") -> None:
        assert populated_index.ntotal == 100
        assert populated_index.is_trained is True
        assert populated_index.embedding_dim == self.DIM

    def test_search_returns_correct_count(self, populated_index: "FAISSFactIndex") -> None:
        q = np.random.randn(self.DIM).astype(np.float32)
        results = populated_index.search(q, top_k=5)
        assert len(results) == 5

    def test_search_returns_tuples(self, populated_index: "FAISSFactIndex") -> None:
        q = np.random.randn(self.DIM).astype(np.float32)
        results = populated_index.search(q, top_k=3)
        for fid, dist in results:
            assert isinstance(fid, str)
            assert fid.startswith("fact_")
            assert isinstance(dist, float)

    def test_search_batch(self, populated_index: "FAISSFactIndex") -> None:
        q = np.random.randn(3, self.DIM).astype(np.float32)
        results = populated_index.search_batch(q, top_k=5)
        assert len(results) == 3
        assert all(len(r) == 5 for r in results)

    def test_search_identity(self, index: "FAISSFactIndex") -> None:
        """Searching with a vector that was added should return itself."""
        embs = _random_embeddings(10, self.DIM)
        ids = [f"id_{i}" for i in range(10)]
        index.build_index(embs, ids)

        results = index.search(embs[3], top_k=1)
        assert results[0][0] == "id_3"

    def test_build_validates_dim_mismatch(self, index: "FAISSFactIndex") -> None:
        embs = np.random.randn(5, self.DIM + 1).astype(np.float32)
        with pytest.raises(ValueError, match="dim mismatch"):
            index.build_index(embs, ["a"] * 5)

    def test_build_validates_length_mismatch(self, index: "FAISSFactIndex") -> None:
        embs = np.random.randn(5, self.DIM).astype(np.float32)
        with pytest.raises(ValueError, match="fact_ids length"):
            index.build_index(embs, ["a"] * 3)

    def test_search_before_build_raises(self, index: "FAISSFactIndex") -> None:
        with pytest.raises(RuntimeError, match="not built"):
            index.search(np.zeros(self.DIM, dtype=np.float32))

    def test_mine_hard_negatives(self, populated_index: "FAISSFactIndex") -> None:
        q = np.random.randn(self.DIM).astype(np.float32)
        negs = populated_index.mine_hard_negatives(
            q, positive_fact_id="fact_0000", num_negatives=5, top_k_candidates=20
        )
        assert len(negs) == 5
        assert "fact_0000" not in negs
        # All returned are valid fact ids
        for fid in negs:
            assert fid.startswith("fact_")

    def test_mine_hard_negatives_excludes_positive(
        self, populated_index: "FAISSFactIndex"
    ) -> None:
        q = np.random.randn(self.DIM).astype(np.float32)
        for _ in range(5):  # repeat to exercise randomness
            negs = populated_index.mine_hard_negatives(
                q, positive_fact_id="fact_0050", num_negatives=10
            )
            assert "fact_0050" not in negs

    def test_save_and_load(self, populated_index: "FAISSFactIndex") -> None:
        from src.embeddings.faiss_index import FAISSFactIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "test_index"
            populated_index.save_index(base)

            # Check files exist
            assert base.with_suffix(".faiss").exists()
            assert base.with_suffix(".meta.json").exists()

            # Load into a fresh instance
            loaded = FAISSFactIndex(
                embedding_dim=self.DIM,
                index_type="Flat",
                metric="L2",
                use_gpu=False,
            )
            loaded.load_index(base)

            assert loaded.ntotal == populated_index.ntotal
            assert loaded.is_trained

            # Search should produce the same top-1
            q = np.random.default_rng(99).standard_normal(self.DIM).astype(np.float32)
            orig_results = populated_index.search(q, top_k=1)
            loaded_results = loaded.search(q, top_k=1)
            assert orig_results[0][0] == loaded_results[0][0]

    def test_load_missing_file_raises(self) -> None:
        from src.embeddings.faiss_index import FAISSFactIndex

        idx = FAISSFactIndex(embedding_dim=self.DIM, index_type="Flat", use_gpu=False)
        with pytest.raises(FileNotFoundError):
            idx.load_index("/nonexistent/path")

    def test_index_stats(self, populated_index: "FAISSFactIndex") -> None:
        stats = populated_index.index_stats()
        assert stats["ntotal"] == 100
        assert stats["embedding_dim"] == self.DIM
        assert stats["is_trained"] is True
        assert stats["num_fact_ids"] == 100

    def test_fact_id_for_index(self, populated_index: "FAISSFactIndex") -> None:
        assert populated_index.fact_id_for_index(0) == "fact_0000"
        assert populated_index.fact_id_for_index(99) == "fact_0099"
        assert populated_index.fact_id_for_index(-1) is None
        assert populated_index.fact_id_for_index(200) is None

    def test_index_for_fact_id(self, populated_index: "FAISSFactIndex") -> None:
        assert populated_index.index_for_fact_id("fact_0042") == 42
        assert populated_index.index_for_fact_id("nonexistent") is None


# ===========================================================================
# Hierarchical index
# ===========================================================================


class TestHierarchicalIndex:
    """Test the four-level hierarchical FAISS index."""

    DIM = 32  # tiny dimension for speed

    @pytest.fixture()
    def facts_and_embeddings(self):
        facts = _make_sample_facts()  # 6 facts, 4 entities
        embs = _random_embeddings(len(facts), self.DIM)
        return facts, embs

    @pytest.fixture()
    def built_hier(self, facts_and_embeddings):
        facts, embs = facts_and_embeddings
        from src.embeddings.hierarchical import HierarchicalIndex
        hier = HierarchicalIndex(
            embedding_dim=self.DIM,
            index_type="Flat",
            metric="L2",
            use_gpu=False,
        )
        hier.build_all_levels(facts, embs)
        return hier

    def test_level_0_count(self, built_hier, facts_and_embeddings) -> None:
        facts, _ = facts_and_embeddings
        assert built_hier._levels[0].ntotal == len(facts)

    def test_level_1_count(self, built_hier) -> None:
        """Level 1 = unique entity pairs. We have 5 distinct pairs in our sample."""
        # Pairs: (Drug,Gene), (Drug,Disease), (Gene,Disease),
        #        (Gene,Protein), (Drug,Protein)
        assert built_hier._levels[1].ntotal == 5

    def test_level_2_count(self, built_hier) -> None:
        """Level 2 = unique entities. 4 entities in sample."""
        assert built_hier._levels[2].ntotal == 4

    def test_level_3_empty_when_no_clusters(self, built_hier) -> None:
        assert built_hier._levels[3].ntotal == 0

    def test_level_3_with_clusters(self, facts_and_embeddings) -> None:
        facts, embs = facts_and_embeddings
        cluster = FactCluster(
            facts=facts[:3],
            cluster_type=ClusterType.CUSTOM,
            anchor_entity="C0001",
            cluster_id="cluster_0",
        )
        from src.embeddings.hierarchical import HierarchicalIndex
        hier = HierarchicalIndex(
            embedding_dim=self.DIM, index_type="Flat", use_gpu=False
        )
        hier.build_all_levels(facts, embs, clusters=[cluster])
        assert hier._levels[3].ntotal == 1

    def test_search_at_level_0(self, built_hier) -> None:
        q = np.random.randn(self.DIM).astype(np.float32)
        results = built_hier.search_at_level(q, level=0, top_k=3)
        assert len(results) == 3
        # All results should be fact_ids
        for fid, _ in results:
            assert len(fid) == 64  # SHA-256 hex

    def test_search_at_level_1(self, built_hier) -> None:
        q = np.random.randn(self.DIM).astype(np.float32)
        results = built_hier.search_at_level(q, level=1, top_k=2)
        assert len(results) == 2
        # Level 1 ids are pair keys
        for key, _ in results:
            assert "||" in key

    def test_search_at_level_2(self, built_hier) -> None:
        q = np.random.randn(self.DIM).astype(np.float32)
        results = built_hier.search_at_level(q, level=2, top_k=4)
        assert len(results) == 4
        # Level 2 ids are entity canonical_ids
        for eid, _ in results:
            assert eid.startswith("C000")

    def test_search_invalid_level_raises(self, built_hier) -> None:
        with pytest.raises(ValueError, match="Level 5"):
            built_hier.search_at_level(np.zeros(self.DIM, dtype=np.float32), level=5)

    def test_expand_to_fact_ids_level_0(self, built_hier) -> None:
        from src.embeddings.hierarchical import LEVEL_ATOMIC
        q = np.random.randn(self.DIM).astype(np.float32)
        results = built_hier.search_at_level(q, level=0, top_k=2)
        ids = [sid for sid, _ in results]
        expanded = built_hier.expand_to_fact_ids(ids, LEVEL_ATOMIC)
        assert expanded == ids  # level 0 returns fact_ids directly

    def test_expand_to_fact_ids_level_1(self, built_hier, facts_and_embeddings) -> None:
        from src.embeddings.hierarchical import LEVEL_RELATION
        facts, _ = facts_and_embeddings
        pair_key = f"{_ENTITY_DRUG.canonical_id}||{_ENTITY_GENE.canonical_id}"
        expanded = built_hier.expand_to_fact_ids([pair_key], LEVEL_RELATION)
        # Should return fact_ids for all facts between Drug and Gene
        # That's facts[0] (INHIBITS) and facts[5] (INHIBITS superseded)
        assert len(expanded) == 2

    def test_expand_to_fact_ids_level_2(self, built_hier, facts_and_embeddings) -> None:
        from src.embeddings.hierarchical import LEVEL_ENTITY
        facts, _ = facts_and_embeddings
        expanded = built_hier.expand_to_fact_ids(
            [_ENTITY_DRUG.canonical_id], LEVEL_ENTITY
        )
        # Drug appears in facts 0, 1, 4, 5 — as subject in all of them
        assert len(expanded) >= 3

    def test_build_validates_length(self) -> None:
        from src.embeddings.hierarchical import HierarchicalIndex
        hier = HierarchicalIndex(embedding_dim=self.DIM, index_type="Flat", use_gpu=False)
        facts = _make_sample_facts()
        embs = _random_embeddings(len(facts) + 1, self.DIM)  # wrong size
        with pytest.raises(ValueError, match="facts length"):
            hier.build_all_levels(facts, embs)

    def test_save_and_load(self, built_hier) -> None:
        from src.embeddings.hierarchical import HierarchicalIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            built_hier.save(tmpdir)

            loaded = HierarchicalIndex(
                embedding_dim=self.DIM, index_type="Flat", use_gpu=False
            )
            loaded.load(tmpdir)

            # Verify all levels match
            for level in range(4):
                orig_ntotal = built_hier._levels.get(level, MagicMock(ntotal=0)).ntotal
                loaded_ntotal = loaded._levels.get(level, MagicMock(ntotal=0)).ntotal
                assert loaded_ntotal == orig_ntotal

            # Verify auxiliary mappings survived
            assert loaded._pair_to_facts == built_hier._pair_to_facts
            assert loaded._entity_to_facts == built_hier._entity_to_facts

    def test_stats(self, built_hier) -> None:
        stats = built_hier.stats()
        assert "atomic" in stats
        assert "relation" in stats
        assert "entity" in stats
        assert "cluster" in stats
        assert stats["atomic"]["ntotal"] == 6
        assert stats["relation"]["ntotal"] == 5
        assert stats["entity"]["ntotal"] == 4
        assert stats["cluster"]["ntotal"] == 0


# ===========================================================================
# Hierarchical resolve (with mock KG client)
# ===========================================================================


class TestHierarchicalResolve:
    """Test the full resolve path: search → expand → temporal filter."""

    DIM = 32

    def test_resolve_without_kg_client_returns_empty(self) -> None:
        from src.embeddings.hierarchical import HierarchicalIndex

        facts = _make_sample_facts()
        embs = _random_embeddings(len(facts), self.DIM)

        hier = HierarchicalIndex(
            embedding_dim=self.DIM, index_type="Flat", use_gpu=False
        )
        hier.build_all_levels(facts, embs)

        q = np.random.randn(self.DIM).astype(np.float32)
        result = hier.resolve(q, level=0, temporal_mode="CURRENT", kg_client=None)
        assert result == []

    def test_resolve_with_mock_kg_client(self) -> None:
        from src.embeddings.hierarchical import HierarchicalIndex

        facts = _make_sample_facts()
        embs = _random_embeddings(len(facts), self.DIM)

        hier = HierarchicalIndex(
            embedding_dim=self.DIM, index_type="Flat", use_gpu=False
        )
        hier.build_all_levels(facts, embs)

        # Create a mock KG client
        fact_map = {f.fact_id: f for f in facts}
        mock_client = MagicMock()
        mock_client.get_fact_by_id.side_effect = lambda fid: fact_map.get(fid)

        q = np.random.randn(self.DIM).astype(np.float32)
        result = hier.resolve(
            q, level=0, temporal_mode="CURRENT", kg_client=mock_client, top_k=6
        )

        # CURRENT mode: only facts with valid_to=None
        for fact in result:
            assert fact.temporal.valid_to is None


# ===========================================================================
# Integration: fact_to_text
# ===========================================================================


class TestFactToText:
    """Verify the text representation used for SapBERT encoding."""

    def test_fact_text_format(self) -> None:
        from src.embeddings.sapbert import _fact_to_text
        fact = _make_fact()
        text = _fact_to_text(fact)
        assert text == "Gefitinib INHIBITS EGFR"

    def test_different_relation(self) -> None:
        from src.embeddings.sapbert import _fact_to_text
        fact = _make_fact(relation="TREATS", obj=_ENTITY_DISEASE)
        text = _fact_to_text(fact)
        assert text == "Gefitinib TREATS NSCLC"


# ===========================================================================
# FAISSFactIndex with IP metric
# ===========================================================================


class TestFAISSInnerProduct:
    """Verify inner-product metric works correctly."""

    DIM = 32

    def test_ip_search(self) -> None:
        from src.embeddings.faiss_index import FAISSFactIndex

        idx = FAISSFactIndex(
            embedding_dim=self.DIM,
            index_type="Flat",
            metric="IP",
            use_gpu=False,
        )
        embs = _random_embeddings(20, self.DIM)
        ids = [f"f_{i}" for i in range(20)]
        idx.build_index(embs, ids)

        results = idx.search(embs[5], top_k=1)
        # The closest by IP should be itself
        assert results[0][0] == "f_5"


# ===========================================================================
# from_config factory methods
# ===========================================================================


class TestFactories:
    """Verify from_config class methods with the default YAML."""

    def test_faiss_from_config(self, default_config: FRLMConfig) -> None:
        from src.embeddings.faiss_index import FAISSFactIndex
        # from_config should not raise
        idx = FAISSFactIndex.from_config(default_config.faiss)
        assert idx.embedding_dim == 768

    def test_hierarchical_from_config(self, default_config: FRLMConfig) -> None:
        from src.embeddings.hierarchical import HierarchicalIndex
        hier = HierarchicalIndex.from_config(default_config.faiss)
        assert hier._embedding_dim == 768
