"""
Tests for FRLM configuration schema and validation.

Validates:
- FRLMConfig loads config/default.yaml correctly
- Pydantic validators catch invalid values
- Environment variable overrides work
- Dot-notation overrides work
- Cross-validation catches mismatched dimensions
- KG schema models (BiomedicalEntity, Relation, Fact, TemporalEnvelope, FactCluster)
"""

import hashlib
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, load_config
from pydantic import ValidationError
from src.kg.schema import (
    BiomedicalEntity,
    ClusterType,
    Fact,
    FactCluster,
    Relation,
    RelationType,
    TemporalEnvelope,
    compute_fact_id,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def default_config() -> FRLMConfig:
    """Load the default configuration once per module."""
    return load_config()


@pytest.fixture()
def config_path() -> Path:
    """Return absolute path to default.yaml."""
    return Path(__file__).resolve().parent.parent / "config" / "default.yaml"


# ===========================================================================
# Config loading basics
# ===========================================================================


class TestConfigLoading:
    """Verify default.yaml loads and produces a fully populated config."""

    def test_load_returns_frlm_config(self, default_config: FRLMConfig) -> None:
        assert isinstance(default_config, FRLMConfig)

    def test_project_metadata(self, default_config: FRLMConfig) -> None:
        assert default_config.project.name == "frlm"
        assert default_config.project.version == "0.1.0"
        assert default_config.project.seed == 42
        assert default_config.project.deterministic is True

    def test_all_sections_present(self, default_config: FRLMConfig) -> None:
        sections = [
            "project", "paths", "model", "sapbert", "neo4j", "faiss",
            "extraction", "labeling", "training", "loss", "deepspeed",
            "wandb", "evaluation", "inference", "serving", "logging",
        ]
        for section in sections:
            assert hasattr(default_config, section), f"Missing section: {section}"
            assert getattr(default_config, section) is not None

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(config_path=tmp_path / "nonexistent.yaml")

    def test_default_yaml_exists(self, config_path: Path) -> None:
        assert config_path.exists()

    def test_model_backbone_values(self, default_config: FRLMConfig) -> None:
        bb = default_config.model.backbone
        assert bb.name == "stanford-crfm/BioMedLM"
        assert bb.hidden_dim == 2560
        assert bb.vocab_size == 50257
        assert bb.max_seq_length == 1024

    def test_loss_weights(self, default_config: FRLMConfig) -> None:
        assert default_config.loss.router_weight == 1.0
        assert default_config.loss.retrieval_weight == 2.0
        assert default_config.loss.generation_weight == 1.0
        assert default_config.loss.contrastive_temperature == 0.07

    def test_faiss_config(self, default_config: FRLMConfig) -> None:
        assert default_config.faiss.embedding_dim == 768
        assert default_config.faiss.nlist == 4096
        assert default_config.faiss.nprobe == 64


# ===========================================================================
# Field-level validation
# ===========================================================================


class TestFieldValidation:
    """Test that Pydantic validators reject invalid values."""

    def test_invalid_dtype_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dtype"):
            FRLMConfig(model={"backbone": {"dtype": "int8"}})

    def test_dropout_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dropout"):
            FRLMConfig(model={"router_head": {"dropout": -0.1}})

    def test_dropout_at_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dropout"):
            FRLMConfig(model={"router_head": {"dropout": 1.0}})

    def test_splits_not_summing_to_one(self) -> None:
        with pytest.raises(ValidationError, match="splits must sum to 1.0"):
            FRLMConfig(training={"splits": {"train": 0.5, "validation": 0.2, "test": 0.1}})

    def test_invalid_scheduler(self) -> None:
        with pytest.raises(ValidationError, match="scheduler"):
            FRLMConfig(training={"router": {"scheduler": "exponential"}})

    def test_invalid_pool_strategy(self) -> None:
        with pytest.raises(ValidationError, match="pool_strategy"):
            FRLMConfig(sapbert={"pool_strategy": "attention"})

    def test_temperature_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="contrastive_temperature"):
            FRLMConfig(loss={"contrastive_temperature": 0.0})

    def test_fp16_and_bf16_both_true(self) -> None:
        with pytest.raises(ValidationError, match="fp16|bf16|Cannot enable both"):
            FRLMConfig(training={"fp16": True, "bf16": True})

    def test_invalid_log_level(self) -> None:
        with pytest.raises(ValidationError, match="level"):
            FRLMConfig(logging={"level": "VERBOSE"})

    def test_granularity_names_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="level_names"):
            FRLMConfig(
                model={"retrieval_head": {"granularity": {"num_levels": 4, "level_names": ["a", "b"]}}}
            )

    def test_temporal_modes_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="mode_names"):
            FRLMConfig(
                model={"retrieval_head": {"temporal": {"num_modes": 3, "mode_names": ["A"]}}}
            )

    def test_invalid_faiss_metric(self) -> None:
        with pytest.raises(ValidationError, match="metric"):
            FRLMConfig(faiss={"metric": "cosine"})


# ===========================================================================
# Environment variable overrides
# ===========================================================================


class TestEnvironmentOverrides:
    """Verify sensitive values are overridden from env vars."""

    def test_neo4j_password_override(self) -> None:
        with patch.dict(os.environ, {"FRLM_NEO4J_PASSWORD": "secret_123"}):
            cfg = load_config()
            assert cfg.neo4j.password == "secret_123"

    def test_neo4j_password_default_without_env(self) -> None:
        env = os.environ.copy()
        env.pop("FRLM_NEO4J_PASSWORD", None)
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()
            assert cfg.neo4j.password == "CHANGE_ME"

    def test_wandb_api_key_override(self) -> None:
        with patch.dict(os.environ, {"WANDB_API_KEY": "wb_key_xyz"}):
            cfg = load_config()
            assert cfg.wandb.api_key == "wb_key_xyz"

    def test_anthropic_api_key_overrides_relation(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            cfg = load_config()
            assert cfg.extraction.relation.api_key == "sk-ant-test"

    def test_anthropic_api_key_overrides_labeling(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-label"}):
            cfg = load_config()
            assert cfg.labeling.api_key == "sk-ant-label"


# ===========================================================================
# Dot-notation overrides
# ===========================================================================


class TestDotNotationOverrides:
    """Verify dot-notation overrides in load_config."""

    def test_single_override(self) -> None:
        cfg = load_config(overrides={"project.seed": 99})
        assert cfg.project.seed == 99

    def test_deeply_nested_override(self) -> None:
        cfg = load_config(overrides={"training.router.epochs": 50})
        assert cfg.training.router.epochs == 50

    def test_multiple_overrides(self) -> None:
        cfg = load_config(overrides={
            "project.seed": 7,
            "training.router.epochs": 25,
            "loss.router_weight": 3.0,
        })
        assert cfg.project.seed == 7
        assert cfg.training.router.epochs == 25
        assert cfg.loss.router_weight == 3.0

    def test_override_preserves_other_fields(self) -> None:
        cfg = load_config(overrides={"project.seed": 123})
        assert cfg.project.name == "frlm"
        assert cfg.project.version == "0.1.0"


# ===========================================================================
# Cross-field validation
# ===========================================================================


class TestCrossValidation:
    """Verify cross-field dimension checks."""

    def test_router_input_dim_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="router_head.input_dim"):
            FRLMConfig(model={"backbone": {"hidden_dim": 2560}, "router_head": {"input_dim": 1024}})

    def test_semantic_input_dim_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="retrieval_head.semantic.input_dim"):
            FRLMConfig(model={"backbone": {"hidden_dim": 2560}, "retrieval_head": {"semantic": {"input_dim": 512}}})

    def test_generation_output_dim_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="generation_head.output_dim"):
            FRLMConfig(model={"backbone": {"vocab_size": 50257}, "generation_head": {"output_dim": 30000}})

    def test_faiss_embedding_dim_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="faiss.embedding_dim"):
            FRLMConfig(sapbert={"embedding_dim": 768}, faiss={"embedding_dim": 512})

    def test_semantic_output_dim_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="retrieval_head.semantic.output_dim"):
            FRLMConfig(sapbert={"embedding_dim": 768}, model={"retrieval_head": {"semantic": {"output_dim": 256}}})

    def test_default_passes_cross_validation(self, default_config: FRLMConfig) -> None:
        hd = default_config.model.backbone.hidden_dim
        assert default_config.model.router_head.input_dim == hd
        assert default_config.model.retrieval_head.semantic.input_dim == hd
        assert default_config.model.retrieval_head.granularity.input_dim == hd
        assert default_config.model.retrieval_head.temporal.input_dim == hd
        assert default_config.model.generation_head.input_dim == hd
        assert default_config.model.generation_head.output_dim == default_config.model.backbone.vocab_size
        assert default_config.model.retrieval_head.semantic.output_dim == default_config.sapbert.embedding_dim
        assert default_config.faiss.embedding_dim == default_config.sapbert.embedding_dim


# ===========================================================================
# KG Schema — fixtures
# ===========================================================================


def _make_entity(
    cid: str = "C0001",
    label: str = "Pembrolizumab",
    entity_type: str = "Drug",
) -> BiomedicalEntity:
    return BiomedicalEntity(
        id=f"ent-{cid}",
        label=label,
        entity_type=entity_type,
        canonical_id=cid,
        source_ontology="UMLS",
    )


def _make_fact(
    subject_cid: str = "C0001",
    relation: str = "TREATS",
    object_cid: str = "C0002",
    valid_from: date = date(2024, 1, 1),
    valid_to: Optional[date] = None,
    confidence: float = 0.95,
) -> Fact:
    return Fact(
        subject=_make_entity(subject_cid, f"Entity-{subject_cid}", "Drug"),
        relation=Relation(type=relation),
        object=_make_entity(object_cid, f"Entity-{object_cid}", "Disease"),
        temporal=TemporalEnvelope(valid_from=valid_from, valid_to=valid_to),
        source="PMID:12345678",
        confidence=confidence,
    )


# ===========================================================================
# BiomedicalEntity tests
# ===========================================================================


class TestBiomedicalEntity:
    """Validate BiomedicalEntity model."""

    def test_creation(self) -> None:
        entity = _make_entity()
        assert entity.canonical_id == "C0001"
        assert entity.label == "Pembrolizumab"
        assert entity.entity_type == "Drug"
        assert entity.source_ontology == "UMLS"

    def test_equality_by_canonical_id(self) -> None:
        e1 = _make_entity("C0001", "Name1", "Drug")
        e2 = _make_entity("C0001", "Name2", "Gene")
        assert e1 == e2

    def test_inequality_different_canonical_id(self) -> None:
        e1 = _make_entity("C0001")
        e2 = _make_entity("C0002")
        assert e1 != e2

    def test_hash_by_canonical_id(self) -> None:
        e1 = _make_entity("C0001", "Name1", "Drug")
        e2 = _make_entity("C0001", "Name2", "Gene")
        assert hash(e1) == hash(e2)
        assert len({e1, e2}) == 1

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BiomedicalEntity(
                id="", label="X", entity_type="Drug",
                canonical_id="C0001", source_ontology="UMLS",
            )

    def test_empty_canonical_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BiomedicalEntity(
                id="e1", label="X", entity_type="Drug",
                canonical_id="", source_ontology="UMLS",
            )


# ===========================================================================
# RelationType tests
# ===========================================================================


class TestRelationType:
    """Validate RelationType enum and Relation model."""

    def test_all_30_types_defined(self) -> None:
        assert len(RelationType) == 30

    def test_string_coercion(self) -> None:
        r = Relation(type="treats")
        assert r.type == RelationType.TREATS

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unknown relation type"):
            Relation(type="HEALS")

    def test_case_insensitive(self) -> None:
        r = Relation(type="Inhibits")
        assert r.type == RelationType.INHIBITS

    def test_enum_values_match(self) -> None:
        expected = [
            "INHIBITS", "ACTIVATES", "TREATS", "CAUSES", "BINDS_TO",
            "METABOLIZED_BY", "ASSOCIATED_WITH", "UPREGULATES", "DOWNREGULATES",
            "CONTRAINDICATED_WITH", "SYNERGISTIC_WITH", "SUBSTRATE_OF",
            "PRODUCT_OF", "BIOMARKER_FOR", "EXPRESSED_IN", "LOCATED_IN",
            "PART_OF", "PRECURSOR_OF", "ANALOG_OF", "TARGET_OF",
            "INDUCES", "PREVENTS", "DIAGNOSES", "PROGNOSTIC_FOR",
            "RESISTANT_TO", "SENSITIVE_TO", "INTERACTS_WITH", "TRANSPORTS",
            "CATALYZES", "ENCODES",
        ]
        assert [rt.value for rt in RelationType] == expected


# ===========================================================================
# TemporalEnvelope tests
# ===========================================================================


class TestTemporalEnvelope:
    """Validate TemporalEnvelope model."""

    def test_current_fact(self) -> None:
        te = TemporalEnvelope(valid_from=date(2024, 1, 1))
        assert te.is_current
        assert te.valid_to is None

    def test_superseded_fact(self) -> None:
        te = TemporalEnvelope(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        assert not te.is_current

    def test_inverted_window_rejected(self) -> None:
        with pytest.raises(ValidationError, match="valid_from.*before.*valid_to"):
            TemporalEnvelope(valid_from=date(2024, 6, 1), valid_to=date(2024, 1, 1))

    def test_same_dates_rejected(self) -> None:
        with pytest.raises(ValidationError, match="valid_from.*before.*valid_to"):
            TemporalEnvelope(valid_from=date(2024, 1, 1), valid_to=date(2024, 1, 1))

    def test_contains_within_window(self) -> None:
        te = TemporalEnvelope(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        assert te.contains(date(2024, 3, 15))

    def test_contains_at_start(self) -> None:
        te = TemporalEnvelope(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        assert te.contains(date(2024, 1, 1))

    def test_not_contains_at_end(self) -> None:
        te = TemporalEnvelope(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        assert not te.contains(date(2024, 6, 1))

    def test_not_contains_before_start(self) -> None:
        te = TemporalEnvelope(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        assert not te.contains(date(2023, 12, 31))

    def test_current_contains_future(self) -> None:
        te = TemporalEnvelope(valid_from=date(2024, 1, 1))
        assert te.contains(date(2099, 12, 31))


# ===========================================================================
# Fact ID tests
# ===========================================================================


class TestFactId:
    """Validate auto-computed SHA-256 fact_id."""

    def test_fact_id_computed_on_creation(self) -> None:
        fact = _make_fact()
        assert len(fact.fact_id) == 64
        assert fact.fact_id != ""

    def test_fact_id_deterministic(self) -> None:
        f1 = _make_fact()
        f2 = _make_fact()
        assert f1.fact_id == f2.fact_id

    def test_fact_id_matches_manual_hash(self) -> None:
        fact = _make_fact("C0001", "TREATS", "C0002", date(2024, 1, 1))
        expected_payload = "||".join(["C0001", "TREATS", "C0002", "2024-01-01"])
        expected_hash = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
        assert fact.fact_id == expected_hash

    def test_different_subject_different_id(self) -> None:
        f1 = _make_fact(subject_cid="C0001")
        f2 = _make_fact(subject_cid="C0099")
        assert f1.fact_id != f2.fact_id

    def test_different_relation_different_id(self) -> None:
        f1 = _make_fact(relation="TREATS")
        f2 = _make_fact(relation="CAUSES")
        assert f1.fact_id != f2.fact_id

    def test_different_object_different_id(self) -> None:
        f1 = _make_fact(object_cid="C0002")
        f2 = _make_fact(object_cid="C0099")
        assert f1.fact_id != f2.fact_id

    def test_different_timestamp_different_id(self) -> None:
        f1 = _make_fact(valid_from=date(2024, 1, 1))
        f2 = _make_fact(valid_from=date(2024, 6, 15))
        assert f1.fact_id != f2.fact_id

    def test_compute_fact_id_standalone(self) -> None:
        fid = compute_fact_id("C0001", "TREATS", "C0002", date(2024, 1, 1))
        expected = hashlib.sha256(
            "C0001||TREATS||C0002||2024-01-01".encode("utf-8")
        ).hexdigest()
        assert fid == expected

    def test_fact_equality_by_id(self) -> None:
        f1 = _make_fact()
        f2 = _make_fact()
        assert f1 == f2
        assert hash(f1) == hash(f2)

    def test_fact_inequality(self) -> None:
        f1 = _make_fact(subject_cid="C0001")
        f2 = _make_fact(subject_cid="C0099")
        assert f1 != f2


# ===========================================================================
# Fact model tests
# ===========================================================================


class TestFactModel:
    """Validate Fact model properties and serialization."""

    def test_is_current_property(self) -> None:
        current = _make_fact(valid_to=None)
        assert current.is_current is True

        superseded = _make_fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        assert superseded.is_current is False

    def test_family_key_same_triple(self) -> None:
        f1 = _make_fact(valid_from=date(2024, 1, 1))
        f2 = _make_fact(valid_from=date(2024, 6, 1))
        assert f1.family_key == f2.family_key

    def test_family_key_different_triple(self) -> None:
        f1 = _make_fact(subject_cid="C0001", object_cid="C0002")
        f2 = _make_fact(subject_cid="C0001", object_cid="C0099")
        assert f1.family_key != f2.family_key

    def test_to_neo4j_properties(self) -> None:
        fact = _make_fact(valid_from=date(2024, 1, 1), valid_to=None)
        props = fact.to_neo4j_properties()
        assert props["fact_id"] == fact.fact_id
        assert props["subject_id"] == "C0001"
        assert props["relation_type"] == "TREATS"
        assert props["object_id"] == "C0002"
        assert props["valid_from"] == "2024-01-01"
        assert props["valid_to"] is None
        assert props["confidence"] == 0.95

    def test_to_neo4j_properties_with_valid_to(self) -> None:
        fact = _make_fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        props = fact.to_neo4j_properties()
        assert props["valid_to"] == "2024-01-01"

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _make_fact(confidence=1.5)
        with pytest.raises(ValidationError):
            _make_fact(confidence=-0.1)

    def test_version_chain_default_empty(self) -> None:
        fact = _make_fact()
        assert fact.version_chain == []


# ===========================================================================
# FactCluster tests
# ===========================================================================


class TestFactCluster:
    """Validate FactCluster model."""

    def test_empty_cluster(self) -> None:
        cluster = FactCluster(cluster_type="entity", anchor_entity="C0001")
        assert cluster.size == 0
        assert cluster.entity_ids == []

    def test_cluster_with_facts(self) -> None:
        f1 = _make_fact("C0001", "TREATS", "C0002")
        f2 = _make_fact("C0001", "INHIBITS", "C0003")
        cluster = FactCluster(
            facts=[f1, f2],
            cluster_type="entity",
            anchor_entity="C0001",
        )
        assert cluster.size == 2
        assert "C0001" in cluster.entity_ids
        assert "C0002" in cluster.entity_ids
        assert "C0003" in cluster.entity_ids
        assert sorted(cluster.relation_types) == ["INHIBITS", "TREATS"]

    def test_cluster_type_coercion(self) -> None:
        c = FactCluster(cluster_type="RELATION", anchor_entity="C0001")
        assert c.cluster_type == ClusterType.RELATION

    def test_invalid_cluster_type(self) -> None:
        with pytest.raises(ValidationError, match="Unknown cluster type"):
            FactCluster(cluster_type="unknown", anchor_entity="C0001")