"""
Tests for Knowledge Graph operations.

Tests:
- Hash key generation and config alignment
- Temporal envelope logic
- Neo4j config validation
- TemporalResolver (resolve_current, resolve_at, resolve_history)
- Temporal consistency validation (overlaps, gaps, multiple current)
- Neo4j client (mocked driver: CRUD, version chains, bulk import)
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, load_config
from src.kg.neo4j_client import Neo4jClient
from src.kg.schema import (
    BiomedicalEntity,
    Fact,
    FactCluster,
    Relation,
    RelationType,
    TemporalEnvelope,
    compute_fact_id,
)
from src.kg.temporal import TemporalResolver


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def default_config() -> FRLMConfig:
    return load_config()


def _entity(
    cid: str = "C0001", label: str = "Pembrolizumab", etype: str = "Drug",
) -> BiomedicalEntity:
    return BiomedicalEntity(
        id=f"ent-{cid}", label=label, entity_type=etype,
        canonical_id=cid, source_ontology="UMLS",
    )


def _fact(
    subj: str = "C0001",
    rel: str = "TREATS",
    obj: str = "C0002",
    valid_from: date = date(2024, 1, 1),
    valid_to: Optional[date] = None,
    confidence: float = 0.95,
    source: str = "PMID:12345",
) -> Fact:
    return Fact(
        subject=_entity(subj, f"Entity-{subj}", "Drug"),
        relation=Relation(type=rel),
        object=_entity(obj, f"Entity-{obj}", "Disease"),
        temporal=TemporalEnvelope(valid_from=valid_from, valid_to=valid_to),
        source=source,
        confidence=confidence,
    )


# ===========================================================================
# Hash key generation
# ===========================================================================


class TestFactHash:
    """Test content-addressable hash generation for facts."""

    def test_sha256_basic(self) -> None:
        separator = "||"
        payload = separator.join(["C0001", "TREATS", "C0002", "2024-01-01"])
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert len(expected) == 64
        assert expected == hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def test_hash_deterministic(self) -> None:
        sep = "||"
        p = sep.join(["C0001", "TREATS", "C0002", "2024-01-01"])
        h1 = hashlib.sha256(p.encode("utf-8")).hexdigest()
        h2 = hashlib.sha256(p.encode("utf-8")).hexdigest()
        assert h1 == h2

    def test_hash_changes_with_different_inputs(self) -> None:
        sep = "||"
        h1 = hashlib.sha256(sep.join(["C0001", "TREATS", "C0002", "2024-01-01"]).encode()).hexdigest()
        h2 = hashlib.sha256(sep.join(["C0001", "TREATS", "C0003", "2024-01-01"]).encode()).hexdigest()
        assert h1 != h2

    def test_hash_changes_with_timestamp(self) -> None:
        sep = "||"
        h1 = hashlib.sha256(sep.join(["C0001", "TREATS", "C0002", "2024-01-01"]).encode()).hexdigest()
        h2 = hashlib.sha256(sep.join(["C0001", "TREATS", "C0002", "2024-06-15"]).encode()).hexdigest()
        assert h1 != h2

    def test_config_hash_settings(self, default_config: FRLMConfig) -> None:
        assert default_config.neo4j.graph_schema.hash_algorithm == "sha256"
        assert default_config.neo4j.graph_schema.hash_separator == "||"

    def test_custom_separator(self) -> None:
        p1 = "||".join(["A", "B", "C", "D"])
        p2 = "::".join(["A", "B", "C", "D"])
        h1 = hashlib.sha256(p1.encode()).hexdigest()
        h2 = hashlib.sha256(p2.encode()).hexdigest()
        assert h1 != h2

    def test_compute_fact_id_matches_fact_model(self) -> None:
        fact = _fact("C0001", "TREATS", "C0002", date(2024, 1, 1))
        standalone = compute_fact_id("C0001", "TREATS", "C0002", date(2024, 1, 1))
        assert fact.fact_id == standalone


# ===========================================================================
# Temporal envelope logic
# ===========================================================================


class TestTemporalEnvelope:
    """Test temporal fact envelope conventions."""

    def test_current_fact_has_null_valid_to(self) -> None:
        fact = _fact(valid_to=None)
        assert fact.temporal.valid_to is None
        assert fact.is_current

    def test_superseded_fact_has_valid_to(self) -> None:
        fact = _fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        assert fact.temporal.valid_to is not None
        assert fact.temporal.valid_from < fact.temporal.valid_to

    def test_temporal_modes_in_config(self, default_config: FRLMConfig) -> None:
        modes = default_config.model.retrieval_head.temporal.mode_names
        assert len(modes) == 3
        assert "CURRENT" in modes
        assert "AT_TIMESTAMP" in modes
        assert "HISTORY" in modes

    def test_version_chain_type(self, default_config: FRLMConfig) -> None:
        assert default_config.neo4j.graph_schema.version_chain_type == "SUPERSEDES"


# ===========================================================================
# Neo4j config validation
# ===========================================================================


class TestNeo4jConfig:
    """Test Neo4j connection configuration."""

    def test_default_uri(self, default_config: FRLMConfig) -> None:
        assert default_config.neo4j.uri == "bolt://localhost:7687"

    def test_default_database(self, default_config: FRLMConfig) -> None:
        assert default_config.neo4j.database == "neo4j"

    def test_batch_settings(self, default_config: FRLMConfig) -> None:
        assert default_config.neo4j.batch.import_batch_size == 5000
        assert default_config.neo4j.batch.query_batch_size == 1000
        assert default_config.neo4j.batch.max_retries == 3

    def test_schema_labels(self, default_config: FRLMConfig) -> None:
        schema = default_config.neo4j.graph_schema
        assert schema.fact_label == "Fact"
        assert schema.entity_label == "Entity"
        assert schema.relation_type == "HAS_FACT"

    def test_connection_timeout(self, default_config: FRLMConfig) -> None:
        assert default_config.neo4j.connection_timeout == 30
        assert default_config.neo4j.max_transaction_retry_time == 30

    def test_pool_size(self, default_config: FRLMConfig) -> None:
        assert default_config.neo4j.max_connection_pool_size == 50


# ===========================================================================
# TemporalResolver — resolve_current
# ===========================================================================


class TestResolverCurrent:
    """Test TemporalResolver.resolve_current."""

    def test_filters_to_current_only(self) -> None:
        current = _fact(valid_from=date(2024, 6, 1), valid_to=None)
        superseded = _fact(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        result = TemporalResolver.resolve_current([current, superseded])
        assert len(result) == 1
        assert result[0].fact_id == current.fact_id

    def test_empty_list(self) -> None:
        assert TemporalResolver.resolve_current([]) == []

    def test_all_current(self) -> None:
        f1 = _fact(subj="C0001", valid_from=date(2024, 1, 1))
        f2 = _fact(subj="C0099", valid_from=date(2024, 3, 1))
        result = TemporalResolver.resolve_current([f1, f2])
        assert len(result) == 2

    def test_none_current(self) -> None:
        f1 = _fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        f2 = _fact(
            subj="C0099", valid_from=date(2023, 6, 1), valid_to=date(2024, 6, 1),
        )
        result = TemporalResolver.resolve_current([f1, f2])
        assert len(result) == 0


# ===========================================================================
# TemporalResolver — resolve_at
# ===========================================================================


class TestResolverAt:
    """Test TemporalResolver.resolve_at."""

    def test_within_window(self) -> None:
        f = _fact(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        result = TemporalResolver.resolve_at([f], date(2024, 3, 15))
        assert len(result) == 1

    def test_before_window(self) -> None:
        f = _fact(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        result = TemporalResolver.resolve_at([f], date(2023, 12, 31))
        assert len(result) == 0

    def test_at_window_end_excluded(self) -> None:
        f = _fact(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        result = TemporalResolver.resolve_at([f], date(2024, 6, 1))
        assert len(result) == 0

    def test_at_window_start_included(self) -> None:
        f = _fact(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        result = TemporalResolver.resolve_at([f], date(2024, 1, 1))
        assert len(result) == 1

    def test_current_fact_valid_at_future(self) -> None:
        f = _fact(valid_from=date(2024, 1, 1), valid_to=None)
        result = TemporalResolver.resolve_at([f], date(2099, 12, 31))
        assert len(result) == 1

    def test_multiple_facts_at_timestamp(self) -> None:
        f1 = _fact(subj="C0001", valid_from=date(2024, 1, 1), valid_to=None)
        f2 = _fact(
            subj="C0099", valid_from=date(2023, 1, 1), valid_to=date(2025, 1, 1),
        )
        f3 = _fact(
            subj="C0088", valid_from=date(2025, 1, 1), valid_to=None,
        )
        result = TemporalResolver.resolve_at([f1, f2, f3], date(2024, 6, 1))
        assert len(result) == 2
        ids = {r.subject.canonical_id for r in result}
        assert ids == {"C0001", "C0099"}


# ===========================================================================
# TemporalResolver — resolve_history
# ===========================================================================


class TestResolverHistory:
    """Test TemporalResolver.resolve_history."""

    def test_returns_all_ordered(self) -> None:
        f1 = _fact(valid_from=date(2024, 6, 1), valid_to=None)
        f2 = _fact(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        f3 = _fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        result = TemporalResolver.resolve_history([f1, f2, f3])
        assert len(result) == 3
        assert result[0].temporal.valid_from == date(2023, 1, 1)
        assert result[1].temporal.valid_from == date(2024, 1, 1)
        assert result[2].temporal.valid_from == date(2024, 6, 1)

    def test_empty_list(self) -> None:
        assert TemporalResolver.resolve_history([]) == []


# ===========================================================================
# TemporalResolver — dispatch
# ===========================================================================


class TestResolverDispatch:
    """Test TemporalResolver.resolve dispatcher."""

    def test_dispatch_current(self) -> None:
        resolver = TemporalResolver()
        f = _fact(valid_to=None)
        result = resolver.resolve([f], "CURRENT")
        assert len(result) == 1

    def test_dispatch_at_timestamp(self) -> None:
        resolver = TemporalResolver()
        f = _fact(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        result = resolver.resolve([f], "AT_TIMESTAMP", timestamp=date(2024, 3, 1))
        assert len(result) == 1

    def test_dispatch_history(self) -> None:
        resolver = TemporalResolver()
        f = _fact()
        result = resolver.resolve([f], "HISTORY")
        assert len(result) == 1

    def test_dispatch_unknown_mode(self) -> None:
        resolver = TemporalResolver()
        with pytest.raises(ValueError, match="Unknown temporal mode"):
            resolver.resolve([], "FUTURE")

    def test_dispatch_at_timestamp_missing_timestamp(self) -> None:
        resolver = TemporalResolver()
        with pytest.raises(ValueError, match="timestamp is required"):
            resolver.resolve([], "AT_TIMESTAMP")

    def test_dispatch_case_insensitive(self) -> None:
        resolver = TemporalResolver()
        f = _fact(valid_to=None)
        result = resolver.resolve([f], "current")
        assert len(result) == 1


# ===========================================================================
# Temporal consistency validation
# ===========================================================================


class TestTemporalConsistency:
    """Test TemporalResolver.validate_temporal_consistency."""

    def test_clean_chain_no_errors(self) -> None:
        f1 = _fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        f2 = _fact(valid_from=date(2024, 1, 1), valid_to=None)
        errors = TemporalResolver.validate_temporal_consistency([f1, f2])
        assert errors == []

    def test_multiple_current_detected(self) -> None:
        f1 = _fact(valid_from=date(2024, 1, 1), valid_to=None)
        f2 = _fact(valid_from=date(2024, 6, 1), valid_to=None)
        errors = TemporalResolver.validate_temporal_consistency([f1, f2])
        assert any("current versions" in e for e in errors)

    def test_overlapping_windows_detected(self) -> None:
        f1 = _fact(valid_from=date(2024, 1, 1), valid_to=date(2024, 8, 1))
        f2 = _fact(valid_from=date(2024, 6, 1), valid_to=None)
        errors = TemporalResolver.validate_temporal_consistency([f1, f2])
        assert any("Overlapping" in e for e in errors)

    def test_gap_in_chain_detected(self) -> None:
        f1 = _fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        # Gap: f1 ends 2024-01-01, f2 starts 2024-03-01
        f2 = _fact(valid_from=date(2024, 3, 1), valid_to=None)
        errors = TemporalResolver.validate_temporal_consistency([f1, f2])
        assert any("Gap" in e for e in errors)

    def test_no_overlap_adjacent_windows(self) -> None:
        f1 = _fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        f2 = _fact(valid_from=date(2024, 1, 1), valid_to=date(2025, 1, 1))
        f3 = _fact(valid_from=date(2025, 1, 1), valid_to=None)
        errors = TemporalResolver.validate_temporal_consistency([f1, f2, f3])
        assert errors == []

    def test_different_families_independent(self) -> None:
        # Two different logical facts — no cross-validation
        f1 = _fact(subj="C0001", obj="C0002", valid_from=date(2024, 1, 1), valid_to=None)
        f2 = _fact(subj="C0001", obj="C0099", valid_from=date(2024, 1, 1), valid_to=None)
        errors = TemporalResolver.validate_temporal_consistency([f1, f2])
        assert errors == []

    def test_empty_facts_no_errors(self) -> None:
        errors = TemporalResolver.validate_temporal_consistency([])
        assert errors == []

    def test_single_current_fact_no_errors(self) -> None:
        f = _fact(valid_from=date(2024, 1, 1), valid_to=None)
        errors = TemporalResolver.validate_temporal_consistency([f])
        assert errors == []


# ===========================================================================
# Neo4j client — construction and config
# ===========================================================================


class TestNeo4jClientConstruction:
    """Test Neo4jClient construction and config-based factory."""

    def test_default_construction(self) -> None:
        client = Neo4jClient()
        assert client._uri == "bolt://localhost:7687"
        assert client._database == "frlm"
        assert client._entity_label == "Entity"
        assert client._fact_label == "Fact"

    def test_from_config(self, default_config: FRLMConfig) -> None:
        client = Neo4jClient.from_config(default_config)
        assert client._uri == "bolt://localhost:7687"
        assert client._database == "neo4j"
        assert client._entity_label == "Entity"
        assert client._fact_label == "Fact"
        assert client._version_chain_type == "SUPERSEDES"
        assert client._batch_size == 5000
        assert client._max_retries == 3

    def test_driver_not_connected_raises(self) -> None:
        client = Neo4jClient()
        with pytest.raises(RuntimeError, match="not connected"):
            _ = client.driver

    def test_format_cypher(self) -> None:
        client = Neo4jClient(entity_label="Ent", fact_label="Fct")
        result = client._format_cypher("MATCH (e:{entity_label})-[:{relation_type}]->(f:{fact_label})")
        assert "Ent" in result
        assert "Fct" in result
        assert "HAS_FACT" in result

    def test_cypher_create_fact_includes_metadata(self) -> None:
        from src.kg.neo4j_client import CYPHER_CREATE_FACT, CYPHER_BULK_MERGE_FACTS
        assert "metadata" in CYPHER_CREATE_FACT
        assert "metadata" in CYPHER_BULK_MERGE_FACTS


# ===========================================================================
# Neo4j client — mocked CRUD operations
# ===========================================================================


class TestNeo4jClientCRUD:
    """Test Neo4j client CRUD with mocked driver."""

    @pytest.fixture()
    def mock_client(self) -> Neo4jClient:
        """Create a client with a mocked driver."""
        client = Neo4jClient(
            entity_label="Entity",
            fact_label="Fact",
            relation_type="HAS_FACT",
            version_chain_type="SUPERSEDES",
            max_retries=1,
            retry_delay=0.0,
        )
        # Mock the driver
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Configure execute_write and execute_read to invoke the function
        def run_tx_func(func):
            mock_tx = MagicMock()
            mock_tx.run.return_value = []
            return func(mock_tx)

        mock_session.execute_write.side_effect = run_tx_func
        mock_session.execute_read.side_effect = run_tx_func

        client._driver = mock_driver
        return client

    def test_create_entity(self, mock_client: Neo4jClient) -> None:
        entity = _entity("C0001")
        result = mock_client.create_entity(entity)
        assert result.canonical_id == "C0001"

    def test_create_fact(self, mock_client: Neo4jClient) -> None:
        fact = _fact()
        result = mock_client.create_fact(fact)
        assert result.fact_id == fact.fact_id

    def test_get_fact_by_id_not_found(self, mock_client: Neo4jClient) -> None:
        result = mock_client.get_fact_by_id("nonexistent_hash")
        assert result is None

    def test_get_entity_not_found(self, mock_client: Neo4jClient) -> None:
        result = mock_client.get_entity("C9999")
        assert result is None

    def test_create_fact_version_missing_old(self, mock_client: Neo4jClient) -> None:
        new_fact = _fact(valid_from=date(2025, 1, 1))
        with pytest.raises(ValueError, match="Old fact not found"):
            mock_client.create_fact_version("nonexistent_id", new_fact)

    def test_bulk_import_empty(self, mock_client: Neo4jClient) -> None:
        count = mock_client.bulk_import_facts([])
        assert count == 0

    def test_bulk_import_facts(self, mock_client: Neo4jClient) -> None:
        facts = [
            _fact("C0001", "TREATS", "C0002"),
            _fact("C0001", "INHIBITS", "C0003"),
        ]
        count = mock_client.bulk_import_facts(facts)
        assert count == 2

    def test_temporal_filter_current(self, mock_client: Neo4jClient) -> None:
        current = _fact(valid_to=None)
        old = _fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        result = Neo4jClient._apply_temporal_filter([current, old], "CURRENT")
        assert len(result) == 1
        assert result[0].fact_id == current.fact_id

    def test_temporal_filter_at_timestamp(self, mock_client: Neo4jClient) -> None:
        f = _fact(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        result = Neo4jClient._apply_temporal_filter([f], "AT_TIMESTAMP", date(2024, 3, 1))
        assert len(result) == 1

    def test_temporal_filter_history(self, mock_client: Neo4jClient) -> None:
        f1 = _fact(valid_from=date(2024, 6, 1))
        f2 = _fact(valid_from=date(2024, 1, 1), valid_to=date(2024, 6, 1))
        result = Neo4jClient._apply_temporal_filter([f1, f2], "HISTORY")
        assert len(result) == 2
        assert result[0].temporal.valid_from < result[1].temporal.valid_from

    def test_temporal_filter_unknown_mode(self, mock_client: Neo4jClient) -> None:
        with pytest.raises(ValueError, match="Unknown temporal_mode"):
            Neo4jClient._apply_temporal_filter([], "FUTURE")

    def test_temporal_filter_at_timestamp_missing(self, mock_client: Neo4jClient) -> None:
        with pytest.raises(ValueError, match="timestamp is required"):
            Neo4jClient._apply_temporal_filter([], "AT_TIMESTAMP")

    def test_neo4j_node_to_fact_with_metadata(self, mock_client: Neo4jClient) -> None:
        """Verify metadata is deserialized from JSON string."""
        node = {
            "fact_id": "abc123",
            "subject_id": "C0001",
            "subject_label": "Drug-A",
            "relation_type": "TREATS",
            "object_id": "C0002",
            "object_label": "Disease-B",
            "valid_from": "2024-01-01",
            "valid_to": None,
            "source": "PMID:12345",
            "confidence": 0.95,
            "metadata": json.dumps({"evidence": "some text", "model": "claude"}),
        }
        fact = mock_client._neo4j_node_to_fact(node)
        assert fact.metadata == {"evidence": "some text", "model": "claude"}

    def test_neo4j_node_to_fact_missing_metadata(self, mock_client: Neo4jClient) -> None:
        """Verify missing metadata defaults to empty dict."""
        node = {
            "fact_id": "abc123",
            "subject_id": "C0001",
            "subject_label": "Drug-A",
            "relation_type": "TREATS",
            "object_id": "C0002",
            "object_label": "Disease-B",
            "valid_from": "2024-01-01",
            "valid_to": None,
            "source": "PMID:12345",
            "confidence": 0.95,
        }
        fact = mock_client._neo4j_node_to_fact(node)
        assert fact.metadata == {}

    def test_fact_properties_include_metadata(self) -> None:
        """Verify to_neo4j_properties includes metadata for Cypher params."""
        fact = _fact()
        props = fact.to_neo4j_properties()
        assert "metadata" in props
        assert isinstance(props["metadata"], str)
        parsed = json.loads(props["metadata"])
        assert isinstance(parsed, dict)


# ===========================================================================
# Version chain integrity (schema-level)
# ===========================================================================


class TestVersionChainIntegrity:
    """Validate version chain logic at the schema level."""

    def test_family_key_links_versions(self) -> None:
        v1 = _fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        v2 = _fact(valid_from=date(2024, 1, 1), valid_to=None)
        assert v1.family_key == v2.family_key

    def test_fact_ids_differ_across_versions(self) -> None:
        v1 = _fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        v2 = _fact(valid_from=date(2024, 1, 1), valid_to=None)
        assert v1.fact_id != v2.fact_id

    def test_family_key_differs_for_different_triples(self) -> None:
        f1 = _fact(subj="C0001", rel="TREATS", obj="C0002")
        f2 = _fact(subj="C0001", rel="TREATS", obj="C0099")
        assert f1.family_key != f2.family_key

    def test_family_key_differs_for_different_relations(self) -> None:
        f1 = _fact(rel="TREATS")
        f2 = _fact(rel="CAUSES")
        assert f1.family_key != f2.family_key

    def test_three_version_chain(self) -> None:
        v1 = _fact(valid_from=date(2022, 1, 1), valid_to=date(2023, 1, 1))
        v2 = _fact(valid_from=date(2023, 1, 1), valid_to=date(2024, 1, 1))
        v3 = _fact(valid_from=date(2024, 1, 1), valid_to=None)

        # All share a family key
        assert v1.family_key == v2.family_key == v3.family_key
        # All have distinct fact_ids
        assert len({v1.fact_id, v2.fact_id, v3.fact_id}) == 3
        # Temporal consistency
        errors = TemporalResolver.validate_temporal_consistency([v1, v2, v3])
        assert errors == []
