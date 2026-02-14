"""
Knowledge Graph module.

Provides Neo4j-backed temporal knowledge graph with CRUD operations,
temporal resolution logic, and KG population pipelines.
"""

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
from src.kg.neo4j_client import Neo4jClient
from src.kg.temporal import TemporalResolver

__all__ = [
    "BiomedicalEntity",
    "ClusterType",
    "Fact",
    "FactCluster",
    "Relation",
    "RelationType",
    "TemporalEnvelope",
    "compute_fact_id",
    "Neo4jClient",
    "TemporalResolver",
]