"""
KG Schema — Pydantic models for the FRLM temporal knowledge graph.

Defines:
    - BiomedicalEntity: node in the KG (UMLS-linked biomedical concept)
    - RelationType: constrained enum of ~30 biomedical relation types
    - TemporalEnvelope: (valid_from, valid_to) window for fact versioning
    - Fact: temporal triple with auto-computed SHA-256 fact_id
    - FactCluster: group of facts for hierarchical indexing
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ===========================================================================
# Relation ontology
# ===========================================================================


class RelationType(str, Enum):
    """Fixed ontology of biomedical relation types.

    Each value is the canonical string stored in Neo4j and used in
    SHA-256 fact-id computation.
    """

    INHIBITS = "INHIBITS"
    ACTIVATES = "ACTIVATES"
    TREATS = "TREATS"
    CAUSES = "CAUSES"
    BINDS_TO = "BINDS_TO"
    METABOLIZED_BY = "METABOLIZED_BY"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    UPREGULATES = "UPREGULATES"
    DOWNREGULATES = "DOWNREGULATES"
    CONTRAINDICATED_WITH = "CONTRAINDICATED_WITH"
    SYNERGISTIC_WITH = "SYNERGISTIC_WITH"
    SUBSTRATE_OF = "SUBSTRATE_OF"
    PRODUCT_OF = "PRODUCT_OF"
    BIOMARKER_FOR = "BIOMARKER_FOR"
    EXPRESSED_IN = "EXPRESSED_IN"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    PRECURSOR_OF = "PRECURSOR_OF"
    ANALOG_OF = "ANALOG_OF"
    TARGET_OF = "TARGET_OF"
    INDUCES = "INDUCES"
    PREVENTS = "PREVENTS"
    DIAGNOSES = "DIAGNOSES"
    PROGNOSTIC_FOR = "PROGNOSTIC_FOR"
    RESISTANT_TO = "RESISTANT_TO"
    SENSITIVE_TO = "SENSITIVE_TO"
    INTERACTS_WITH = "INTERACTS_WITH"
    TRANSPORTS = "TRANSPORTS"
    CATALYZES = "CATALYZES"
    ENCODES = "ENCODES"


# ===========================================================================
# Entity
# ===========================================================================


class BiomedicalEntity(BaseModel):
    """A biomedical concept node in the knowledge graph.

    Attributes
    ----------
    id : str
        Internal unique identifier (typically a UUID or sequential id).
    label : str
        Human-readable preferred name (e.g. "Pembrolizumab").
    entity_type : str
        Broad semantic category (e.g. "Drug", "Gene", "Disease", "Protein").
    canonical_id : str
        Ontology-linked canonical identifier (e.g. UMLS CUI "C1234567").
        Used in SHA-256 fact-id computation.
    source_ontology : str
        Origin ontology for the canonical_id (e.g. "UMLS", "MeSH", "ChEBI").
    """

    id: str = Field(..., min_length=1, description="Internal unique identifier")
    label: str = Field(..., min_length=1, description="Human-readable name")
    entity_type: str = Field(
        ..., min_length=1, description="Semantic type (Drug, Gene, Disease, ...)"
    )
    canonical_id: str = Field(
        ..., min_length=1, description="Ontology CUI (e.g. C1234567)"
    )
    source_ontology: str = Field(
        default="UMLS", description="Source ontology (UMLS, MeSH, ChEBI, ...)"
    )

    def __hash__(self) -> int:
        return hash(self.canonical_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BiomedicalEntity):
            return NotImplemented
        return self.canonical_id == other.canonical_id


# ===========================================================================
# Relation wrapper
# ===========================================================================


class Relation(BaseModel):
    """A typed biomedical relation constrained to the fixed ontology.

    Attributes
    ----------
    type : RelationType
        One of the 30 canonical biomedical relation types.
    """

    type: RelationType = Field(..., description="Biomedical relation type")

    @field_validator("type", mode="before")
    @classmethod
    def coerce_string_to_enum(cls, v: Any) -> RelationType:
        """Accept plain strings and convert to RelationType."""
        if isinstance(v, str):
            v_upper = v.upper()
            try:
                return RelationType(v_upper)
            except ValueError:
                raise ValueError(
                    f"Unknown relation type '{v}'. "
                    f"Must be one of: {[r.value for r in RelationType]}"
                )
        return v


# ===========================================================================
# Temporal envelope
# ===========================================================================


class TemporalEnvelope(BaseModel):
    """Validity window for a fact.

    Implements the append-only temporal model:
    - ``valid_from``: date when the fact became valid.
    - ``valid_to``: date when the fact was superseded.
      ``None`` means the fact is current.
    """

    valid_from: date = Field(..., description="Start of validity (inclusive)")
    valid_to: Optional[date] = Field(
        default=None, description="End of validity (exclusive). None = current."
    )

    @model_validator(mode="after")
    def validate_window(self) -> "TemporalEnvelope":
        """Ensure valid_from < valid_to when both are set."""
        if self.valid_to is not None and self.valid_from >= self.valid_to:
            raise ValueError(
                f"valid_from ({self.valid_from}) must be before "
                f"valid_to ({self.valid_to})"
            )
        return self

    @property
    def is_current(self) -> bool:
        """Return True if the fact is currently valid (not superseded)."""
        return self.valid_to is None

    def contains(self, timestamp: date) -> bool:
        """Return True if the timestamp falls within the validity window."""
        if timestamp < self.valid_from:
            return False
        if self.valid_to is not None and timestamp >= self.valid_to:
            return False
        return True


# ===========================================================================
# Fact
# ===========================================================================


# Default config values — used when computing fact_id.
_DEFAULT_HASH_ALGORITHM = "sha256"
_DEFAULT_HASH_SEPARATOR = "||"


def compute_fact_id(
    subject_canonical_id: str,
    relation_type: str,
    object_canonical_id: str,
    valid_from: date,
    *,
    algorithm: str = _DEFAULT_HASH_ALGORITHM,
    separator: str = _DEFAULT_HASH_SEPARATOR,
) -> str:
    """Compute the content-addressable SHA-256 fact identifier.

    ``SHA-256(subject_canonical_id || relation_type || object_canonical_id || valid_from)``

    Parameters
    ----------
    subject_canonical_id : str
        Canonical ontology id of the subject entity.
    relation_type : str
        String relation type (e.g. "TREATS").
    object_canonical_id : str
        Canonical ontology id of the object entity.
    valid_from : date
        Start of validity for temporal versioning.
    algorithm : str
        Hash algorithm (default ``sha256``).
    separator : str
        Field separator (default ``||``).

    Returns
    -------
    str
        64-character hex digest.
    """
    payload = separator.join([
        subject_canonical_id,
        relation_type,
        object_canonical_id,
        valid_from.isoformat(),
    ])
    return hashlib.new(algorithm, payload.encode("utf-8")).hexdigest()


class Fact(BaseModel):
    """A temporal triple in the knowledge graph.

    The ``fact_id`` is auto-computed from
    ``(subject.canonical_id, relation.type, object.canonical_id, temporal.valid_from)``
    using SHA-256.

    Attributes
    ----------
    subject : BiomedicalEntity
        Subject entity of the triple.
    relation : Relation
        Typed biomedical relation.
    object : BiomedicalEntity
        Object entity of the triple.
    temporal : TemporalEnvelope
        Validity window.
    source : str
        Provenance string (e.g. "PMID:12345678", "PMC:OA-001").
    confidence : float
        Extraction confidence score in [0, 1].
    metadata : dict
        Arbitrary extra metadata (evidence spans, extraction model, etc.).
    fact_id : str
        Auto-computed SHA-256 hash. Read-only.
    version_chain : List[str]
        Ordered list of fact_ids in this fact's version lineage
        (oldest first). May be empty for newly created facts.
    """

    subject: BiomedicalEntity
    relation: Relation
    object: BiomedicalEntity
    temporal: TemporalEnvelope
    source: str = Field(default="", description="Provenance (e.g. PMID)")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence [0, 1]"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    fact_id: str = Field(default="", description="Auto-computed SHA-256 hash")
    version_chain: List[str] = Field(
        default_factory=list,
        description="Ordered fact_ids in the version lineage (oldest first)",
    )

    @model_validator(mode="after")
    def _compute_fact_id(self) -> "Fact":
        """Auto-compute fact_id from the triple + temporal key."""
        computed = compute_fact_id(
            subject_canonical_id=self.subject.canonical_id,
            relation_type=self.relation.type.value,
            object_canonical_id=self.object.canonical_id,
            valid_from=self.temporal.valid_from,
        )
        self.fact_id = computed
        return self

    @property
    def is_current(self) -> bool:
        """Delegate to the temporal envelope."""
        return self.temporal.is_current

    @property
    def family_key(self) -> str:
        """A key that identifies all temporal versions of the same logical fact.

        Two facts share a family_key iff they differ only in valid_from/valid_to.
        ``family_key = SHA-256(subject_canonical_id || relation || object_canonical_id || "FAMILY")``
        """
        separator = _DEFAULT_HASH_SEPARATOR
        payload = separator.join([
            self.subject.canonical_id,
            self.relation.type.value,
            self.object.canonical_id,
            "FAMILY",
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Serialize to a flat dict suitable for Neo4j node properties."""
        return {
            "fact_id": self.fact_id,
            "subject_id": self.subject.canonical_id,
            "subject_label": self.subject.label,
            "relation_type": self.relation.type.value,
            "object_id": self.object.canonical_id,
            "object_label": self.object.label,
            "valid_from": self.temporal.valid_from.isoformat(),
            "valid_to": (
                self.temporal.valid_to.isoformat()
                if self.temporal.valid_to
                else None
            ),
            "source": self.source,
            "confidence": self.confidence,
            "family_key": self.family_key,
        }

    def __hash__(self) -> int:
        return hash(self.fact_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fact):
            return NotImplemented
        return self.fact_id == other.fact_id


# ===========================================================================
# Fact cluster (hierarchical indexing)
# ===========================================================================


class ClusterType(str, Enum):
    """Type of fact grouping for hierarchical FAISS indexing."""

    RELATION = "relation"
    ENTITY = "entity"
    CUSTOM = "custom"


class FactCluster(BaseModel):
    """A group of related facts for hierarchical index levels.

    Attributes
    ----------
    facts : List[Fact]
        The facts in this cluster.
    cluster_type : ClusterType
        How the cluster was formed (relation-based, entity-based, or custom).
    anchor_entity : str
        Canonical id of the entity that anchors this cluster.
    cluster_id : str
        Optional identifier for this cluster.
    """

    facts: List[Fact] = Field(default_factory=list)
    cluster_type: ClusterType = Field(
        default=ClusterType.ENTITY, description="Cluster formation strategy"
    )
    anchor_entity: str = Field(
        default="", description="Canonical id of the anchor entity"
    )
    cluster_id: str = Field(default="", description="Optional cluster identifier")

    @field_validator("cluster_type", mode="before")
    @classmethod
    def coerce_cluster_type(cls, v: Any) -> ClusterType:
        """Accept plain strings and convert to ClusterType."""
        if isinstance(v, str):
            v_lower = v.lower()
            try:
                return ClusterType(v_lower)
            except ValueError:
                raise ValueError(
                    f"Unknown cluster type '{v}'. "
                    f"Must be one of: {[c.value for c in ClusterType]}"
                )
        return v

    @property
    def size(self) -> int:
        """Number of facts in the cluster."""
        return len(self.facts)

    @property
    def entity_ids(self) -> List[str]:
        """Unique canonical entity ids referenced in this cluster."""
        ids: set[str] = set()
        for fact in self.facts:
            ids.add(fact.subject.canonical_id)
            ids.add(fact.object.canonical_id)
        return sorted(ids)

    @property
    def relation_types(self) -> List[str]:
        """Unique relation types in this cluster."""
        return sorted({f.relation.type.value for f in self.facts})
