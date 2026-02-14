"""
Neo4j Client — Full CRUD client for the FRLM temporal knowledge graph.

Provides connection management with retry logic, entity/fact CRUD,
temporal queries, version chain management, bulk import, and index creation.

All Cypher queries are defined as module-level constants for transparency
and testability.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

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

logger = logging.getLogger(__name__)


# ===========================================================================
# Cypher query constants
# ===========================================================================

# --- Schema / index creation ------------------------------------------------

CYPHER_CREATE_ENTITY_CONSTRAINT = (
    "CREATE CONSTRAINT IF NOT EXISTS "
    "FOR (e:{entity_label}) REQUIRE e.canonical_id IS UNIQUE"
)

CYPHER_CREATE_FACT_CONSTRAINT = (
    "CREATE CONSTRAINT IF NOT EXISTS "
    "FOR (f:{fact_label}) REQUIRE f.fact_id IS UNIQUE"
)

CYPHER_CREATE_FACT_INDEX_SUBJECT = (
    "CREATE INDEX IF NOT EXISTS FOR (f:{fact_label}) ON (f.subject_id)"
)

CYPHER_CREATE_FACT_INDEX_OBJECT = (
    "CREATE INDEX IF NOT EXISTS FOR (f:{fact_label}) ON (f.object_id)"
)

CYPHER_CREATE_FACT_INDEX_VALID_FROM = (
    "CREATE INDEX IF NOT EXISTS FOR (f:{fact_label}) ON (f.valid_from)"
)

CYPHER_CREATE_FACT_INDEX_VALID_TO = (
    "CREATE INDEX IF NOT EXISTS FOR (f:{fact_label}) ON (f.valid_to)"
)

CYPHER_CREATE_FACT_INDEX_FAMILY_KEY = (
    "CREATE INDEX IF NOT EXISTS FOR (f:{fact_label}) ON (f.family_key)"
)

CYPHER_CREATE_ENTITY_INDEX_CANONICAL = (
    "CREATE INDEX IF NOT EXISTS FOR (e:{entity_label}) ON (e.canonical_id)"
)

# --- Entity CRUD ------------------------------------------------------------

CYPHER_MERGE_ENTITY = (
    "MERGE (e:{entity_label} {{canonical_id: $canonical_id}}) "
    "ON CREATE SET e.id = $id, e.label = $label, "
    "e.entity_type = $entity_type, e.source_ontology = $source_ontology "
    "ON MATCH SET e.label = CASE WHEN $label <> '' THEN $label ELSE e.label END, "
    "e.entity_type = CASE WHEN $entity_type <> '' THEN $entity_type ELSE e.entity_type END "
    "RETURN e"
)

CYPHER_GET_ENTITY = (
    "MATCH (e:{entity_label} {{canonical_id: $canonical_id}}) "
    "RETURN e"
)

# --- Fact CRUD --------------------------------------------------------------

CYPHER_CREATE_FACT = (
    "MERGE (f:{fact_label} {{fact_id: $fact_id}}) "
    "ON CREATE SET f.subject_id = $subject_id, f.subject_label = $subject_label, "
    "f.relation_type = $relation_type, f.object_id = $object_id, "
    "f.object_label = $object_label, f.valid_from = $valid_from, "
    "f.valid_to = $valid_to, f.source = $source, f.confidence = $confidence, "
    "f.family_key = $family_key "
    "RETURN f"
)

CYPHER_LINK_FACT_TO_SUBJECT = (
    "MATCH (e:{entity_label} {{canonical_id: $subject_id}}), "
    "(f:{fact_label} {{fact_id: $fact_id}}) "
    "MERGE (e)-[:{relation_type}]->(f) "
    "RETURN e, f"
)

CYPHER_LINK_FACT_TO_OBJECT = (
    "MATCH (f:{fact_label} {{fact_id: $fact_id}}), "
    "(e:{entity_label} {{canonical_id: $object_id}}) "
    "MERGE (f)-[:{relation_type}]->(e) "
    "RETURN f, e"
)

CYPHER_GET_FACT_BY_ID = (
    "MATCH (f:{fact_label} {{fact_id: $fact_id}}) "
    "RETURN f"
)

# --- Version chain ----------------------------------------------------------

CYPHER_CREATE_VERSION_CHAIN = (
    "MATCH (old:{fact_label} {{fact_id: $old_fact_id}}), "
    "(new:{fact_label} {{fact_id: $new_fact_id}}) "
    "MERGE (new)-[:{version_chain_type}]->(old) "
    "RETURN old, new"
)

CYPHER_SET_VALID_TO = (
    "MATCH (f:{fact_label} {{fact_id: $fact_id}}) "
    "SET f.valid_to = $valid_to "
    "RETURN f"
)

# --- Queries ----------------------------------------------------------------

CYPHER_FACTS_FOR_ENTITY = (
    "MATCH (f:{fact_label}) "
    "WHERE f.subject_id = $entity_id OR f.object_id = $entity_id "
    "RETURN f ORDER BY f.valid_from DESC"
)

CYPHER_FACTS_BETWEEN_ENTITIES = (
    "MATCH (f:{fact_label}) "
    "WHERE f.subject_id = $entity1_id AND f.object_id = $entity2_id "
    "RETURN f ORDER BY f.valid_from DESC"
)

CYPHER_FACTS_BETWEEN_ENTITIES_EITHER = (
    "MATCH (f:{fact_label}) "
    "WHERE (f.subject_id = $entity1_id AND f.object_id = $entity2_id) "
    "OR (f.subject_id = $entity2_id AND f.object_id = $entity1_id) "
    "RETURN f ORDER BY f.valid_from DESC"
)

CYPHER_ENTITY_SUBGRAPH = (
    "MATCH (f:{fact_label}) "
    "WHERE f.subject_id = $entity_id OR f.object_id = $entity_id "
    "RETURN f ORDER BY f.valid_from DESC"
)

CYPHER_FACT_HISTORY = (
    "MATCH (f:{fact_label}) "
    "WHERE f.family_key = $family_key "
    "RETURN f ORDER BY f.valid_from ASC"
)

# --- Bulk import ------------------------------------------------------------

CYPHER_BULK_MERGE_ENTITIES = (
    "UNWIND $entities AS e "
    "MERGE (n:{entity_label} {{canonical_id: e.canonical_id}}) "
    "ON CREATE SET n.id = e.id, n.label = e.label, "
    "n.entity_type = e.entity_type, n.source_ontology = e.source_ontology "
    "RETURN count(n) AS imported"
)

CYPHER_BULK_MERGE_FACTS = (
    "UNWIND $facts AS f "
    "MERGE (n:{fact_label} {{fact_id: f.fact_id}}) "
    "ON CREATE SET n.subject_id = f.subject_id, n.subject_label = f.subject_label, "
    "n.relation_type = f.relation_type, n.object_id = f.object_id, "
    "n.object_label = f.object_label, n.valid_from = f.valid_from, "
    "n.valid_to = f.valid_to, n.source = f.source, n.confidence = f.confidence, "
    "n.family_key = f.family_key "
    "RETURN count(n) AS imported"
)


# ===========================================================================
# Neo4j Client
# ===========================================================================


class Neo4jClient:
    """Full CRUD client for the FRLM temporal knowledge graph.

    Parameters
    ----------
    uri : str
        Neo4j bolt URI.
    username : str
        Neo4j username.
    password : str
        Neo4j password.
    database : str
        Neo4j database name.
    max_connection_pool_size : int
        Max connections in the driver pool.
    connection_timeout : int
        Connection timeout in seconds.
    max_transaction_retry_time : int
        Max retry time for transient errors.
    encrypted : bool
        Whether to use TLS.
    trust : str
        TLS trust setting.
    entity_label : str
        Neo4j node label for entities.
    fact_label : str
        Neo4j node label for facts.
    relation_type : str
        Neo4j relationship type linking entities to facts.
    version_chain_type : str
        Neo4j relationship type for version chains (SUPERSEDES).
    batch_size : int
        Default batch size for bulk operations.
    max_retries : int
        Max retries for transient failures.
    retry_delay : float
        Base delay between retries (seconds).
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "CHANGE_ME",
        database: str = "frlm",
        max_connection_pool_size: int = 50,
        connection_timeout: int = 30,
        max_transaction_retry_time: int = 30,
        encrypted: bool = False,
        trust: str = "TRUST_ALL_CERTIFICATES",
        entity_label: str = "Entity",
        fact_label: str = "Fact",
        relation_type: str = "HAS_FACT",
        version_chain_type: str = "SUPERSEDES",
        batch_size: int = 5000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._max_connection_pool_size = max_connection_pool_size
        self._connection_timeout = connection_timeout
        self._max_transaction_retry_time = max_transaction_retry_time
        self._encrypted = encrypted
        self._trust = trust

        # Schema labels
        self._entity_label = entity_label
        self._fact_label = fact_label
        self._relation_type = relation_type
        self._version_chain_type = version_chain_type

        # Batch settings
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Lazy driver
        self._driver: Any = None

    # --- Class methods for config-based construction -------------------------

    @classmethod
    def from_config(cls, cfg: Any) -> "Neo4jClient":
        """Create a client from an FRLMConfig object.

        Parameters
        ----------
        cfg : FRLMConfig
            The full FRLM configuration.

        Returns
        -------
        Neo4jClient
        """
        neo4j_cfg = cfg.neo4j
        schema = neo4j_cfg.graph_schema
        batch_cfg = neo4j_cfg.batch

        return cls(
            uri=neo4j_cfg.uri,
            username=neo4j_cfg.username,
            password=neo4j_cfg.password,
            database=neo4j_cfg.database,
            max_connection_pool_size=neo4j_cfg.max_connection_pool_size,
            connection_timeout=neo4j_cfg.connection_timeout,
            max_transaction_retry_time=neo4j_cfg.max_transaction_retry_time,
            encrypted=neo4j_cfg.encrypted,
            trust=neo4j_cfg.trust,
            entity_label=schema.entity_label,
            fact_label=schema.fact_label,
            relation_type=schema.relation_type,
            version_chain_type=schema.version_chain_type,
            batch_size=batch_cfg.import_batch_size,
            max_retries=batch_cfg.max_retries,
            retry_delay=batch_cfg.retry_delay,
        )

    # --- Connection management -----------------------------------------------

    def connect(self) -> None:
        """Establish connection to Neo4j.

        Raises
        ------
        ImportError
            If the ``neo4j`` package is not installed.
        RuntimeError
            If the connection cannot be established.
        """
        try:
            from neo4j import GraphDatabase  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "neo4j Python driver is required. Install with: pip install neo4j"
            )

        logger.info("Connecting to Neo4j at %s (database=%s)", self._uri, self._database)

        try:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._username, self._password),
                max_connection_pool_size=self._max_connection_pool_size,
                connection_timeout=self._connection_timeout,
                max_transaction_retry_time=self._max_transaction_retry_time,
                encrypted=self._encrypted,
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j")
        except Exception as exc:
            logger.error("Failed to connect to Neo4j: %s", exc)
            raise RuntimeError(f"Neo4j connection failed: {exc}") from exc

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @property
    def driver(self) -> Any:
        """Return the active driver, raising if not connected."""
        if self._driver is None:
            raise RuntimeError(
                "Neo4j driver is not connected. Call connect() first."
            )
        return self._driver

    # --- Internal helpers ----------------------------------------------------

    def _format_cypher(self, template: str) -> str:
        """Substitute schema labels into a Cypher template."""
        return template.format(
            entity_label=self._entity_label,
            fact_label=self._fact_label,
            relation_type=self._relation_type,
            version_chain_type=self._version_chain_type,
        )

    def _execute_with_retry(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        write: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query with retry logic for transient failures.

        Parameters
        ----------
        cypher : str
            The Cypher query to execute.
        parameters : dict, optional
            Query parameters.
        write : bool
            If True, use a write transaction; otherwise read.

        Returns
        -------
        list of dict
            Each dict is a record from the result.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                with self.driver.session(database=self._database) as session:
                    if write:
                        result = session.execute_write(
                            lambda tx: list(tx.run(cypher, parameters or {}))
                        )
                    else:
                        result = session.execute_read(
                            lambda tx: list(tx.run(cypher, parameters or {}))
                        )
                    return [dict(record) for record in result]
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._retry_delay * attempt
                    logger.warning(
                        "Neo4j query failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt, self._max_retries, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Neo4j query failed after %d attempts: %s",
                        self._max_retries, exc,
                    )

        raise RuntimeError(
            f"Neo4j query failed after {self._max_retries} retries"
        ) from last_exc

    def _neo4j_node_to_entity(self, node: Any) -> BiomedicalEntity:
        """Convert a Neo4j node dict to a BiomedicalEntity."""
        props = dict(node) if not isinstance(node, dict) else node
        return BiomedicalEntity(
            id=props.get("id", props.get("canonical_id", "")),
            label=props.get("label", ""),
            entity_type=props.get("entity_type", ""),
            canonical_id=props.get("canonical_id", ""),
            source_ontology=props.get("source_ontology", "UMLS"),
        )

    def _neo4j_node_to_fact(self, node: Any) -> Fact:
        """Convert a Neo4j node dict to a Fact.

        The subject and object entities are reconstructed from the stored
        ids and labels. Full entity details can be fetched separately if needed.
        """
        props = dict(node) if not isinstance(node, dict) else node

        subject = BiomedicalEntity(
            id=props.get("subject_id", ""),
            label=props.get("subject_label", ""),
            entity_type="",
            canonical_id=props.get("subject_id", ""),
        )
        obj = BiomedicalEntity(
            id=props.get("object_id", ""),
            label=props.get("object_label", ""),
            entity_type="",
            canonical_id=props.get("object_id", ""),
        )

        valid_to_raw = props.get("valid_to")
        valid_to = (
            date.fromisoformat(valid_to_raw)
            if valid_to_raw is not None
            else None
        )

        return Fact(
            subject=subject,
            relation=Relation(type=props.get("relation_type", "ASSOCIATED_WITH")),
            object=obj,
            temporal=TemporalEnvelope(
                valid_from=date.fromisoformat(props.get("valid_from", "2000-01-01")),
                valid_to=valid_to,
            ),
            source=props.get("source", ""),
            confidence=float(props.get("confidence", 1.0)),
        )

    # --- Schema / index creation ---------------------------------------------

    def create_indexes(self) -> None:
        """Create all uniqueness constraints and indexes on the Neo4j schema."""
        templates = [
            CYPHER_CREATE_ENTITY_CONSTRAINT,
            CYPHER_CREATE_FACT_CONSTRAINT,
            CYPHER_CREATE_FACT_INDEX_SUBJECT,
            CYPHER_CREATE_FACT_INDEX_OBJECT,
            CYPHER_CREATE_FACT_INDEX_VALID_FROM,
            CYPHER_CREATE_FACT_INDEX_VALID_TO,
            CYPHER_CREATE_FACT_INDEX_FAMILY_KEY,
            CYPHER_CREATE_ENTITY_INDEX_CANONICAL,
        ]

        logger.info("Creating %d schema constraints and indexes", len(templates))

        for template in templates:
            cypher = self._format_cypher(template)
            logger.debug("Executing: %s", cypher[:120])
            self._execute_with_retry(cypher, write=True)

        logger.info("All schema constraints and indexes created")

    # --- Entity CRUD ---------------------------------------------------------

    def create_entity(self, entity: BiomedicalEntity) -> BiomedicalEntity:
        """Create or merge an entity node in Neo4j.

        Parameters
        ----------
        entity : BiomedicalEntity
            The entity to create or merge.

        Returns
        -------
        BiomedicalEntity
            The entity as stored in Neo4j.
        """
        cypher = self._format_cypher(CYPHER_MERGE_ENTITY)
        params = {
            "id": entity.id,
            "label": entity.label,
            "entity_type": entity.entity_type,
            "canonical_id": entity.canonical_id,
            "source_ontology": entity.source_ontology,
        }

        logger.debug("Creating/merging entity: %s (%s)", entity.label, entity.canonical_id)
        results = self._execute_with_retry(cypher, params, write=True)

        if results:
            return self._neo4j_node_to_entity(results[0]["e"])
        return entity

    def get_entity(self, canonical_id: str) -> Optional[BiomedicalEntity]:
        """Retrieve an entity by canonical id.

        Parameters
        ----------
        canonical_id : str
            The canonical ontology identifier.

        Returns
        -------
        BiomedicalEntity or None
        """
        cypher = self._format_cypher(CYPHER_GET_ENTITY)
        results = self._execute_with_retry(cypher, {"canonical_id": canonical_id})

        if results:
            return self._neo4j_node_to_entity(results[0]["e"])
        return None

    # --- Fact CRUD -----------------------------------------------------------

    def create_fact(self, fact: Fact) -> Fact:
        """Create a fact node and link it to subject/object entities.

        Entities are auto-created (merged) if they don't exist.

        Parameters
        ----------
        fact : Fact
            The fact to create.

        Returns
        -------
        Fact
            The fact as stored.
        """
        # Ensure entities exist
        self.create_entity(fact.subject)
        self.create_entity(fact.object)

        # Create fact node
        props = fact.to_neo4j_properties()
        cypher = self._format_cypher(CYPHER_CREATE_FACT)
        logger.debug("Creating fact: %s", fact.fact_id[:16])
        self._execute_with_retry(cypher, props, write=True)

        # Link to subject entity
        cypher_link_subj = self._format_cypher(CYPHER_LINK_FACT_TO_SUBJECT)
        self._execute_with_retry(
            cypher_link_subj,
            {"subject_id": fact.subject.canonical_id, "fact_id": fact.fact_id},
            write=True,
        )

        # Link to object entity
        cypher_link_obj = self._format_cypher(CYPHER_LINK_FACT_TO_OBJECT)
        self._execute_with_retry(
            cypher_link_obj,
            {"object_id": fact.object.canonical_id, "fact_id": fact.fact_id},
            write=True,
        )

        logger.info(
            "Created fact %s: %s -[%s]-> %s",
            fact.fact_id[:16], fact.subject.label,
            fact.relation.type.value, fact.object.label,
        )
        return fact

    def get_fact_by_id(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a fact by its SHA-256 fact_id.

        Parameters
        ----------
        fact_id : str
            The 64-character hex digest fact identifier.

        Returns
        -------
        Fact or None
        """
        cypher = self._format_cypher(CYPHER_GET_FACT_BY_ID)
        results = self._execute_with_retry(cypher, {"fact_id": fact_id})

        if results:
            return self._neo4j_node_to_fact(results[0]["f"])
        return None

    # --- Version chain management --------------------------------------------

    def create_fact_version(
        self,
        old_fact_id: str,
        new_fact: Fact,
    ) -> Fact:
        """Supersede an existing fact with a new version.

        1. Sets ``valid_to`` on the old fact to the new fact's ``valid_from``.
        2. Creates the new fact.
        3. Creates a SUPERSEDES edge from new to old.

        Parameters
        ----------
        old_fact_id : str
            The fact_id of the fact being superseded.
        new_fact : Fact
            The new version of the fact.

        Returns
        -------
        Fact
            The newly created fact.

        Raises
        ------
        ValueError
            If the old fact is not found.
        """
        # Verify old fact exists
        old_fact = self.get_fact_by_id(old_fact_id)
        if old_fact is None:
            raise ValueError(f"Old fact not found: {old_fact_id}")

        # Close old fact's temporal window
        cypher_close = self._format_cypher(CYPHER_SET_VALID_TO)
        self._execute_with_retry(
            cypher_close,
            {
                "fact_id": old_fact_id,
                "valid_to": new_fact.temporal.valid_from.isoformat(),
            },
            write=True,
        )
        logger.info(
            "Closed old fact %s valid_to=%s",
            old_fact_id[:16], new_fact.temporal.valid_from,
        )

        # Create new fact
        created = self.create_fact(new_fact)

        # Build version chain edge
        cypher_chain = self._format_cypher(CYPHER_CREATE_VERSION_CHAIN)
        self._execute_with_retry(
            cypher_chain,
            {"old_fact_id": old_fact_id, "new_fact_id": created.fact_id},
            write=True,
        )
        logger.info(
            "Version chain: %s -[SUPERSEDES]-> %s",
            created.fact_id[:16], old_fact_id[:16],
        )

        return created

    # --- Query operations ----------------------------------------------------

    def get_facts_for_entity(
        self,
        entity_id: str,
        temporal_mode: str = "CURRENT",
        timestamp: Optional[date] = None,
    ) -> List[Fact]:
        """Retrieve all facts involving an entity, filtered by temporal mode.

        Parameters
        ----------
        entity_id : str
            Canonical id of the entity.
        temporal_mode : str
            One of "CURRENT", "AT_TIMESTAMP", "HISTORY".
        timestamp : date, optional
            Required when temporal_mode is "AT_TIMESTAMP".

        Returns
        -------
        list of Fact
        """
        cypher = self._format_cypher(CYPHER_FACTS_FOR_ENTITY)
        results = self._execute_with_retry(cypher, {"entity_id": entity_id})

        facts = [self._neo4j_node_to_fact(r["f"]) for r in results]
        return self._apply_temporal_filter(facts, temporal_mode, timestamp)

    def get_facts_between_entities(
        self,
        entity1_id: str,
        entity2_id: str,
        temporal_mode: str = "CURRENT",
        timestamp: Optional[date] = None,
    ) -> List[Fact]:
        """Retrieve all facts between two entities.

        Parameters
        ----------
        entity1_id : str
            Canonical id of the first entity.
        entity2_id : str
            Canonical id of the second entity.
        temporal_mode : str
            One of "CURRENT", "AT_TIMESTAMP", "HISTORY".
        timestamp : date, optional
            Required when temporal_mode is "AT_TIMESTAMP".

        Returns
        -------
        list of Fact
        """
        cypher = self._format_cypher(CYPHER_FACTS_BETWEEN_ENTITIES_EITHER)
        results = self._execute_with_retry(
            cypher, {"entity1_id": entity1_id, "entity2_id": entity2_id}
        )

        facts = [self._neo4j_node_to_fact(r["f"]) for r in results]
        return self._apply_temporal_filter(facts, temporal_mode, timestamp)

    def get_entity_subgraph(
        self,
        entity_id: str,
        depth: int = 1,
        temporal_mode: str = "CURRENT",
        timestamp: Optional[date] = None,
    ) -> FactCluster:
        """Retrieve a subgraph around an entity up to a given depth.

        For depth=1, returns all facts directly involving the entity.
        For depth>1, recursively expands to neighbouring entities.

        Parameters
        ----------
        entity_id : str
            Canonical id of the anchor entity.
        depth : int
            Expansion depth (default 1).
        temporal_mode : str
            Temporal filter mode.
        timestamp : date, optional
            Required for AT_TIMESTAMP mode.

        Returns
        -------
        FactCluster
        """
        visited_entities: set[str] = set()
        all_facts: List[Fact] = []
        frontier = {entity_id}

        for d in range(depth):
            next_frontier: set[str] = set()
            for eid in frontier:
                if eid in visited_entities:
                    continue
                visited_entities.add(eid)

                facts = self.get_facts_for_entity(eid, temporal_mode, timestamp)
                for f in facts:
                    if f.fact_id not in {af.fact_id for af in all_facts}:
                        all_facts.append(f)
                    next_frontier.add(f.subject.canonical_id)
                    next_frontier.add(f.object.canonical_id)

            frontier = next_frontier - visited_entities

        return FactCluster(
            facts=all_facts,
            cluster_type=ClusterType.ENTITY,
            anchor_entity=entity_id,
        )

    def get_fact_history(self, family_key: str) -> List[Fact]:
        """Retrieve all temporal versions of a fact, ordered by valid_from.

        Parameters
        ----------
        family_key : str
            The family key shared by all versions of a logical fact.

        Returns
        -------
        list of Fact
            Ordered oldest-first by valid_from.
        """
        cypher = self._format_cypher(CYPHER_FACT_HISTORY)
        results = self._execute_with_retry(cypher, {"family_key": family_key})

        return [self._neo4j_node_to_fact(r["f"]) for r in results]

    # --- Bulk operations -----------------------------------------------------

    def bulk_import_facts(self, facts: List[Fact]) -> int:
        """Batch-import facts with transaction management.

        Entities are extracted and imported first, then facts are imported
        in batches.

        Parameters
        ----------
        facts : list of Fact
            Facts to import.

        Returns
        -------
        int
            Total number of facts imported.
        """
        if not facts:
            logger.warning("bulk_import_facts called with empty list")
            return 0

        logger.info("Bulk importing %d facts (batch_size=%d)", len(facts), self._batch_size)

        # Collect unique entities
        entities_map: Dict[str, BiomedicalEntity] = {}
        for fact in facts:
            entities_map[fact.subject.canonical_id] = fact.subject
            entities_map[fact.object.canonical_id] = fact.object

        unique_entities = list(entities_map.values())
        logger.info("Importing %d unique entities", len(unique_entities))

        # Bulk import entities
        entity_dicts = [
            {
                "id": e.id,
                "label": e.label,
                "entity_type": e.entity_type,
                "canonical_id": e.canonical_id,
                "source_ontology": e.source_ontology,
            }
            for e in unique_entities
        ]

        for i in range(0, len(entity_dicts), self._batch_size):
            batch = entity_dicts[i : i + self._batch_size]
            cypher = self._format_cypher(CYPHER_BULK_MERGE_ENTITIES)
            self._execute_with_retry(cypher, {"entities": batch}, write=True)
            logger.debug(
                "Entity batch %d-%d of %d",
                i + 1, min(i + self._batch_size, len(entity_dicts)), len(entity_dicts),
            )

        # Bulk import facts
        fact_dicts = [f.to_neo4j_properties() for f in facts]
        imported = 0

        for i in range(0, len(fact_dicts), self._batch_size):
            batch = fact_dicts[i : i + self._batch_size]
            cypher = self._format_cypher(CYPHER_BULK_MERGE_FACTS)
            self._execute_with_retry(cypher, {"facts": batch}, write=True)
            imported += len(batch)
            logger.debug(
                "Fact batch %d-%d of %d",
                i + 1, min(i + self._batch_size, len(fact_dicts)), len(fact_dicts),
            )

        logger.info("Bulk import complete: %d facts imported", imported)
        return imported

    # --- Temporal filtering helper -------------------------------------------

    @staticmethod
    def _apply_temporal_filter(
        facts: List[Fact],
        temporal_mode: str,
        timestamp: Optional[date] = None,
    ) -> List[Fact]:
        """Apply a temporal filter to a list of facts.

        Parameters
        ----------
        facts : list of Fact
            Unfiltered facts.
        temporal_mode : str
            "CURRENT", "AT_TIMESTAMP", or "HISTORY".
        timestamp : date, optional
            Required for AT_TIMESTAMP.

        Returns
        -------
        list of Fact
        """
        mode = temporal_mode.upper()

        if mode == "CURRENT":
            return [f for f in facts if f.is_current]

        if mode == "AT_TIMESTAMP":
            if timestamp is None:
                raise ValueError("timestamp is required for AT_TIMESTAMP mode")
            return [f for f in facts if f.temporal.contains(timestamp)]

        if mode == "HISTORY":
            return sorted(facts, key=lambda f: f.temporal.valid_from)

        raise ValueError(
            f"Unknown temporal_mode '{temporal_mode}'. "
            f"Must be one of: CURRENT, AT_TIMESTAMP, HISTORY"
        )
