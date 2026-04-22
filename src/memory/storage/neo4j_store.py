"""Neo4j graph store with local fallback graph index."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set

from src.memory.base import MemoryConfig


class Neo4jStore:
    """Thin Neo4j wrapper used by semantic memory."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig.from_env()
        self.driver = None
        self._resolved_database: Optional[str] = None
        self._initialized = False
        self._local_nodes: Dict[str, Dict[str, Any]] = {}
        self._local_relations: List[Dict[str, Any]] = []

    def _ensure_initialized(self):
        if self._initialized:
            return

        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                self.config.graph_store_url,
                auth=(self.config.graph_username, self.config.graph_password),
                connection_timeout=self.config.graph_connection_timeout,
                max_connection_lifetime=self.config.graph_max_connection_lifetime,
                max_connection_pool_size=self.config.graph_max_connection_pool_size,
            )
            preferred_database = (self.config.graph_database or "").strip()
            if preferred_database:
                try:
                    with self.driver.session(database=preferred_database) as session:
                        session.run("RETURN 1 AS ok").single()
                    self._resolved_database = preferred_database
                except Exception:
                    with self.driver.session() as session:
                        session.run("RETURN 1 AS ok").single()
                    self._resolved_database = None
            else:
                with self.driver.session() as session:
                    session.run("RETURN 1 AS ok").single()
        except ImportError:
            self.driver = None
        except Exception as exc:
            print(f"warning: neo4j unavailable, fallback to local graph: {exc}")
            self.driver = None
        finally:
            self._initialized = True

    @contextmanager
    def _session(self):
        self._ensure_initialized()
        if self.driver is None:
            raise RuntimeError("neo4j driver unavailable")
        session_kwargs = {}
        if self._resolved_database:
            session_kwargs["database"] = self._resolved_database
        with self.driver.session(**session_kwargs) as session:
            yield session

    def create_node(self, node_id: str, node_type: str, properties: Dict[str, Any]) -> bool:
        self._ensure_initialized()
        self._local_nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "properties": dict(properties or {}),
        }

        if self.driver is None:
            return True

        try:
            with self._session() as session:
                session.run(
                    """
                    MERGE (n:Node {id: $node_id})
                    SET n.type = $node_type,
                        n += $properties,
                        n.created_at = timestamp()
                    """,
                    node_id=node_id,
                    node_type=node_type,
                    properties=properties or {},
                )
            return True
        except Exception as exc:
            print(f"create node failed: {exc}")
            return False

    def create_relation(
        self,
        from_id: str,
        to_id: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self._ensure_initialized()
        relation_record = {
            "from_id": from_id,
            "to_id": to_id,
            "type": relation_type,
            "properties": dict(properties or {}),
        }
        if relation_record not in self._local_relations:
            self._local_relations.append(relation_record)

        if self.driver is None:
            return True

        try:
            with self._session() as session:
                session.run(
                    """
                    MATCH (a:Node {id: $from_id})
                    MATCH (b:Node {id: $to_id})
                    MERGE (a)-[r:RELATION {type: $relation_type}]->(b)
                    SET r += $properties,
                        r.created_at = timestamp()
                    """,
                    from_id=from_id,
                    to_id=to_id,
                    relation_type=relation_type,
                    properties=properties or {},
                )
            return True
        except Exception as exc:
            print(f"create relation failed: {exc}")
            return False

    def query(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        if self.driver is None:
            return []

        try:
            with self._session() as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as exc:
            print(f"query failed: {exc}")
            return []

    def get_related_nodes(self, node_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        if self.driver is None:
            return []

        try:
            with self._session() as session:
                result = session.run(
                    """
                    MATCH path = (start:Node {id: $node_id})-[*1..$depth]-(related:Node)
                    RETURN related, length(path) AS distance
                    ORDER BY distance
                    LIMIT 50
                    """,
                    node_id=node_id,
                    depth=depth,
                )
                return [record.data() for record in result]
        except Exception as exc:
            print(f"related node query failed: {exc}")
            return []

    def delete_nodes_by_source(self, source_memory: str) -> bool:
        self._ensure_initialized()

        local_ids_to_delete = {
            node_id
            for node_id, node in self._local_nodes.items()
            if node.get("properties", {}).get("source_memory") == source_memory
        }
        if local_ids_to_delete:
            self._local_relations = [
                relation
                for relation in self._local_relations
                if relation.get("from_id") not in local_ids_to_delete
                and relation.get("to_id") not in local_ids_to_delete
            ]
            for node_id in local_ids_to_delete:
                self._local_nodes.pop(node_id, None)

        if self.driver is None:
            return True

        try:
            with self._session() as session:
                session.run(
                    """
                    MATCH (n:Node {source_memory: $source_memory})
                    DETACH DELETE n
                    """,
                    source_memory=source_memory,
                )
            return True
        except Exception as exc:
            print(f"delete nodes failed: {exc}")
            return False

    def clear(self) -> bool:
        self._ensure_initialized()
        self._local_nodes.clear()
        self._local_relations.clear()
        if self.driver is None:
            return True

        try:
            with self._session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            return True
        except Exception as exc:
            print(f"clear graph failed: {exc}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        self._ensure_initialized()
        if self.driver is None:
            return {
                "status": "simulated",
                "node_count": len(self._local_nodes),
                "relation_count": len(self._local_relations),
            }

        try:
            with self._session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
                relation_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            return {"status": "ok", "node_count": node_count, "relation_count": relation_count}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def is_available(self) -> bool:
        self._ensure_initialized()
        return self.driver is not None

    def search_by_text(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []

        query_text = (query or "").strip().lower()
        if not query_text:
            return []

        results = []
        for node in self._local_nodes.values():
            properties = node.get("properties", {}) or {}
            text_parts = [
                str(node.get("id", "")),
                str(node.get("type", "")),
                str(properties.get("name", "")),
                str(properties.get("source_memory", "")),
            ]
            text_parts.extend(str(value) for value in properties.values() if value is not None)
            haystack = " ".join(text_parts).lower()
            if query_text in haystack:
                results.append(
                    {
                        "id": node.get("id", ""),
                        "content": properties.get("name", ""),
                        "properties": properties,
                    }
                )
        return results[:limit]

    def score_memories_by_entities(self, query_entities: List[str], limit: int = 20) -> Dict[str, float]:
        if limit <= 0:
            return {}

        normalized_entities = {entity.strip().lower() for entity in query_entities if entity.strip()}
        if not normalized_entities:
            return {}

        entity_to_nodes: Dict[str, Set[str]] = {}
        for node_id, node in self._local_nodes.items():
            properties = node.get("properties", {}) or {}
            entity_name = str(properties.get("name", "")).strip().lower()
            if entity_name:
                entity_to_nodes.setdefault(entity_name, set()).add(node_id)

        matched_node_ids: Set[str] = set()
        for entity in normalized_entities:
            matched_node_ids.update(entity_to_nodes.get(entity, set()))

        if not matched_node_ids:
            return {}

        expanded_node_ids = set(matched_node_ids)
        for relation in self._local_relations:
            from_id = relation.get("from_id")
            to_id = relation.get("to_id")
            if from_id in matched_node_ids:
                expanded_node_ids.add(to_id)
            if to_id in matched_node_ids:
                expanded_node_ids.add(from_id)

        memory_to_nodes: Dict[str, Set[str]] = {}
        memory_to_match_count: Dict[str, int] = {}
        for node_id in expanded_node_ids:
            node = self._local_nodes.get(node_id, {})
            source_memory = str(node.get("properties", {}).get("source_memory", "")).strip()
            if not source_memory:
                continue
            memory_to_nodes.setdefault(source_memory, set()).add(node_id)
            if node_id in matched_node_ids:
                memory_to_match_count[source_memory] = memory_to_match_count.get(source_memory, 0) + 1

        scores: Dict[str, float] = {}
        query_size = len(normalized_entities)
        for memory_id, node_ids in memory_to_nodes.items():
            direct_overlap = memory_to_match_count.get(memory_id, 0) / query_size
            relation_bonus = min(
                max((len(node_ids) - memory_to_match_count.get(memory_id, 0)) * 0.15, 0.0),
                0.4,
            )
            scores[memory_id] = min(direct_overlap + relation_bonus, 1.0)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return dict(ranked[:limit])
