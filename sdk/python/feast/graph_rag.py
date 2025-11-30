"""GraphRAG helpers built on top of Feast + Milvus.

This module provides a thin orchestration layer for the "Graph RAG with Milvus"
pattern described in https://milvus.io/docs/graph_rag_with_milvus.md. It
assumes users already configured Feast FeatureViews backed by Milvus with
vector indices and adjacency arrays, and focuses on stitching together entity,
relation, and passage retrieval.

Recommended FeatureView layouts
-------------------------------
Entity FeatureView::

    from datetime import timedelta
    from feast import Entity, FeatureView, Field
    from feast.types import Array, Float32, String, UnixTimestamp

    entity = Entity(name="entity_id", join_keys=["entity_id"])
    entity_view = FeatureView(
        name="graph_entities",
        entities=[entity],
        schema=[
            Field("entity_id", String),
            Field("embedding", Array(Float32), vector_index=True, vector_search_metric="COSINE"),
            Field("neighbors", Array(String)),  # neighbor entity ids
            Field("relation_ids", Array(String)),  # relations touching this entity
            Field("event_timestamp", UnixTimestamp),
        ],
        ttl=timedelta(days=1),
        source=...,  # offline source not used at retrieval time
    )

Relation FeatureView::

    relation = Entity(name="relation_id", join_keys=["relation_id"])
    relation_view = FeatureView(
        name="graph_relations",
        entities=[relation],
        schema=[
            Field("relation_id", String),
            Field("embedding", Array(Float32), vector_index=True, vector_search_metric="COSINE"),
            Field("connected_entities", Array(String)),
            Field("passage_ids", Array(String)),  # passages that describe this relation
            Field("event_timestamp", UnixTimestamp),
        ],
        ttl=timedelta(days=1),
        source=...,  # offline source not used at retrieval time
    )

Passage FeatureView::

    passage = Entity(name="passage_id", join_keys=["passage_id"])
    passage_view = FeatureView(
        name="graph_passages",
        entities=[passage],
        schema=[
            Field("passage_id", String),
            Field("text", String),
            Field("embedding", Array(Float32), vector_index=True, vector_search_metric="COSINE"),
            Field("relation_ids", Array(String)),
            Field("event_timestamp", UnixTimestamp),
        ],
        ttl=timedelta(days=1),
        source=...,  # offline source not used at retrieval time
    )

Usage::

    store = FeatureStore(repo_path=".")
    retriever = GraphRAGRetriever(store)
    result = retriever.retrieve(
        query_embedding=[...],
        entity_view="graph_entities",
        relation_view="graph_relations",
        passage_view="graph_passages",
        top_k_entities=5,
        top_k_relations=5,
        top_k_passages=5,
    )

The returned :class:`GraphRAGResult` contains the retrieved entities, relations,
passages, and the in-memory edges derived from adjacency arrays.

Notes:
- Subgraph expansion relies on adjacency arrays stored in Milvus; it does not issue
  additional vector searches for expanded nodes beyond the configured hops.
- Passage lookups prefer relation-to-passage mappings and will fall back to
  filtering passage search results by relation IDs when available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Set

from feast import FeatureStore

Reranker = Callable[[List["RelationCandidate"], Optional[str]], List["RelationCandidate"]]


@dataclass
class GraphNode:
    node_id: str
    node_type: Literal["entity", "relation"]
    metadata: Dict[str, Any]


@dataclass
class GraphEdge:
    src_id: str
    dst_id: str
    edge_type: str


@dataclass
class GraphPassage:
    passage_id: str
    text: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class RelationCandidate:
    relation_id: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class GraphRAGResult:
    entities: List[GraphNode]
    relations: List[GraphNode]
    passages: List[GraphPassage]
    edges: List[GraphEdge]


class GraphRAGRetriever:
    """Utility class that orchestrates GraphRAG retrieval using Feast APIs."""

    def __init__(self, store: FeatureStore, project: Optional[str] = None):
        self.store = store
        self.project = project or store.project

    def retrieve(
        self,
        query_embedding: List[float],
        *,
        entity_view: str,
        relation_view: str,
        passage_view: str,
        entity_embedding_field: str = "embedding",
        relation_embedding_field: str = "embedding",
        passage_embedding_field: str = "embedding",
        entity_id_field: str = "entity_id",
        relation_id_field: str = "relation_id",
        passage_id_field: str = "passage_id",
        passage_text_field: Optional[str] = "text",
        entity_neighbors_field: Optional[str] = "neighbors",
        entity_relation_field: Optional[str] = "relation_ids",
        relation_entity_field: Optional[str] = "connected_entities",
        relation_passage_field: Optional[str] = "passage_ids",
        passage_relation_field: Optional[str] = "relation_ids",
        top_k_entities: int = 10,
        top_k_relations: int = 10,
        top_k_passages: int = 10,
        max_hops: int = 2,
        max_expanded_entities: int = 50,
        max_expanded_relations: int = 50,
        reranker: Optional[Reranker] = None,
        query_text: Optional[str] = None,
    ) -> GraphRAGResult:
        """Run entity + relation search, expand the subgraph, then fetch passages."""

        entity_hits = self._search_view(
            view=entity_view,
            embedding_field=entity_embedding_field,
            id_field=entity_id_field,
            node_type="entity",
            adjacency_fields=[entity_neighbors_field, entity_relation_field],
            top_k=top_k_entities,
            query=query_embedding,
        )
        relation_hits = self._search_view(
            view=relation_view,
            embedding_field=relation_embedding_field,
            id_field=relation_id_field,
            node_type="relation",
            adjacency_fields=[relation_entity_field, relation_passage_field],
            top_k=top_k_relations,
            query=query_embedding,
        )

        ordered_relation_hits = self._rerank_relations(
            relation_hits, reranker, query_text
        )
        relation_hits = ordered_relation_hits[:top_k_relations]

        nodes, edges = self._expand_subgraph(
            entity_hits=entity_hits,
            relation_hits=relation_hits,
            entity_neighbors_field=entity_neighbors_field,
            entity_relation_field=entity_relation_field,
            relation_entity_field=relation_entity_field,
            relation_passage_field=relation_passage_field,
            max_hops=max_hops,
            max_entities=max_expanded_entities,
            max_relations=max_expanded_relations,
        )

        passage_ids: Set[str] = set()
        for relation in relation_hits:
            if relation_passage_field and relation_passage_field in relation.metadata:
                passage_ids.update(relation.metadata.get(relation_passage_field, []) or [])

        if passage_relation_field and not passage_ids:
            passage_ids.update(
                self._passages_from_relation_field(
                    passage_view=passage_view,
                    passage_id_field=passage_id_field,
                    passage_relation_field=passage_relation_field,
                    passage_embedding_field=passage_embedding_field,
                    query_embedding=query_embedding,
                    relation_ids=[rel.node_id for rel in nodes["relations"].values()],
                    top_k=top_k_passages,
                )
            )

        passages = self._fetch_passages(
            passage_view=passage_view,
            passage_id_field=passage_id_field,
            passage_relation_field=passage_relation_field,
            passage_text_field=passage_text_field,
            passage_ids=list(passage_ids)[:top_k_passages],
        )

        return GraphRAGResult(
            entities=list(nodes["entities"].values()),
            relations=list(nodes["relations"].values()),
            passages=passages,
            edges=edges,
        )

    def _search_view(
        self,
        view: str,
        embedding_field: str,
        id_field: str,
        node_type: Literal["entity", "relation"],
        adjacency_fields: Sequence[Optional[str]],
        top_k: int,
        query: List[float],
    ) -> List[GraphNode]:
        features = [f"{view}:{embedding_field}", f"{view}:{id_field}"]
        for field in adjacency_fields:
            if field:
                features.append(f"{view}:{field}")
        response = self.store.retrieve_online_documents_v2(
            features=features, query=query, top_k=top_k
        )
        result = response.to_dict()
        ids = result.get(id_field, [])
        distances = result.get("distance", [0.0] * len(ids))
        hits: List[GraphNode] = []
        for idx, node_id in enumerate(ids):
            metadata = {
                key: result.get(key, [None] * len(ids))[idx]
                for key in result.keys()
                if key not in {"distance"}
            }
            metadata.pop("id", None)
            hits.append(
                GraphNode(
                    node_id=str(node_id),
                    node_type=node_type,
                    metadata={**metadata, "score": distances[idx] if idx < len(distances) else None},
                )
            )
        return hits

    def _rerank_relations(
        self,
        relation_hits: List[GraphNode],
        reranker: Optional[Reranker],
        query_text: Optional[str],
    ) -> List[GraphNode]:
        if not reranker:
            return relation_hits

        candidates = [
            RelationCandidate(
                relation_id=node.node_id,
                score=float(node.metadata.get("score", 0.0)),
                metadata=node.metadata,
            )
            for node in relation_hits
        ]
        reranked = reranker(candidates, query_text)
        reordered: List[GraphNode] = []
        id_to_node = {node.node_id: node for node in relation_hits}
        for candidate in reranked:
            if candidate.relation_id in id_to_node:
                reordered.append(id_to_node[candidate.relation_id])
        for node in relation_hits:
            if node not in reordered:
                reordered.append(node)
        return reordered

    def _expand_subgraph(
        self,
        *,
        entity_hits: List[GraphNode],
        relation_hits: List[GraphNode],
        entity_neighbors_field: Optional[str],
        entity_relation_field: Optional[str],
        relation_entity_field: Optional[str],
        relation_passage_field: Optional[str],
        max_hops: int,
        max_entities: int,
        max_relations: int,
    ) -> tuple[Dict[str, Dict[str, GraphNode]], List[GraphEdge]]:
        entities: Dict[str, GraphNode] = {}
        relations: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []

        def add_entity(node: GraphNode):
            if node.node_id not in entities and len(entities) < max_entities:
                entities[node.node_id] = GraphNode(
                    node_id=node.node_id, node_type="entity", metadata=node.metadata
                )

        def add_relation(node: GraphNode):
            if node.node_id not in relations and len(relations) < max_relations:
                relations[node.node_id] = GraphNode(
                    node_id=node.node_id, node_type="relation", metadata=node.metadata
                )

        for node in entity_hits:
            add_entity(node)
        for node in relation_hits:
            add_relation(node)

        frontier_entities: Set[str] = set(entities.keys())
        frontier_relations: Set[str] = set(relations.keys())
        processed_entities: Set[str] = set()
        processed_relations: Set[str] = set()

        for _ in range(max_hops):
            next_entities: Set[str] = set()
            next_relations: Set[str] = set()

            for entity_id in list(frontier_entities):
                node = entities.get(entity_id)
                if not node:
                    continue
                neighbors = self._safe_list(node.metadata.get(entity_neighbors_field))
                for neighbor_id in neighbors:
                    if neighbor_id not in entities and len(entities) < max_entities:
                        add_entity(GraphNode(neighbor_id, "entity", {}))
                    edges.append(GraphEdge(entity_id, str(neighbor_id), "neighbor"))
                    next_entities.add(str(neighbor_id))
                if entity_relation_field:
                    rel_ids = self._safe_list(node.metadata.get(entity_relation_field))
                    for rel_id in rel_ids:
                        if rel_id not in relations and len(relations) < max_relations:
                            add_relation(GraphNode(str(rel_id), "relation", {}))
                        edges.append(GraphEdge(entity_id, str(rel_id), "relation"))
                        next_relations.add(str(rel_id))

            for relation_id in list(frontier_relations):
                node = relations.get(relation_id)
                if not node:
                    continue
                if relation_entity_field:
                    ent_ids = self._safe_list(node.metadata.get(relation_entity_field))
                    for ent_id in ent_ids:
                        if ent_id not in entities and len(entities) < max_entities:
                            add_entity(GraphNode(str(ent_id), "entity", {}))
                        edges.append(GraphEdge(relation_id, str(ent_id), "connected"))
                        next_entities.add(str(ent_id))
                if relation_passage_field:
                    passage_ids = self._safe_list(node.metadata.get(relation_passage_field))
                    for passage_id in passage_ids:
                        edges.append(GraphEdge(relation_id, str(passage_id), "passage"))

            processed_entities.update(frontier_entities)
            processed_relations.update(frontier_relations)
            frontier_entities = next_entities - processed_entities
            frontier_relations = next_relations - processed_relations

        return {"entities": entities, "relations": relations}, edges

    def _safe_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _passages_from_relation_field(
        self,
        *,
        passage_view: str,
        passage_id_field: str,
        passage_relation_field: str,
        passage_embedding_field: str,
        query_embedding: List[float],
        relation_ids: Iterable[str],
        top_k: int,
    ) -> Set[str]:
        response = self.store.retrieve_online_documents_v2(
            features=[
                f"{passage_view}:{passage_embedding_field}",
                f"{passage_view}:{passage_relation_field}",
                f"{passage_view}:{passage_id_field}",
            ],
            query=query_embedding,
            top_k=top_k,
        )
        result = response.to_dict()
        passage_ids: Set[str] = set()
        for idx, rid_list in enumerate(result.get(passage_relation_field, [])):
            rid_set = set(self._safe_list(rid_list))
            if rid_set.intersection(set(relation_ids)):
                pid_list = result.get(passage_id_field, [])
                if idx < len(pid_list):
                    passage_ids.add(str(pid_list[idx]))
        return passage_ids

    def _fetch_passages(
        self,
        *,
        passage_view: str,
        passage_id_field: str,
        passage_relation_field: Optional[str],
        passage_text_field: Optional[str],
        passage_ids: List[str],
    ) -> List[GraphPassage]:
        if not passage_ids:
            return []
        features = [
            f"{passage_view}:{passage_relation_field}" if passage_relation_field else None,
            f"{passage_view}:{passage_text_field}" if passage_text_field else None,
        ]
        features = [f for f in features if f]
        response = self.store.get_online_features(
            features=features,
            entity_rows=[{passage_id_field: pid} for pid in passage_ids],
            full_feature_names=False,
        )
        result = response.to_dict()
        passages: List[GraphPassage] = []
        for idx, pid in enumerate(passage_ids):
            metadata = {key: val[idx] for key, val in result.items()}
            text_val = (
                metadata.pop(passage_text_field, None)
                if passage_text_field and passage_text_field in metadata
                else None
            )
            passages.append(
                GraphPassage(passage_id=str(pid), text=text_val, metadata=metadata)
            )
        return passages
