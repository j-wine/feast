from datetime import timedelta
from typing import List

import numpy as np
import pytest
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, GraphRAGRetriever
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import FloatList as FloatListProto
from feast.protos.feast.types.Value_pb2 import StringList as StringListProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.types import Array, Float32, String, UnixTimestamp, ValueType
from feast.utils import _utc_now
from tests.utils.cli_repo_creator import CliRunner, get_example_repo


def _write_rows(store: FeatureStore, view: FeatureView, rows: List[dict]) -> None:
    provider = store._get_provider()
    data = []
    if not view.entities:
        raise ValueError("FeatureView must define at least one entity")
    first_entity = view.entities[0]
    join_key = first_entity.join_keys[0] if hasattr(first_entity, "join_keys") else first_entity
    for row in rows:
        entity_key = EntityKeyProto(
            join_keys=[join_key], entity_values=[ValueProto(string_val=row[join_key])]
        )
        feature_data = {
            "embedding": ValueProto(float_list_val=FloatListProto(val=row["embedding"])),
            join_key: ValueProto(string_val=row[join_key]),
        }
        if "neighbors" in row:
            feature_data["neighbors"] = ValueProto(
                string_list_val=StringListProto(val=row["neighbors"])
            )
        if "relation_ids" in row:
            feature_data["relation_ids"] = ValueProto(
                string_list_val=StringListProto(val=row["relation_ids"])
            )
        if "connected_entities" in row:
            feature_data["connected_entities"] = ValueProto(
                string_list_val=StringListProto(val=row["connected_entities"])
            )
        if "passage_ids" in row:
            feature_data["passage_ids"] = ValueProto(
                string_list_val=StringListProto(val=row["passage_ids"])
            )
        if "text" in row:
            feature_data["text"] = ValueProto(string_val=row["text"])
        provider.online_write_batch(
            config=store.config,
            table=view,
            data=[(entity_key, feature_data, _utc_now(), _utc_now())],
            progress=None,
        )


def test_graph_rag_retriever_expands_neighbors_and_passages() -> None:
    pytest.importorskip("pymilvus")
    runner = CliRunner()
    vector_length = 10

    with runner.local_repo(
        get_example_repo("example_feature_repo_1.py"),
        offline_store="file",
        online_store="milvus",
        apply=False,
        teardown=False,
    ) as store:
        entity = Entity(name="entity_id", join_keys=["entity_id"], value_type=ValueType.STRING)
        relation = Entity(
            name="relation_id", join_keys=["relation_id"], value_type=ValueType.STRING
        )
        passage = Entity(
            name="passage_id", join_keys=["passage_id"], value_type=ValueType.STRING
        )

        source = FileSource(
            path="data/graph_rag.parquet",
            timestamp_field="event_timestamp",
        )

        entity_view = FeatureView(
            name="graph_entities",
            entities=[entity],
            schema=[
                Field(name="entity_id", dtype=String),
                Field(
                    name="embedding",
                    dtype=Array(Float32),
                    vector_index=True,
                    vector_search_metric="COSINE",
                ),
                Field(name="neighbors", dtype=Array(String)),
                Field(name="relation_ids", dtype=Array(String)),
                Field(name="event_timestamp", dtype=UnixTimestamp),
            ],
            source=source,
            ttl=timedelta(hours=1),
        )

        relation_view = FeatureView(
            name="graph_relations",
            entities=[relation],
            schema=[
                Field(name="relation_id", dtype=String),
                Field(
                    name="embedding",
                    dtype=Array(Float32),
                    vector_index=True,
                    vector_search_metric="COSINE",
                ),
                Field(name="connected_entities", dtype=Array(String)),
                Field(name="passage_ids", dtype=Array(String)),
                Field(name="event_timestamp", dtype=UnixTimestamp),
            ],
            source=source,
            ttl=timedelta(hours=1),
        )

        passage_view = FeatureView(
            name="graph_passages",
            entities=[passage],
            schema=[
                Field(name="passage_id", dtype=String),
                Field(name="text", dtype=String),
                Field(
                    name="embedding",
                    dtype=Array(Float32),
                    vector_index=True,
                    vector_search_metric="COSINE",
                ),
                Field(name="relation_ids", dtype=Array(String)),
                Field(name="event_timestamp", dtype=UnixTimestamp),
            ],
            source=source,
            ttl=timedelta(hours=1),
        )

        store.apply([entity, relation, passage, entity_view, relation_view, passage_view])

        entity_rows = [
            {
                "entity_id": "n1",
                "neighbors": ["n2"],
                "relation_ids": ["r1"],
                "embedding": np.linspace(0.1, 1.0, vector_length),
            },
            {
                "entity_id": "n2",
                "neighbors": ["n1"],
                "relation_ids": ["r1"],
                "embedding": np.linspace(0.2, 1.1, vector_length),
            },
        ]

        relation_rows = [
            {
                "relation_id": "r1",
                "connected_entities": ["n1", "n2"],
                "passage_ids": ["p1"],
                "embedding": np.linspace(0.15, 1.05, vector_length),
            }
        ]

        passage_rows = [
            {
                "passage_id": "p1",
                "relation_ids": ["r1"],
                "text": "n1 is linked to n2",
                "embedding": np.linspace(0.12, 1.02, vector_length),
            }
        ]

        _write_rows(store, entity_view, entity_rows)
        _write_rows(store, relation_view, relation_rows)
        _write_rows(store, passage_view, passage_rows)

        retriever = GraphRAGRetriever(store)
        result = retriever.retrieve(
            query_embedding=entity_rows[0]["embedding"].tolist(),
            entity_view="graph_entities",
            relation_view="graph_relations",
            passage_view="graph_passages",
            top_k_entities=1,
            top_k_relations=1,
            top_k_passages=1,
            max_hops=2,
        )

        entity_ids = {node.node_id for node in result.entities}
        relation_ids = {node.node_id for node in result.relations}
        passage_ids = {p.passage_id for p in result.passages}

        assert {"n1", "n2"}.issubset(entity_ids)
        assert "r1" in relation_ids
        assert "p1" in passage_ids

        edge_pairs = {(edge.src_id, edge.dst_id) for edge in result.edges}
        assert ("n1", "n2") in edge_pairs or ("n2", "n1") in edge_pairs
