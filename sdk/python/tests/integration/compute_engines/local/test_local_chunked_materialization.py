from datetime import datetime, timedelta, timezone
import socket

import pandas as pd
import pytest

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.spark_offline_store.spark import (
    SparkOfflineStoreConfig,
)
from feast.infra.offline_stores.contrib.spark_offline_store.tests.data_source import (
    SparkDataSourceCreator,
)
from feast.types import Float32
from tests.integration.feature_repos.integration_test_repo_config import (
    IntegrationTestRepoConfig,
)
from tests.integration.feature_repos.repo_configuration import construct_test_environment


@pytest.mark.integration
def test_local_materialization_with_spark_staging_chunked(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_LOCAL_IP", "127.0.0.1")
    monkeypatch.setenv("SPARK_LOCAL_HOSTNAME", "localhost")
    monkeypatch.setenv(
        "PYSPARK_SUBMIT_ARGS",
        "--conf spark.driver.host=127.0.0.1 "
        "--conf spark.driver.bindAddress=127.0.0.1 pyspark-shell",
    )

    try:
        socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        pytest.skip("Local hostname is not resolvable for Spark initialization")
    config = IntegrationTestRepoConfig(
        provider="local",
        offline_store_creator=SparkDataSourceCreator,
        online_store="sqlite",
        batch_engine={"type": "local", "chunk_size": 1},
    )
    environment = construct_test_environment(
        config, None, test_suite_name="local_spark_chunked"
    )
    environment.setup()

    try:
        offline_config = environment.config.offline_store
        assert isinstance(offline_config, SparkOfflineStoreConfig)
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        offline_config.staging_location = str(staging_dir)
        offline_config.staging_allow_materialize = True

        now = datetime.now(timezone.utc)
        df = pd.DataFrame(
            [
                {
                    "driver_id": 1001,
                    "event_timestamp": now - timedelta(hours=2),
                    "created": now - timedelta(hours=2),
                    "conv_rate": 0.3,
                },
                {
                    "driver_id": 1001,
                    "event_timestamp": now - timedelta(hours=1),
                    "created": now - timedelta(hours=1),
                    "conv_rate": 0.8,
                },
                {
                    "driver_id": 1002,
                    "event_timestamp": now - timedelta(hours=3),
                    "created": now - timedelta(hours=3),
                    "conv_rate": 0.2,
                },
                {
                    "driver_id": 1002,
                    "event_timestamp": now - timedelta(hours=1),
                    "created": now - timedelta(hours=1),
                    "conv_rate": 0.6,
                },
            ]
        )

        data_source = environment.data_source_creator.create_data_source(
            df,
            environment.feature_store.project,
            timestamp_field="event_timestamp",
            created_timestamp_column="created",
        )

        driver = Entity(name="driver_id", join_keys=["driver_id"])
        feature_view = FeatureView(
            name="driver_stats",
            entities=[driver],
            schema=[Field(name="conv_rate", dtype=Float32)],
            source=data_source,
        )

        store = environment.feature_store
        store.apply([driver, feature_view])

        start_date = df["event_timestamp"].min().to_pydatetime() - timedelta(seconds=1)
        end_date = df["event_timestamp"].max().to_pydatetime() + timedelta(seconds=1)

        store.materialize(start_date, end_date)

        response = store.get_online_features(
            features=["driver_stats:conv_rate"],
            entity_rows=[{"driver_id": 1001}, {"driver_id": 1002}],
            full_feature_names=True,
        ).to_dict()

        assert response["driver_id"] == [1001, 1002]
        assert response["driver_stats__conv_rate"] == pytest.approx([0.8, 0.6])

        parquet_files = list(staging_dir.rglob("*.parquet"))
        assert parquet_files, "Expected staging parquet files to be written"
    finally:
        environment.teardown()


@pytest.mark.integration
def test_local_materialization_with_spark_staging_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_LOCAL_IP", "127.0.0.1")
    monkeypatch.setenv("SPARK_LOCAL_HOSTNAME", "localhost")
    monkeypatch.setenv(
        "PYSPARK_SUBMIT_ARGS",
        "--conf spark.driver.host=127.0.0.1 "
        "--conf spark.driver.bindAddress=127.0.0.1 pyspark-shell",
    )

    try:
        socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        pytest.skip("Local hostname is not resolvable for Spark initialization")
    config = IntegrationTestRepoConfig(
        provider="local",
        offline_store_creator=SparkDataSourceCreator,
        online_store="sqlite",
        batch_engine={"type": "local"},
    )
    environment = construct_test_environment(
        config, None, test_suite_name="local_spark_staging"
    )
    environment.setup()

    try:
        offline_config = environment.config.offline_store
        assert isinstance(offline_config, SparkOfflineStoreConfig)
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        offline_config.staging_location = str(staging_dir)
        offline_config.staging_allow_materialize = True

        now = datetime.now(timezone.utc)
        df = pd.DataFrame(
            [
                {
                    "driver_id": 1001,
                    "event_timestamp": now - timedelta(hours=2),
                    "created": now - timedelta(hours=2),
                    "conv_rate": 0.3,
                },
                {
                    "driver_id": 1001,
                    "event_timestamp": now - timedelta(hours=1),
                    "created": now - timedelta(hours=1),
                    "conv_rate": 0.8,
                },
            ]
        )

        data_source = environment.data_source_creator.create_data_source(
            df,
            environment.feature_store.project,
            timestamp_field="event_timestamp",
            created_timestamp_column="created",
        )

        driver = Entity(name="driver_id", join_keys=["driver_id"])
        feature_view = FeatureView(
            name="driver_stats",
            entities=[driver],
            schema=[Field(name="conv_rate", dtype=Float32)],
            source=data_source,
        )

        store = environment.feature_store
        store.apply([driver, feature_view])

        start_date = df["event_timestamp"].min().to_pydatetime() - timedelta(seconds=1)
        end_date = df["event_timestamp"].max().to_pydatetime() + timedelta(seconds=1)

        store.materialize(start_date, end_date)

        parquet_files = list(staging_dir.rglob("*.parquet"))
        assert parquet_files, "Expected staging parquet files to be written"
    finally:
        environment.teardown()


@pytest.mark.integration
def test_local_materialization_with_spark_chunking_multiple_entities(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SPARK_LOCAL_IP", "127.0.0.1")
    monkeypatch.setenv("SPARK_LOCAL_HOSTNAME", "localhost")
    monkeypatch.setenv(
        "PYSPARK_SUBMIT_ARGS",
        "--conf spark.driver.host=127.0.0.1 "
        "--conf spark.driver.bindAddress=127.0.0.1 pyspark-shell",
    )

    try:
        socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        pytest.skip("Local hostname is not resolvable for Spark initialization")
    config = IntegrationTestRepoConfig(
        provider="local",
        offline_store_creator=SparkDataSourceCreator,
        online_store="sqlite",
        batch_engine={"type": "local", "chunk_size": 1},
    )
    environment = construct_test_environment(
        config, None, test_suite_name="local_spark_chunking"
    )
    environment.setup()

    try:
        offline_config = environment.config.offline_store
        assert isinstance(offline_config, SparkOfflineStoreConfig)
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        offline_config.staging_location = str(staging_dir)
        offline_config.staging_allow_materialize = True

        now = datetime.now(timezone.utc)
        rows = []
        for driver_id in range(1001, 1011):
            rows.append(
                {
                    "driver_id": driver_id,
                    "event_timestamp": now - timedelta(hours=1),
                    "created": now - timedelta(hours=1),
                    "conv_rate": float(driver_id % 10) / 10.0,
                }
            )
        df = pd.DataFrame(rows)

        data_source = environment.data_source_creator.create_data_source(
            df,
            environment.feature_store.project,
            timestamp_field="event_timestamp",
            created_timestamp_column="created",
        )

        driver = Entity(name="driver_id", join_keys=["driver_id"])
        feature_view = FeatureView(
            name="driver_stats",
            entities=[driver],
            schema=[Field(name="conv_rate", dtype=Float32)],
            source=data_source,
        )

        store = environment.feature_store
        store.apply([driver, feature_view])

        start_date = df["event_timestamp"].min().to_pydatetime() - timedelta(seconds=1)
        end_date = df["event_timestamp"].max().to_pydatetime() + timedelta(seconds=1)

        store.materialize(start_date, end_date)

        response = store.get_online_features(
            features=["driver_stats:conv_rate"],
            entity_rows=[{"driver_id": driver_id} for driver_id in range(1001, 1011)],
            full_feature_names=True,
        ).to_dict()

        assert response["driver_id"] == list(range(1001, 1011))
        expected_rates = [float(driver_id % 10) / 10.0 for driver_id in range(1001, 1011)]
        assert response["driver_stats__conv_rate"] == pytest.approx(expected_rates)

        parquet_files = list(staging_dir.rglob("*.parquet"))
        assert parquet_files, "Expected staging parquet files to be written"
    finally:
        environment.teardown()
