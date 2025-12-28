import argparse
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timedelta, timezone

SDK_PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(SDK_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_PYTHON_ROOT))

from pyspark.sql import functions as F

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.spark_offline_store.spark import (
    SparkOfflineStoreConfig,
)
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import (
    SparkSource,
)
from feast.infra.offline_stores.contrib.spark_offline_store.tests.data_source import (
    SparkDataSourceCreator,
)
from feast.types import Float32, String
from tests.integration.feature_repos.integration_test_repo_config import (
    IntegrationTestRepoConfig,
)
from tests.integration.feature_repos.repo_configuration import construct_test_environment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manual stress test that can trigger OOM in the non-staging path. "
            "Use with care."
        )
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=50_000_000,
        help="Number of rows to generate in Spark.",
    )
    parser.add_argument(
        "--payload-size",
        type=int,
        default=1024,
        help="Size in bytes for the payload column per row.",
    )
    parser.add_argument(
        "--driver-cardinality",
        type=int,
        default=100_000,
        help="Number of distinct driver_id values.",
    )
    parser.add_argument(
        "--use-staging",
        action="store_true",
        help="Enable staging for Spark retrieval to avoid toPandas.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Chunk size for LocalComputeEngine output writes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Acknowledge that this test may OOM the machine.",
    )
    return parser.parse_args()


def _build_spark_source(spark_session, rows: int, payload_size: int, path: str):
    df = (
        spark_session.range(rows)
        .withColumn(
            "driver_id",
            (F.col("id") % F.lit(100_000)).cast("long"),
        )
        .withColumn("event_timestamp", F.current_timestamp())
        .withColumn("created", F.current_timestamp())
        .withColumn(
            "conv_rate",
            (F.col("id") % F.lit(100)).cast("double") / F.lit(100.0),
        )
        .withColumn("payload", F.repeat(F.lit("x"), payload_size))
        .drop("id")
    )

    df.write.mode("overwrite").parquet(path)
    return SparkSource(
        name="oom_source",
        file_format="parquet",
        path=path,
        timestamp_field="event_timestamp",
        created_timestamp_column="created",
    )


def main() -> None:
    args = _parse_args()
    if not args.force and os.getenv("FEAST_RUN_OOM") != "1":
        raise SystemExit(
            "Refusing to run without --force or FEAST_RUN_OOM=1. "
            "This script may OOM the machine."
        )

    config = IntegrationTestRepoConfig(
        provider="local",
        offline_store_creator=SparkDataSourceCreator,
        online_store="sqlite",
        batch_engine={"type": "local", "chunk_size": args.chunk_size}
        if args.chunk_size
        else {"type": "local"},
    )
    environment = construct_test_environment(
        config, None, test_suite_name="manual_oom"
    )
    environment.setup()

    try:
        offline_config = environment.config.offline_store
        if not isinstance(offline_config, SparkOfflineStoreConfig):
            raise RuntimeError("Expected SparkOfflineStoreConfig")

        staging_dir = tempfile.mkdtemp(prefix="feast_oom_staging_")
        offline_config.staging_location = staging_dir
        offline_config.staging_allow_materialize = bool(args.use_staging)

        spark_session = environment.data_source_creator.spark_session
        data_path = tempfile.mkdtemp(prefix="feast_oom_data_")
        source = _build_spark_source(
            spark_session, args.rows, args.payload_size, data_path
        )

        driver = Entity(name="driver_id", join_keys=["driver_id"])
        feature_view = FeatureView(
            name="driver_stats",
            entities=[driver],
            schema=[
                Field(name="conv_rate", dtype=Float32),
                Field(name="payload", dtype=String),
            ],
            source=source,
        )

        store = environment.feature_store
        store.apply([driver, feature_view])

        now = datetime.now(timezone.utc)
        start_date = now - timedelta(hours=1)
        end_date = now + timedelta(hours=1)

        mode = "staging" if args.use_staging else "no-staging"
        print(f"Running materialize in {mode} mode.")
        print(
            f"Rows={args.rows}, payload_size={args.payload_size}, "
            f"chunk_size={args.chunk_size}"
        )
        store.materialize(start_date, end_date)
        print("Materialize completed successfully.")
    finally:
        environment.teardown()


if __name__ == "__main__":
    main()
