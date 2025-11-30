# Copyright 2024 The Feast Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import pytest

from feast.infra.online_stores.milvus_online_store.milvus import (
    MilvusOnlineStoreConfig,
)


class TestMilvusOnlineStoreConfig(unittest.TestCase):
    def test_default_config(self):
        """Test that default configuration values are set correctly."""
        config = MilvusOnlineStoreConfig()

        assert config.type == "milvus"
        assert config.path == ""
        assert config.host == "http://localhost"
        assert config.port == 19530
        assert config.index_type == "FLAT"
        assert config.metric_type == "COSINE"
        assert config.embedding_dim == 128
        assert config.vector_enabled is True
        assert config.text_search_enabled is False
        assert config.nlist == 128
        assert config.username == ""
        assert config.password == ""
        assert config.db_name == ""

    def test_custom_config(self):
        """Test that custom configuration values can be set."""
        config = MilvusOnlineStoreConfig(
            host="http://milvus-server",
            port=19531,
            index_type="IVF_FLAT",
            metric_type="L2",
            embedding_dim=768,
            username="admin",
            password="secret",
            db_name="marketing_db",
        )

        assert config.host == "http://milvus-server"
        assert config.port == 19531
        assert config.index_type == "IVF_FLAT"
        assert config.metric_type == "L2"
        assert config.embedding_dim == 768
        assert config.username == "admin"
        assert config.password == "secret"
        assert config.db_name == "marketing_db"

    def test_db_name_for_multi_tenancy(self):
        """Test that db_name can be configured for database-level multi-tenancy."""
        # Marketing team configuration
        marketing_config = MilvusOnlineStoreConfig(
            db_name="marketing_db",
            embedding_dim=768,
            metric_type="COSINE",
        )

        # Sales team configuration
        sales_config = MilvusOnlineStoreConfig(
            db_name="sales_db",
            embedding_dim=128,
            metric_type="L2",
        )

        # Verify configurations are independent
        assert marketing_config.db_name == "marketing_db"
        assert sales_config.db_name == "sales_db"
        assert marketing_config.embedding_dim != sales_config.embedding_dim
        assert marketing_config.metric_type != sales_config.metric_type

    def test_local_mode_config(self):
        """Test configuration for local file-based Milvus."""
        config = MilvusOnlineStoreConfig(
            path="data/online_store.db",
            embedding_dim=256,
        )

        assert config.path == "data/online_store.db"
        assert config.embedding_dim == 256

    def test_config_serialization(self):
        """Test that configuration can be serialized to dict."""
        config = MilvusOnlineStoreConfig(
            db_name="test_db",
            embedding_dim=512,
        )

        config_dict = config.model_dump()

        assert config_dict["type"] == "milvus"
        assert config_dict["db_name"] == "test_db"
        assert config_dict["embedding_dim"] == 512


if __name__ == "__main__":
    unittest.main()
