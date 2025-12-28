import pytest

from feast.entity import Entity
from feast.infra.registry.registry import Registry
from feast.project import Project
from feast.repo_config import RegistryConfig
from feast.value_type import ValueType


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_apply_project_preserves_existing_registry_data(tmp_path, cache_enabled):
    registry_path = tmp_path / "registry.db"
    initial_config = RegistryConfig(path=str(registry_path), cache_enabled=True)
    initial_registry = Registry("existing_project", initial_config, tmp_path)
    initial_registry.apply_entity(
        Entity(name="driver_id", value_type=ValueType.STRING),
        project="existing_project",
    )

    registry_config = RegistryConfig(
        path=str(registry_path), cache_enabled=cache_enabled
    )
    registry = Registry("new_project", registry_config, tmp_path)
    registry.apply_project(Project(name="new_project"))

    registry_proto = registry._registry_store.get_registry_proto()

    assert any(
        entity.spec.name == "driver_id" for entity in registry_proto.entities
    )
    assert any(
        project.spec.name == "existing_project"
        for project in registry_proto.projects
    )
