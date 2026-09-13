import importlib.metadata

import pytest

from verifiers.v1.tasksets.inspect import InspectConfig, InspectTaskset
from verifiers.v1.tasksets.inspect.compat import require_inspect
from verifiers.v1.utils.loaders import taskset_class


def test_inspect_taskset_is_discoverable_without_inspect_dependency() -> None:
    assert taskset_class("inspect") is InspectTaskset


def test_missing_inspect_dependency_fails_only_when_used(monkeypatch) -> None:
    def missing_version(distribution_name: str) -> str:
        assert distribution_name == "inspect-ai"
        raise importlib.metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(importlib.metadata, "version", missing_version)

    with pytest.raises(ModuleNotFoundError, match="requires inspect-ai"):
        require_inspect()

    # Config construction and plugin discovery remain available without Inspect.
    InspectConfig(id="inspect", source="package/task")
