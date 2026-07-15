"""Harbor task.toml resource parsing across schema versions."""

import pytest

from verifiers.v1.tasksets.harbor.taskset import parse_resources, size_to_mb


def test_current_schema_mb_fields():
    resources = parse_resources(
        {"cpus": 2, "memory_mb": 8192, "storage_mb": 20480, "gpus": 0}
    )
    assert (resources.cpu, resources.memory, resources.disk, resources.gpu) == (
        2,
        8.0,
        20.0,
        None,
    )


def test_legacy_size_strings():
    resources = parse_resources({"memory": "8G", "storage": "512M"})
    assert resources.memory == 8.0
    assert resources.disk == 0.5


def test_current_schema_wins_and_multiplier_applies():
    resources = parse_resources({"memory_mb": 1024, "memory": "8G"}, multiplier=2.0)
    assert resources.memory == 2.0


def test_invalid_size_string():
    with pytest.raises(ValueError):
        size_to_mb("8QB")
