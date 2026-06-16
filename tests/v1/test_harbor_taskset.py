from pathlib import Path

import pytest
from pydantic import ValidationError

from tasksets.harbor_v1 import HarborConfig, parse_task


@pytest.fixture
def harbor_task_dir(tmp_path: Path) -> Path:
    (tmp_path / "instruction.md").write_text("Do the task.")
    (tmp_path / "task.toml").write_text(
        """
[task]
name = "scaled-task"

[agent]
timeout_sec = 900

[verifier]
timeout_sec = 600

[environment]
cpus = 2
memory_mb = 4096
storage_mb = 8192
gpus = 2
""".strip()
    )
    return tmp_path


def test_harbor_multipliers_default_to_one(harbor_task_dir: Path):
    task = parse_task(harbor_task_dir, 0, HarborConfig())

    assert task.harness_timeout == 900
    assert task.scoring_timeout == 600
    assert task.resources.cpu == 2
    assert task.resources.memory == 4
    assert task.resources.disk == 8
    assert task.resources.gpu == "2"


def test_harbor_multipliers_scale_timeouts_and_resources(harbor_task_dir: Path):
    config = HarborConfig(timeout_multiplier=2, resource_multiplier=1.5)
    task = parse_task(harbor_task_dir, 0, config)

    assert task.harness_timeout == 1800
    assert task.scoring_timeout == 1200
    assert task.resources.cpu == 3
    assert task.resources.memory == 6
    assert task.resources.disk == 12
    assert task.resources.gpu == "2"


@pytest.mark.parametrize("field", ["timeout_multiplier", "resource_multiplier"])
def test_harbor_multipliers_must_be_positive(field: str):
    with pytest.raises(ValidationError):
        HarborConfig(**{field: 0})
