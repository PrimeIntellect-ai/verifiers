"""A task that declares the kernel it needs is never run by a taskset that cannot boot it."""

import pytest

from verifiers.v1.tasksets.harbor import taskset as harbor


def test_a_plain_harbor_taskset_refuses_a_task_that_declares_a_kernel(
    tmp_path, monkeypatch
):
    task = tmp_path / "needs-landlock"
    (task / "tests").mkdir(parents=True)
    (task / "tests/test.sh").write_text("exit 0\n")
    (task / "instruction.md").write_text("Implement the requested change.")
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "python:3.11-slim"\n'
        '[metadata.prime_kernel]\npackage = "linux-image-6.12.113+deb13-amd64"\n'
    )
    monkeypatch.setattr(harbor, "dataset_dir", lambda config: tmp_path)
    with pytest.raises(ValueError, match="sft-gen-kernel-harbor"):
        list(harbor.HarborTaskset(harbor.HarborConfig()))
    [loaded] = list(harbor.HarborTaskset(harbor.HarborConfig(boots_kernels=True)))
    assert loaded.data.name == "needs-landlock"


def test_a_task_toml_inside_another_task_is_part_of_it(tmp_path, monkeypatch):
    """A fixture task definition a task's environment ships is not loaded as a task."""
    task = tmp_path / "outer"
    (task / "tests").mkdir(parents=True)
    (task / "tests/test.sh").write_text("exit 0\n")
    (task / "instruction.md").write_text("Implement the requested change.")
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "python:3.11-slim"\n'
    )
    nested = task / "environment/compose/verifier-task"
    nested.mkdir(parents=True)
    (nested / "instruction.md").write_text(
        "A fixture, built into the outer task's image."
    )
    (nested / "task.toml").write_text(
        "[environment]\n"
    )  # no image: loading it would fail
    monkeypatch.setattr(harbor, "dataset_dir", lambda config: tmp_path)
    assert [t.data.name for t in harbor.HarborTaskset(harbor.HarborConfig())] == [
        "outer"
    ]


def test_a_verifier_table_naming_only_its_image_is_sized_like_the_environment(
    tmp_path,
):
    """A separate verifier whose `[verifier.environment]` names only an image grades
    with the task's declared sizes rather than the provider's smallest box; sizes the
    table states still win."""
    task = tmp_path / "sized"
    (task / "tests").mkdir(parents=True)
    (task / "tests/test.sh").write_text("exit 0\n")
    (task / "instruction.md").write_text("Implement the requested change.")
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "python:3.11-slim"\n'
        "cpus = 4\nmemory_mb = 7168\nstorage_mb = 10240\n"
        '[verifier]\nenvironment_mode = "separate"\n'
        '[verifier.environment]\ndocker_image = "python:3.12-slim"\ncpus = 2\n'
    )
    data = harbor.parse_task(task, 0, harbor.HarborConfig())
    assert data.verifier_image == "python:3.12-slim"
    assert data.verifier.resources.model_dump() == {
        "cpu": 2,
        "memory": 7.0,
        "gpu": None,
        "disk": 10.0,
    }
