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
