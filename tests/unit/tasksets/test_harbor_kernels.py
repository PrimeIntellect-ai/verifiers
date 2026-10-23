"""A task that declares the kernel it needs is never run by a taskset that cannot boot it."""

from pathlib import Path

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
    # The parsed task keeps what it declares, so its key does not move ...
    assert data.verifier.resources.model_dump() == {
        "cpu": 2,
        "memory": None,
        "gpu": None,
        "disk": None,
    }
    # ... and the verifier's box takes the environment's sizes for the rest.
    assert harbor.verifier_box_data(data).resources.model_dump() == {
        "cpu": 2,
        "memory": 7.0,
        "gpu": None,
        "disk": 10.0,
    }


def test_a_bare_name_caches_what_its_registry_resolves_to_now(tmp_path):
    """A bare dataset name follows its local registry's current `head`, so moving
    head to a new commit is a new cache entry; a pinned version keeps its own."""
    import json

    registry = tmp_path / "registry.json"

    def publish(commit):
        tasks = [{"name": "t", "git_url": "file:///r", "git_commit_id": commit}]
        registry.write_text(
            json.dumps(
                [
                    {"name": "x", "version": "v1", "tasks": tasks},
                    {"name": "x", "version": "head", "tasks": tasks},
                ]
            )
        )

    bare = harbor.HarborConfig(dataset="x", registry_path=registry)
    pinned = harbor.HarborConfig(dataset="x@v1", registry_path=registry)
    publish("aaa")
    first, first_pinned = harbor.cache_dir(bare), harbor.cache_dir(pinned)
    assert harbor.cache_dir(bare) == first
    publish("bbb")
    assert harbor.cache_dir(bare) != first
    assert harbor.cache_dir(pinned) == first_pinned


def test_staged_tests_are_root_owned(tmp_path):
    """/tests lands root-owned, with git's modes, as Harbor's copy leaves it."""
    import io
    import tarfile

    (tmp_path / "sub").mkdir(mode=0o775)
    (tmp_path / "sub" / "test.sh").write_text("exit 0\n")
    (tmp_path / "sub" / "test.sh").chmod(0o664)
    data = harbor.make_tar(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        owners = {(m.name, m.uid, m.gid, m.uname, m.mode) for m in tar.getmembers()}
    assert owners == {
        ("sub", 0, 0, "root", 0o755),
        ("sub/test.sh", 0, 0, "root", 0o644),
    }


def test_a_verifier_and_its_collect_hooks_may_name_their_user(tmp_path):
    """`[verifier].user` and a collect hook's `user` load, as Harbor runs them."""
    task = tmp_path / "users"
    (task / "tests").mkdir(parents=True)
    (task / "tests/test.sh").write_text("exit 0\n")
    (task / "instruction.md").write_text("Implement the requested change.")
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "python:3.11-slim"\n'
        '[verifier]\nuser = "grader"\n'
        '[[verifier.collect]]\ncommand = "true"\nuser = "root"\n'
    )
    data = harbor.parse_task(task, 0, harbor.HarborConfig())
    assert data.verifier_user == "grader"
    assert [hook.user for hook in data.collect] == ["root"]


def test_a_download_holding_lfs_pointers_is_not_cached(tmp_path, monkeypatch):
    """A Harbor export whose `git lfs pull` failed holds pointers where the task's
    binaries belong; it is refused rather than cached and graded from."""
    import subprocess

    def download(command, **kwargs):
        task = Path(command[command.index("-o") + 1]) / "t" / "solution"
        task.mkdir(parents=True)
        (task / "gold.bin").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 2742090\n"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(harbor, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(harbor, "harbor_cli", lambda: "harbor")
    monkeypatch.setattr(harbor.subprocess, "run", download)
    with pytest.raises(RuntimeError, match="Git LFS"):
        harbor.dataset_dir(harbor.HarborConfig(dataset="x@1"))
    assert not harbor.cache_dir(harbor.HarborConfig(dataset="x@1")).exists()


def test_unset_users_leave_the_task_data_dump_as_it_was():
    """`verifier_user` and a collect hook's `user` appear in the dump only when set,
    so tasks that name none keep their data hash (and every call fingerprint on it)."""
    data = harbor.HarborData(
        idx=0, prompt="p", collect=[harbor.CollectHook(command="true")]
    )
    dumped = data.model_dump(mode="json")
    assert "verifier_user" not in dumped and "user" not in dumped["collect"][0]
    named = data.model_copy(update={"verifier_user": "grader"})
    assert named.model_dump(mode="json")["verifier_user"] == "grader"
    assert harbor.HarborData.model_validate(dumped) == data


def test_a_compose_task_declares_its_host_image(tmp_path, monkeypatch):
    task = tmp_path / "stack"
    (task / "tests").mkdir(parents=True)
    (task / "tests/test.sh").write_text("exit 0\n")
    (task / "instruction.md").write_text("Do it.")
    toml = '[metadata.prime_compose]\nhost_image = "prime/team/stack:p-1"\n'
    (task / "task.toml").write_text(toml)
    monkeypatch.setattr(harbor, "dataset_dir", lambda config: tmp_path)
    with pytest.raises(ValueError, match="docker-compose"):
        list(harbor.HarborTaskset(harbor.HarborConfig()))
    (task / "environment").mkdir()
    (task / "environment/docker-compose.yaml").write_text(
        "services:\n  main:\n    image: python:3.11-slim\n"
    )
    [loaded] = list(harbor.HarborTaskset(harbor.HarborConfig()))
    assert loaded.data.compose_host_image == "prime/team/stack:p-1"
    assert loaded.deployment_spec().host_image == "prime/team/stack:p-1"
