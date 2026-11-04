"""Scope safety tests use temporary regular files, never live cgroups or signals."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import _process_scope as controller
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.runtimes.process_scope import HELPER, ProcessScope
from verifiers.v1.runtimes.runtime import Runtime


@pytest.fixture
def fake_scope(tmp_path, monkeypatch):
    state = tmp_path / "state"
    parent = tmp_path / "delegated"
    root = parent / "vf-test"
    state.mkdir()
    root.mkdir(parents=True)
    for role in controller.ROLES:
        leaf = root / role
        leaf.mkdir()
        for name in ("cgroup.procs", "cgroup.freeze", "cgroup.kill"):
            (leaf / name).write_text("")
        (leaf / "cgroup.events").write_text("populated 0\nfrozen 0\n")
        (state / (role + ".state")).write_text("active")
    (state / "scope.json").write_text(json.dumps({"root": str(root)}))
    (state / "lock").touch()
    monkeypatch.setattr(controller, "STATE", state)
    monkeypatch.setattr(controller, "PARENT", parent)
    monkeypatch.setattr(controller, "secure_path", lambda path: None)
    return state, root


def test_freeze_requires_kernel_ack_and_thaw_never_reopens_gate(
    fake_scope, monkeypatch
):
    state, root = fake_scope
    monkeypatch.setattr(controller, "WAIT_SECONDS", 0)
    with pytest.raises(RuntimeError, match="not confirmed"):
        controller.transition("freeze", "agent")
    assert (state / "agent.state").read_text() == "frozen"
    with pytest.raises(RuntimeError, match="not accepting"):
        controller.launch("agent", ["true"], "worker")
    (root / "agent/cgroup.events").write_text("populated 0\nfrozen 1\n")
    controller.transition("freeze", "agent")
    (root / "agent/cgroup.events").write_text("populated 0\nfrozen 0\n")
    controller.transition("thaw", "agent")
    assert (state / "agent.state").read_text() == "quiescing"
    with pytest.raises(RuntimeError, match="not accepting"):
        controller.launch("agent", ["true"], "worker")


def test_kill_roles_are_isolated_and_scorer_reset_reopens_only_scorer(fake_scope):
    state, root = fake_scope
    controller.transition("kill", "scorer")
    assert (root / "scorer/cgroup.kill").read_text() == "1"
    assert (root / "agent/cgroup.kill").read_text() == ""
    assert (state / "scorer.state").read_text() == "active"
    controller.transition("kill", "agent")
    controller.transition("kill", "agent")
    assert (state / "agent.state").read_text() == "closed"


def test_launch_enrolls_before_exec_and_preserves_environment_after_drop(
    fake_scope, monkeypatch
):
    _, root = fake_scope
    monkeypatch.setattr(
        controller.pwd,
        "getpwnam",
        lambda user: SimpleNamespace(pw_uid=1001, pw_gid=1002),
    )
    monkeypatch.setenv("IMAGE_SETTING", "retained")
    monkeypatch.setenv("PATH", "/image/bin:/usr/bin")
    monkeypatch.setenv(
        "VF_SCOPE_CHILD_ENV",
        json.dumps({"HOME": "/home/worker", "LD_PRELOAD": "untrusted"}),
    )
    argv, env = controller.launch("agent", ["tool", "arg"], "worker")
    assert (root / "agent/cgroup.procs").read_text() == str(controller.os.getpid())
    assert argv[:3] == ["/usr/bin/setpriv", "--reuid", "1001"]
    assert "--no-new-privs" in argv and "--bounding-set=-all" in argv
    assert "IMAGE_SETTING=retained" in argv and "PATH=/image/bin:/usr/bin" in argv
    assert "LD_PRELOAD=untrusted" in argv
    assert env == {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"}
    assert not any(part.startswith("VF_SCOPE_CHILD_ENV=") for part in argv)


def test_root_agent_and_self_targeting_controller_are_rejected(fake_scope, monkeypatch):
    _, root = fake_scope
    monkeypatch.setattr(
        controller.pwd, "getpwnam", lambda user: SimpleNamespace(pw_uid=0, pw_gid=0)
    )
    with pytest.raises(RuntimeError, match="non-root"):
        controller.launch("agent", ["true"], "root")
    (root / "scorer/cgroup.procs").write_text(str(controller.os.getpid()))
    with pytest.raises(RuntimeError, match="outside"):
        controller.transition("kill", "scorer")


def test_controller_rejects_symlinks_and_writable_ancestors(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="canonical"):
        controller.secure_path(link)
    target.chmod(0o777)
    with pytest.raises(RuntimeError, match="protected"):
        controller.secure_path(target)


def test_registration_cannot_target_parent_or_other_subtrees(fake_scope):
    state, root = fake_scope
    for value in (str(root.parent), "/sys/fs/cgroup", str(root.parent / "other")):
        (state / "scope.json").write_text(json.dumps({"root": value}))
        with pytest.raises(RuntimeError, match="registration"):
            controller.scope_path("agent")
    with pytest.raises(ValueError, match="role"):
        controller.scope_path("init")


@pytest.mark.parametrize(
    "operation,options",
    [
        ("run", ["agent", "--user", "worker"]),
        ("exec", ["scorer"]),
    ],
)
def test_cli_preserves_existing_drop_argv(fake_scope, monkeypatch, operation, options):
    commands = []
    monkeypatch.setattr(controller.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        controller,
        "launch",
        lambda role, argv, user: commands.append((role, argv, user)) or (argv, {}),
    )
    monkeypatch.setattr(controller.os, "execvpe", lambda *args: None)
    monkeypatch.setattr(
        controller.sys,
        "argv",
        [
            "control",
            operation,
            *options,
            "--",
            "/usr/bin/runuser",
            "-u",
            "worker",
            "--",
            "true",
        ],
    )
    controller.main()
    assert commands[0][1] == ["/usr/bin/runuser", "-u", "worker", "--", "true"]


@pytest.fixture
def fake_prepare(tmp_path, monkeypatch):
    state = tmp_path / "state"
    mount = tmp_path / "cgroup2"
    parent = mount / "vf-task-scopes"
    state.mkdir(mode=0o700)
    mount.mkdir(mode=0o755)
    owners = {}
    real_stat = controller.Path.stat
    real_mkdir = controller.Path.mkdir
    real_is_file = controller.Path.is_file

    def root_owned_stat(path, *args, **kwargs):
        info = real_stat(path, *args, **kwargs)
        mode = info.st_mode
        if path == tmp_path or path in tmp_path.parents:
            mode &= ~0o022
        return SimpleNamespace(st_uid=owners.get(path, 0), st_mode=mode)

    def cgroup_mkdir(path, *args, **kwargs):
        real_mkdir(path, *args, **kwargs)
        if path == parent:
            (path / "cgroup.controllers").touch()
        elif path.parent.parent == parent:
            for name in ("cgroup.procs", "cgroup.freeze", "cgroup.kill"):
                (path / name).touch()
            (path / "cgroup.events").write_text("populated 0\nfrozen 0\n")

    monkeypatch.setattr(controller, "STATE", state)
    monkeypatch.setattr(controller, "PARENT", parent)
    # Emulate ownership, kernel-created files and mount type only on temporary files.
    monkeypatch.setattr(controller.Path, "stat", root_owned_stat)
    monkeypatch.setattr(controller.Path, "mkdir", cgroup_mkdir)
    monkeypatch.setattr(
        controller.Path,
        "is_file",
        lambda path: (
            str(path) in ("/usr/bin/setpriv", "/usr/bin/env") or real_is_file(path)
        ),
    )
    monkeypatch.setattr(controller, "cgroup2_mount", lambda path: True)
    return SimpleNamespace(state=state, parent=parent, mount=mount, owners=owners)


def test_prepare_bootstraps_missing_parent_on_protected_cgroup2(fake_prepare):
    fixture = fake_prepare
    controller.prepare()
    assert fixture.parent.stat().st_mode & 0o777 == 0o700
    root = controller.Path(
        json.loads((fixture.state / "scope.json").read_text())["root"]
    )
    assert root.parent == fixture.parent
    assert all((root / role).is_dir() for role in controller.ROLES)
    assert (fixture.state / "agent.state").read_text() == "active"


def test_prepare_preserves_existing_secure_parent_permissions(fake_prepare):
    fake_prepare.parent.mkdir(mode=0o750)
    controller.prepare()
    assert fake_prepare.parent.stat().st_mode & 0o777 == 0o750


@pytest.mark.parametrize("insecure", ["mount", "parent_mode", "parent_owner"])
def test_prepare_rejects_unprotected_paths_without_chmod(fake_prepare, insecure):
    fixture = fake_prepare
    if insecure == "mount":
        fixture.mount.chmod(0o777)
    else:
        fixture.parent.mkdir(mode=0o700)
        if insecure == "parent_mode":
            fixture.parent.chmod(0o777)
        else:
            fixture.owners[fixture.parent] = 1000
    with pytest.raises(RuntimeError, match="protected"):
        controller.prepare()
    assert not (fixture.state / "scope.json").exists()
    if insecure == "mount":
        assert not fixture.parent.exists()
        assert fixture.mount.stat().st_mode & 0o777 == 0o777
    elif insecure == "parent_mode":
        assert fixture.parent.stat().st_mode & 0o777 == 0o777


@pytest.mark.parametrize("existing_parent", [False, True])
def test_prepare_rejects_non_cgroup2_before_creating_scopes(
    fake_prepare, monkeypatch, existing_parent
):
    fixture = fake_prepare
    if existing_parent:
        fixture.parent.mkdir(mode=0o700)
    monkeypatch.setattr(controller, "cgroup2_mount", lambda path: False)
    with pytest.raises(RuntimeError, match="cgroup-v2"):
        controller.prepare()
    assert fixture.parent.exists() == existing_parent
    assert not (fixture.state / "scope.json").exists()


def test_prepare_rejects_non_cgroup2_mount_over_existing_parent(
    fake_prepare, monkeypatch
):
    fixture = fake_prepare
    fixture.parent.mkdir(mode=0o700)
    monkeypatch.setattr(controller, "cgroup2_mount", lambda path: path == fixture.mount)
    with pytest.raises(RuntimeError, match="cgroup-v2"):
        controller.prepare()
    assert not (fixture.state / "scope.json").exists()


class RecordingBackend:
    SUPPORTS_USER_SWITCHING = True
    SUPPORTS_TASK_IMAGES = True
    config = SimpleNamespace(type="recording")

    @property
    def type(self):
        return self.config.type

    def __init__(self):
        self.calls = []

    async def run(self, runtime, argv, env):
        self.calls.append(("run", runtime, argv, env, None))
        return ProgramResult(0, "", "")

    async def exec(self, runtime, argv, env):
        self.calls.append(("exec", runtime, argv, env, None))
        return object()

    async def run_background(self, runtime, argv, env, log):
        self.calls.append(("background", runtime, argv, env, log))


async def test_all_runtime_launches_use_root_entry_then_nonroot_workload():
    backend = RecordingBackend()
    root = Runtime(backend)
    scope = ProcessScope(root)
    agent = replace(root.with_user("worker", "/home/worker"), process_scope=scope)
    await agent.run(["true"], {"LD_PRELOAD": "untrusted"})
    await agent.exec(["true"], {})
    await agent.run_background(["true"], {}, "/home/worker/log")
    assert len(backend.calls) == 3
    for _, runtime, argv, env, _ in backend.calls:
        assert runtime.user is None and runtime.process_scope is None
        assert argv[:6] == [HELPER, "run", "agent", "--user", "worker", "--"]
        assert set(env) == {"VF_SCOPE_CHILD_ENV"}
        assert json.loads(env["VF_SCOPE_CHILD_ENV"])["HOME"] == "/home/worker"
    assert backend.calls[-1][-1] == "/dev/null"
    assert agent.with_user(None).process_scope is None


@pytest.mark.parametrize(
    "backend_type,supports_images,supports_users",
    [
        ("prime", False, True),
        ("prime", True, False),
        ("docker", True, True),
        ("subprocess", True, True),
        ("recording", True, True),
    ],
)
async def test_unsupported_runtime_fails_before_any_command(
    backend_type, supports_images, supports_users
):
    backend = RecordingBackend()
    backend.config = SimpleNamespace(type=backend_type)
    backend.SUPPORTS_TASK_IMAGES = supports_images
    backend.SUPPORTS_USER_SWITCHING = supports_users
    with pytest.raises(SandboxError, match="isolated"):
        await ProcessScope.prepare(Runtime(backend))
    assert not backend.calls


async def test_scorer_failure_kills_agent_without_thaw(monkeypatch):
    calls = []

    async def command(self, *argv):
        calls.append(argv)
        if argv == ("kill", "scorer"):
            raise SandboxError("unconfirmed")

    monkeypatch.setattr(ProcessScope, "command", command)
    scope = ProcessScope(Runtime(RecordingBackend()))
    with pytest.raises(SandboxError, match="unconfirmed"):
        await scope.finish_verification()
    assert calls == [("kill", "scorer"), ("kill", "agent")]


async def test_cancellation_finishes_scorer_cleanup_and_kills_agent(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def command(self, *argv):
        calls.append(argv)
        if argv == ("kill", "scorer"):
            entered.set()
            await release.wait()

    monkeypatch.setattr(ProcessScope, "command", command)
    scope = ProcessScope(Runtime(RecordingBackend()))
    task = asyncio.create_task(scope.finish_verification())
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == [("kill", "scorer"), ("kill", "agent")]


async def test_prepare_uses_root_protected_image_python():
    class ImageBackend(RecordingBackend):
        config = SimpleNamespace(type="prime")

        async def run(self, runtime, argv, env):
            self.calls.append(("run", runtime, argv, env, None))
            if argv[:2] == ["/usr/local/bin/python3", "-I"]:
                return ProgramResult(0, "/usr/local/bin/python3.13\n", "")
            return ProgramResult(0, "", "")

        async def write(self, runtime, path, data):
            self.helper = data

    backend = ImageBackend()
    await ProcessScope.prepare(Runtime(backend))
    assert backend.helper.startswith(b"#!/usr/local/bin/python3.13 -I\n")
