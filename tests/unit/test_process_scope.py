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


class RecordingBackend:
    SUPPORTS_USER_SWITCHING = True
    SUPPORTS_TASK_IMAGES = True
    config = SimpleNamespace(type="recording")

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


async def test_missing_capability_fails_before_any_command():
    backend = RecordingBackend()
    backend.SUPPORTS_TASK_IMAGES = False
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
