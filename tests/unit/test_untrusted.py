"""Running delivered code untrusted during grading, on real boxes: `vf.untrusted` runs it
without the grader's privileges and kills what it leaves behind, and Harbor's shared
grading stops the agent's processes before it reads the reward.

Docker needs a reachable daemon and Prime needs `PRIME_API_KEY`; otherwise those cells
skip."""

import asyncio
import contextlib
import os
import shutil
import subprocess
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path

import pytest
from pydantic import TypeAdapter

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError, TaskError
from verifiers.v1.runtimes import Runtime, RuntimeConfig, provision_runtime
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask

IMAGES = {"docker": "python:3.12-alpine", "prime": "python:3.11-slim"}
"""Busybox on Docker, coreutils on Prime (an image Prime has cached)."""

# The live processes of a uid, as root.
LIVE = (
    "grep -H -E '^(State|Uid):' /proc/[0-9]*/status 2>/dev/null | awk -F: -v uid=\"$1\" "
    '\'{split($1, p, "/"); split($3, v, " ")} $2 == "State" {st[p[3]] = v[1]} '
    '$2 == "Uid" && v[1] == uid && st[p[3]] != "Z" {print p[3]}\''
)


@cache
def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return (
        subprocess.run(["docker", "info"], capture_output=True, check=False).returncode
        == 0
    )


@contextlib.asynccontextmanager
async def box(name: str) -> AsyncIterator[Runtime]:
    if name == "docker" and not docker_available():
        pytest.skip("needs a reachable Docker daemon")
    if name == "prime" and not os.environ.get("PRIME_API_KEY"):
        pytest.skip("needs PRIME_API_KEY")
    data = {"type": name, "image": IMAGES[name]}
    if name == "prime":
        data["labels"] = ["vf-ci"]
    config = TypeAdapter(RuntimeConfig).validate_python(data)
    async with asyncio.timeout(600), provision_runtime(config) as view:
        yield view.with_user("root")


def task(**data) -> vf.Task:
    return vf.Task(data=vf.TaskData(idx=0, prompt=None, **data))


async def sh(view: Runtime, script: str, *args: str) -> str:
    result = await view.run(["sh", "-c", script, "test", *args], {})
    assert result.exit_code == 0, result
    return result.stdout.strip()


async def live(view: Runtime, uid: str) -> list[str]:
    return (await sh(view, LIVE, uid)).split()


# Every process's command line, one per line: slim images have no `ps`.
PROCESSES = (
    "for f in /proc/[0-9]*/cmdline; do tr '\\0' ' ' < $f 2>/dev/null; echo; done"
)


# A daemon that leaves its session and keeps writing a file, as a hack would.
def daemon(path: str) -> list[str]:
    loop = f"while :; do echo 1 > {path}; sleep 0.2; done"
    return ["sh", "-c", f"(setsid sh -c '{loop}' >/dev/null 2>&1 &); exit 0"]


@pytest.mark.parametrize("runtime", ["docker", "prime"])
async def test_delivered_code_runs_as_the_agent_and_dies_with_the_block(runtime):
    async with box(runtime) as root:
        await sh(
            root, "mkdir -p /tests && echo secret > /tests/answer && chmod 755 /tests"
        )
        async with vf.untrusted(
            root, task(agent_user="nobody"), private=["/tests"]
        ) as untrusted:
            assert (await untrusted.run(["id", "-un"], {})).stdout.strip() == "nobody"
            assert (await untrusted.run(["cat", "/tests/answer"], {})).exit_code != 0
            hooks = await untrusted.run(
                ["sh", "-c", 'echo "$BASH_ENV|$PYTHONSTARTUP|$PYTHONNOUSERSITE"'],
                {"BASH_ENV": "/tmp/hook", "PYTHONSTARTUP": "/tmp/hook"},
            )
            assert hooks.stdout.strip() == "||1", hooks
            assert (await untrusted.run(daemon("/tmp/forged"), {})).exit_code == 0
            await untrusted.run_background(["sleep", "1000"], {}, log="/tmp/serve.log")
            # A link the code planted to a file it cannot read does not read.
            await untrusted.run(["ln", "-s", "/tests/answer", "/tmp/out"], {})
            with pytest.raises(SandboxError):
                await untrusted.read("/tmp/out")
            assert await live(root, "65534")
        assert await live(root, "65534") == []
        assert await sh(root, "stat -c '%u %a' /tests") == "0 755"
        assert await sh(root, "cat /tests/answer") == "secret"
        await sh(root, "rm -f /tmp/forged; sleep 1; [ ! -e /tmp/forged ]")


@pytest.mark.parametrize("runtime", ["docker", "prime"])
async def test_a_fresh_user_has_no_account_and_leaves_nothing(runtime):
    async with box(runtime) as root:
        if runtime == "prime":  # its exec looks the user up in the box's passwd
            with pytest.raises(ValueError, match="cannot run as a UID with no account"):
                async with vf.untrusted(root, task(), user="fresh"):
                    pass
            return
        async with vf.untrusted(root, task(), user="fresh") as untrusted:
            uid = (await untrusted.run(["id", "-u"], {})).stdout.strip()
            assert int(uid) >= 20000
            assert (
                await untrusted.run(["sh", "-c", "touch $HOME/x"], {})
            ).exit_code == 0
            await untrusted.run_background(["sleep", "1000"], {}, log="/tmp/fresh.log")
            assert await live(root, uid)
        assert await live(root, uid) == []


@pytest.mark.parametrize("runtime", ["docker", "prime"])
async def test_a_root_agent_loses_only_the_sessions_it_started(runtime):
    async with box(runtime) as root:
        await root.run_background(["sleep", "999"], {}, log="/tmp/grader.log")
        async with vf.untrusted(root, task()) as untrusted:
            assert (await untrusted.run(["id", "-u"], {})).stdout.strip() == "0"
            await untrusted.run_background(["sleep", "1000"], {}, log="/tmp/agent.log")
        alive = await sh(root, PROCESSES)
        assert "sleep 999" in alive and "sleep 1000" not in alive


async def test_the_subprocess_runtime_refuses_it():
    async with provision_runtime(vf.SubprocessConfig()) as view:
        with pytest.raises(ValueError, match="cannot run delivered code untrusted"):
            async with vf.untrusted(view, task(agent_user="nobody")):
                pass


def harbor(tmp_path: Path, test_sh: str) -> HarborTask:
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test.sh").write_text(test_sh)
    return HarborTask(
        HarborData(prompt="Do nothing", task_dir=str(tmp_path), agent_user="nobody")
    )


def trace() -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
    )


@pytest.mark.parametrize("runtime", ["docker", "prime"])
async def test_harbor_shared_grading_refuses_a_reward_the_agent_wrote(
    runtime, tmp_path
):
    async with box(runtime) as root:
        agent = root.with_user("nobody")
        # The agent's service, which test.sh grades, and a process forging the verdict.
        await agent.run_background(["sleep", "1000"], {}, log="/tmp/service.log")
        # test.sh grades the service, so it must still be up.
        honest = harbor(
            tmp_path,
            f'{PROCESSES} | grep -q "^sleep 1000" && echo 1 > /logs/verifier/reward.txt\n',
        )
        await honest.stage_tests(root)
        assert await honest.run_verifier(root, trace()) == {"reward": 1.0}
        assert await live(root, "65534") == []

        forging = harbor(tmp_path, "sleep 1; echo 0 > /logs/verifier/reward.txt\n")
        await forging.stage_tests(root)
        loop = "while :; do echo '{\"reward\": 1}' > /logs/verifier/reward.json; sleep 0.1; done"
        await agent.run_background(["sh", "-c", loop], {}, log="/tmp/forge.log")
        with pytest.raises(TaskError, match="agent's user"):
            await forging.run_verifier(root, trace())
        assert await live(root, "65534") == []
