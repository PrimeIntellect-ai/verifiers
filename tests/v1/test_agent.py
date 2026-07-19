"""Agent guards that fire before any model or runtime I/O (no live calls)."""

import asyncio
from unittest.mock import patch

import pytest

import verifiers.v1 as vf
import verifiers.v1.agent
import verifiers.v1.rollout
from verifiers.v1.agent import _check_borrowed_placement
from verifiers.v1.harnesses.bash import BashHarness, BashHarnessConfig
from verifiers.v1.runtimes import DockerConfig, SubprocessConfig, make_runtime


def _agent() -> vf.Agent:
    # The guard under test raises before the model leg is touched, so the
    # client can be a stub.
    return vf.Agent(
        BashHarness(BashHarnessConfig()),
        "test-model",
        None,  # type: ignore[arg-type]
    )


async def test_borrowed_subprocess_box_refuses_task_image():
    """Parity with the provisioning path: `resolve_runtime_config` refuses a task
    `image` on the subprocess runtime, and the borrowed branch never resolves — so
    the borrowed guard must refuse the same pairing."""
    box = make_runtime(SubprocessConfig())
    task = vf.Task(vf.TaskData(idx=0, prompt="hi", image="python:3.12-slim"))
    with pytest.raises(ValueError, match="requires image"):
        await _agent().run(task, runtime=box)


def test_borrowed_container_box_with_other_image_warns():
    """A container box whose image differs from the task's can still be the point of
    borrowing (a judge in a solver's world) — signal, don't refuse."""
    box = make_runtime(DockerConfig(image="ubuntu:24.04"))
    task = vf.Task(vf.TaskData(idx=0, prompt="hi", image="python:3.12-slim"))
    # The `verifiers` package logger doesn't propagate to root, so caplog never sees
    # these records; assert on the module logger itself.
    with patch.object(verifiers.v1.agent.logger, "warning") as warn:
        _check_borrowed_placement(task, box)
    warn.assert_called_once()
    assert "never re-provisioned" in warn.call_args[0][0]


def test_borrowed_matching_box_is_silent():
    box = make_runtime(DockerConfig(image="python:3.12-slim"))
    task = vf.Task(vf.TaskData(idx=0, prompt="hi", image="python:3.12-slim"))
    with patch.object(verifiers.v1.agent.logger, "warning") as warn:
        _check_borrowed_placement(task, box)
    warn.assert_not_called()


async def test_shared_tools_hit_the_mcp_pairing_guard():
    """`shared_tools` puts MCP in play exactly like a task's own tool servers, so a
    non-MCP harness must be refused — the same `validate_pairing` check an eval runs
    at init against the taskset's declared tools. Raises before any runtime or model
    I/O, so a stub server value is safe."""

    class NoMcpHarness(BashHarness):
        SUPPORTS_MCP = False

    agent = vf.Agent(
        NoMcpHarness(BashHarnessConfig()),
        "test-model",
        None,  # type: ignore[arg-type]
    )
    task = vf.Task(vf.TaskData(idx=0, prompt="hi"))
    with pytest.raises(ValueError, match="does not support MCP"):
        await agent.run(task, shared_tools={"search": object()})  # type: ignore[dict-item]


async def test_cancelled_run_frees_the_owned_runtime(monkeypatch):
    """A cancellation mid-setup escapes open() before the driver reaches close();
    the run must still free what it holds — the owned runtime, the entered
    servers — on the way out (`RolloutRun.abort`)."""
    started = asyncio.Event()

    class HangingSetup(vf.Task):
        async def setup(self, trace, runtime):
            started.set()
            await asyncio.sleep(60)

    boxes = []
    real = verifiers.v1.rollout.make_runtime

    def spy(config, **kwargs):
        boxes.append(box := real(config, **kwargs))
        return box

    monkeypatch.setattr(verifiers.v1.rollout, "make_runtime", spy)
    run = asyncio.create_task(
        _agent().run(HangingSetup(vf.TaskData(idx=0, prompt="hi")))
    )
    await asyncio.wait_for(started.wait(), 10)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert boxes and boxes[0].stopped  # the owned box was torn down on the way out
