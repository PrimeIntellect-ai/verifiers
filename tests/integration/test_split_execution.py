"""Split harness preflight/config integration and live Codex placement coverage."""

import json
import tomllib
from contextlib import AsyncExitStack

import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.harnesses.codex.harness import CodexHarness, CodexHarnessConfig
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import SubprocessConfig, provision_backend
from verifiers.v1.runtimes.deployment import ExecutionConnection


@pytest.mark.parametrize("entrypoint", ["run", "interaction"])
@pytest.mark.parametrize("harness_id,skills", [("bash", False), ("codex", True)])
async def test_split_preflight_before_rollout(
    monkeypatch, tmp_path, entrypoint, harness_id, skills
):
    """Both public execution paths reject unsupported capabilities before any boot."""

    async def unexpected_open(self):
        pytest.fail("unsupported split configuration reached rollout provisioning")

    monkeypatch.setattr(Rollout, "open", unexpected_open)
    agent = vf.Agent(
        vf.AgentConfig(
            model="unused",
            harness={"id": harness_id},
            runtime={"type": "prime"},
            deployment={"mode": "split"},
        )
    )
    task = vf.Task(vf.TaskData(prompt="unused", skills=[tmp_path] if skills else []))
    message = "custom skills" if skills else "does not support split execution"
    with pytest.raises(ValueError, match=message):
        if entrypoint == "run":
            await agent.run(task)
        else:
            async with agent.interaction(task):
                pytest.fail("unsupported interaction was opened")


@pytest.mark.parametrize("split", [False, True])
async def test_codex_execution_config_uses_workspace(monkeypatch, tmp_path, split):
    """Real runtime IO joins the adapter, execution connection, and ACP config.

    Only package installation is replaced; this test needs no model, npm, or VM.
    """
    import verifiers.v1.runtimes.subprocess as subprocess_runtime

    monkeypatch.setattr(subprocess_runtime, "CACHE_DIR", tmp_path)
    installed = []

    async def installed_worker(self, runtime):
        installed.append(runtime)

    monkeypatch.setattr(CodexHarness, "setup", installed_worker)
    harness = CodexHarness(CodexHarnessConfig(id="codex"))
    data = vf.TaskData(prompt="unused")
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=data),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
    )
    async with AsyncExitStack() as stack:
        brain = await stack.enter_async_context(provision_backend(SubprocessConfig()))
        workspace = await stack.enter_async_context(
            provision_backend(SubprocessConfig())
        )
        home = brain.workdir / "codex-home"
        monkeypatch.setattr(harness, "trace_home", lambda trace: str(home))
        source, target = brain.bind(), workspace.bind(env={"TASK_VALUE": "bound"})
        command = []

        def launch(argv):
            command[:] = ["transport", "--target", "workspace with spaces", *argv]
            return command.copy()

        if split:
            await harness.setup_execution(
                source, target, ExecutionConnection(source, target, launch)
            )
        config = await harness.prepare_acp(
            ModelContext(model="unused", client=EvalClientConfig()),
            trace,
            source,
            "http://model.invalid/v1",
            "test-secret",
            {},
            data,
        )
        settings = json.loads(config.env["CODEX_CONFIG"])
        if split:
            assert installed == [target]
            assert config.cwd == str(workspace.workdir.resolve())
            assert config.cwd != str(brain.workdir.resolve())
            environments = tomllib.loads((home / "environments.toml").read_text())
            assert environments == {
                "default": "workspace",
                "include_local": False,
                "environments": [
                    {"id": "workspace", "program": command[0], "args": command[1:]}
                ],
            }
            assert command[-3:] == ["exec-server", "--listen", "stdio"]
            assert settings["features"]["deferred_executor"] is True
        else:
            assert not installed
            assert config.cwd is None
            assert not (home / "environments.toml").exists()
            assert "deferred_executor" not in settings["features"]
        await harness.cleanup(trace, source)
        assert not home.exists()


@pytest.mark.prime
@pytest.mark.docker
@pytest.mark.codex
@pytest.mark.parametrize("scenario", ["files", "loss"])
async def test_codex_split_execution(run, scenario):
    """A real Prime VM owns the harness; a nested container owns task execution."""
    (episode,) = await run(
        "split-execution-v1",
        harness="codex",
        taskset_overrides={"scenario": scenario},
        runtime={
            "type": "prime",
            "image": "python:3.11-slim-trixie",
            "workdir": "/harness",
            "cpu": 4,
            "memory": 8,
            "disk": 20,
        },
        env={
            "retries": {"max_retries": 0},
            "agent": {
                "deployment": {"mode": "split"},
                "execution": {
                    "type": "docker",
                    "image": "python:3.11-slim-trixie",
                    "workdir": "/workspace",
                    "memory": 4,
                },
                "execution_failure": {"workspace_loss": "zero"},
                "timeout": {"setup": 1200, "rollout": 300, "scoring": 60},
                "retries": {"max_retries": 0},
            },
        },
        max_turns=10,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert not trace.errors
    assert trace.agent.runtime.type == "docker"
    assert trace.agent.harness_runtime.type == "prime"
    assert trace.agent.runtime.id != trace.agent.harness_runtime.id
    if scenario == "files":
        assert trace.ok
        assert trace.reward == 1.0
        assert trace.termination is None
        assert trace.info["workspace_report"]["cwd"] == "/workspace"
    else:
        assert trace.stop_condition == "execution_failed"
        assert trace.reward == 0.0
        assert trace.termination.kind == "target_lost"
        assert trace.termination.cause == "container_exited"
        assert trace.termination.valid_sample
        assert trace.termination.policy == "zero"
        assert trace.rewards["execution_failure"].score == 0.0
        assert "workspace_files" not in trace.rewards
