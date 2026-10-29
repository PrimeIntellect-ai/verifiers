"""Split deployment: the harness on a Prime VM of its own, its tools in the workspace."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.rollout import Rollout


@pytest.mark.parametrize("entrypoint", ["run", "interaction"])
async def test_split_preflight_before_rollout(monkeypatch, entrypoint):
    """Both public execution paths reject an unsupported harness before any boot."""

    async def unexpected_open(self):
        pytest.fail("unsupported split configuration reached rollout provisioning")

    monkeypatch.setattr(Rollout, "open", unexpected_open)
    agent = vf.Agent(
        vf.AgentConfig(
            model="unused",
            harness={"id": "bash"},
            runtime={"type": "prime"},
            deployment={"mode": "split"},
        )
    )
    task = vf.Task(vf.TaskData(prompt="unused"))
    with pytest.raises(ValueError, match="does not support split deployment"):
        if entrypoint == "run":
            await agent.run(task)
        else:
            async with agent.interaction(task):
                pytest.fail("unsupported interaction was opened")


@pytest.mark.prime
@pytest.mark.rlm
async def test_rlm_split_execution(run):
    """RLM's loop runs on the harness's VM; its tools act on the workspace's files,
    working directory and environment over the rollout's private network."""
    (episode,) = await run(
        "split-execution-v1",
        harness="rlm",
        runtime={"type": "prime", "workdir": "/workspace"},
        env={
            "agent": {
                "deployment": {"mode": "split"},
                "timeout": {"setup": 900, "rollout": 300, "scoring": 60},
            },
        },
        max_turns=10,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert not trace.errors
    assert trace.agent.runtime.type == trace.agent.harness_runtime.type == "prime"
    assert trace.agent.runtime.id != trace.agent.harness_runtime.id
    assert trace.reward == 1.0, trace.info.get("workspace_report")
