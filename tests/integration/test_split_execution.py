"""Split deployment: the harness on a Prime VM of its own, its tools in the workspace,
for every harness that declares `SUPPORTS_SPLIT_EXECUTION`."""

import pytest
from grid import grid

import verifiers.v1 as vf
from verifiers.v1.harnesses.utils import prebuilt
from verifiers.v1.rollout import Rollout
from verifiers.v1.utils.loaders import harness_class

SPLIT = {
    "deployment": {"mode": "split"},
    "timeout": {"setup": 900, "rollout": 300, "scoring": 60},
}


@pytest.mark.parametrize("entrypoint", ["run", "interaction"])
@pytest.mark.parametrize("harness_id,skills", [("bash", False), ("codex", True)])
async def test_split_preflight_before_rollout(
    monkeypatch, tmp_path, entrypoint, harness_id, skills
):
    """Both public execution paths reject an unsupported harness or capability before
    any boot."""

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
    message = "custom skills" if skills else "does not support split deployment"
    with pytest.raises(ValueError, match=message):
        if entrypoint == "run":
            await agent.run(task)
        else:
            async with agent.interaction(task):
                pytest.fail("unsupported interaction was opened")


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("shell", "split")), indirect=True
)
async def test_split_execution(run, harness, harness_runtime):
    """The agent loop runs on the harness's VM; its tools act on the workspace's files,
    working directory and environment over the rollout's private network. A harness
    with a prebuilt install runs from it on both VMs instead of installing."""
    install = harness_class(harness.id)(harness).prebuilt()
    if install is not None:
        await prebuilt.build(install)
    (episode,) = await run(
        "split-execution-v1",
        harness=harness,
        runtime={**harness_runtime, "workdir": "/workspace"},
        env={"agent": SPLIT},
        max_turns=10,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert not trace.errors
    assert trace.agent.runtime.type == trace.agent.harness_runtime.type == "prime"
    assert trace.agent.runtime.id != trace.agent.harness_runtime.id
    if install is not None:
        assert install.mount in trace.agent.runtime.mounts
        assert install.mount in trace.agent.harness_runtime.mounts
    assert trace.reward == 1.0, trace.info.get("workspace_report")


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("shell", "split")), indirect=True
)
async def test_split_workspace_loss(run, harness, harness_runtime):
    """A workspace the agent powers off ends the rollout as a valid zero sample."""
    (episode,) = await run(
        "split-execution-v1",
        harness=harness,
        taskset_overrides={"scenario": "loss"},
        runtime={**harness_runtime, "workdir": "/workspace"},
        env={"agent": {**SPLIT, "scoring": {"skip_on_workspace_loss": True}}},
        max_turns=10,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert trace.ok
    assert trace.stop_condition == "workspace_lost"
    assert trace.info["scoring_skipped"] == "workspace_lost"
    assert trace.info["workspace_loss"]["status"] == "TERMINATED"
    assert not trace.rewards and trace.reward == 0.0
