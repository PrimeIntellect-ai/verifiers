"""Split deployment: the harness on a Prime VM of its own, its tools in the workspace."""

import os
import shlex

import pytest
from conftest import scripted_chat_reply, serve_scripted_model
from grid import HARNESSES, grid, runtimes_grid

import verifiers.v1 as vf
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.tasksets.harbor.toolset import HarborMCPAdapter, HarborMCPConfig
from verifiers.v1.utils.prime import load_prime_config

TIMEOUTS = {"setup": 900, "rollout": 300, "scoring": 60}


@pytest.mark.parametrize(
    "harness_runtime", runtimes_grid(("subprocess", "prime")), indirect=True
)
async def test_harbor_mcp_runs_in_borrowed_workspace(harness_runtime):
    if harness_runtime["type"] == "prime":
        if not (os.environ.get("PRIME_API_KEY") or load_prime_config().get("api_key")):
            pytest.skip("Prime sandbox credentials unavailable")
        harness_runtime = {**harness_runtime, "labels": ["vf-ci"]}

    class HarborTask(vf.Task):
        @classmethod
        def toolsets(cls, config):
            return [
                HarborMCPAdapter(
                    HarborMCPConfig(
                        server={
                            "name": "task",
                            "transport": "stdio",
                            "command": "sh",
                            "args": ["serve-task.sh"],
                        }
                    )
                )
            ]

        async def setup(self, runtime):
            await runtime.write("input.txt", b"workspace input")
            command = await runtime.prepare_uv_script(
                """# /// script
# dependencies = ["mcp==2.2.0"]
# ///
import os
from pathlib import Path
from mcp.server.mcpserver import MCPServer
Path("worker.pid").write_text(str(os.getpid()))
server = MCPServer("workspace")
@server.tool()
def copy_input() -> str:
    assert not {"VF_HARNESS_ONLY", "TOOL_SERVICE_MCP_SECRET", "VF_STATE_SECRET"} & os.environ.keys()
    text = Path("input.txt").read_text()
    Path("output.txt").write_text(text)
    return text
server.run()
""",
                activate=False,
            )
            await runtime.write(
                "serve-task.sh", ("exec " + shlex.join(command)).encode()
            )

    def answer(path, body):
        if body["messages"][-1]["role"] == "tool":
            reply = scripted_chat_reply(body["model"], content="done")
        else:
            assert [tool["function"]["name"] for tool in body["tools"]] == [
                "task_copy_input"
            ]
            reply = scripted_chat_reply(
                body["model"], tool="task_copy_input", call_id="copy"
            )
        return ChatDialect(), reply

    async with serve_scripted_model(answer) as client:
        agent = vf.Agent(
            vf.AgentConfig(
                model="scripted",
                harness={"id": "null", "env": {"VF_HARNESS_ONLY": "private"}},
                runtime=harness_runtime,
                client={"type": "eval", **client},
            )
        )
        task = HarborTask(vf.TaskData(prompt="Copy the input."))
        async with agent.provision(task) as runtime:
            trace = await agent.run(task, runtime=runtime)
            assert trace.ok, trace.errors
            assert trace.agent.runtime.id != trace.agent.harness_runtime.id
            assert await runtime.read("output.txt") == b"workspace input"
            pid = (await runtime.read("worker.pid")).decode()
            assert (await runtime.run(["kill", "-0", pid], {})).exit_code != 0


@pytest.mark.parametrize(
    "harness,harness_runtime",
    grid(
        [*HARNESSES, {"id": "kimi-code", "dialect": "responses"}],
        needs=("shell",),
    ),
    indirect=True,
)
async def test_split_execution(run, harness, harness_runtime):
    """The agent loop runs on the harness's VM; its tools act on the workspace's files,
    working directory and environment through the evaluator relay."""
    runtime = {**harness_runtime, "workdir": "/workspace"}
    if harness.id in {"claude-code", "codex", "pi"}:
        # Exercise the worker adapters on an image that ships neither Python nor tar.
        runtime["image"] = "rockylinux/rockylinux:9-minimal"
    (episode,) = await run(
        "split-execution-v1",
        harness=harness,
        runtime=runtime,
        env={"agent": {"timeout": TIMEOUTS}},
        max_turns=10,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert not trace.errors
    assert trace.agent.runtime.type == trace.agent.harness_runtime.type == "prime"
    assert trace.agent.runtime.id != trace.agent.harness_runtime.id
    assert trace.reward == 1.0, trace.info.get("workspace_report")


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("shell",)), indirect=True
)
async def test_split_workspace_loss(run, harness, harness_runtime):
    """A workspace the agent powers off ends the rollout as a valid zero sample."""
    (episode,) = await run(
        "split-execution-v1",
        harness=harness,
        taskset_overrides={"scenario": "loss"},
        runtime={**harness_runtime, "workdir": "/workspace"},
        env={"agent": {"timeout": TIMEOUTS}},
        max_turns=10,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert trace.ok
    assert trace.stop_condition == "workspace_lost"
    assert trace.info["scoring_skipped"] == "workspace_lost"
    assert trace.info["workspace_loss"]["status"] == "TERMINATED"
    assert not trace.rewards and trace.reward == 0.0
