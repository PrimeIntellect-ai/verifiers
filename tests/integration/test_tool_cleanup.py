"""A rollout's tool server releases what its setup acquired when the rollout ends, whether
the harness succeeds or fails: the server finishes its own teardown before its process
exits, as one holding sandboxes must. Model-free, on the subprocess runtime."""

import pytest
from cleanup_tool import CleanupTask

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.harnesses.utils.mcp import mcp_client


class CallOnce(vf.Harness):
    SUPPORTS_MCP = True
    NEEDS_CONTAINER = False

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **_):
        async with mcp_client({"url": mcp_urls["cleanup"]}) as client:
            await client.call_tool("ping", {})
        return await runtime.run(["sh", "-c", f"exit {trace.task.data.idx}"], {})


@pytest.mark.parametrize("exit_code", [0, 1], ids=["harness-succeeds", "harness-fails"])
async def test_a_tool_server_releases_its_resources_when_the_rollout_ends(
    tmp_path, monkeypatch, exit_code
):
    marker = tmp_path / "released"
    monkeypatch.setenv("CLEANUP_MARKER", str(marker))
    config = vf.AgentConfig(
        model="unused",
        runtime=vf.SubprocessConfig(),
        timeout=TimeoutConfig(setup=60, finalize=30, scoring=30),
        retries=vf.RetryConfig(max_retries=0),
    )
    async with vf.Agent(config) as agent:
        agent.harness = CallOnce(vf.HarnessConfig())
        trace = await agent.run(CleanupTask(vf.TaskData(idx=exit_code, prompt="p")))
    assert trace.ok is (exit_code == 0)
    assert marker.read_text() == "released"
