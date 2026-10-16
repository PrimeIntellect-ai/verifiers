"""A run whose box is not ready within the setup budget (a slow boot or restore) failed
on the platform, not the task: the trace is not ok and carries an `io` SandboxError,
so a caller retries it as an outage instead of keeping a stopped trace as its result."""

import asyncio

from test_agent_resume import NotesTask, ScriptHarness, fake_boxes  # noqa: F401 - fixtures
from test_provider_retry import Provider

import verifiers.v1 as vf
from verifiers.v1.errors import outage_kind


class SlowSetup(NotesTask):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await asyncio.sleep(5)


async def test_a_setup_that_outlasts_its_budget_is_an_io_outage_not_an_ok_stop():
    async with Provider() as provider:
        agent = vf.Agent(
            vf.AgentConfig(
                model="scripted",
                client={
                    "type": "eval",
                    "base_url": provider.url,
                    "api_key_var": "SCRIPTED_KEY",
                },
                runtime={"type": "subprocess"},
                harness={"id": "null"},
                timeout={"setup": 0.2},
            )
        )
        agent.harness = ScriptHarness(agent.harness.config)
        trace = await agent.run(SlowSetup(vf.TaskData(idx=0, prompt="start")))
    assert not trace.ok
    assert trace.last_error is not None and trace.last_error.kind == "io"
    assert outage_kind(trace.last_error) == "io"
