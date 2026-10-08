import asyncio

import verifiers.v1 as vf
from verifiers.v1.agent import TimeoutConfig, _EpisodeAgent
from verifiers.v1.harnesses.null import NullHarnessConfig


def test_with_config_keeps_an_episode_agents_standing():
    """A per-run config (a harness naming a per-run credential, a borrowed box's
    runtime policy) on an env's episode agent: a copy that is still the episode's."""
    completed, gate, warned = [], asyncio.Semaphore(1), set()
    seat = _EpisodeAgent(
        vf.AgentConfig(model="m", runtime={"type": "subprocess"}),
        interception=None,
        name="solver",
        shared_tools={},
        task_cls=vf.Task,
        gate=gate,
        completed=completed,
        on_trace=None,
        on_discard=None,
        warned_resources=warned,
    )
    seat.trainable = False
    harness = NullHarnessConfig(id="null", mcp_header_env={"world": {"X": "VAR"}})
    timeout = TimeoutConfig(rollout=5)

    run = seat.with_config(harness=harness, timeout=timeout, max_turns=3)

    assert type(run) is _EpisodeAgent and run is not seat
    assert run.config.harness == harness and run.harness.config == harness
    assert run.timeout == timeout and run.limits.max_turns == 3
    assert run.ctx.model == "m" and run.runtime_config == seat.runtime_config
    assert (run._name, run.trainable, run._gate) == ("solver", False, gate)
    assert run._completed is completed and run._warned_resources is warned
    assert seat.config.harness.id == "bash" and seat.limits.max_turns is None


async def test_an_acp_session_reports_its_process_lost_between_turns_but_not_its_own_close():
    """A live ACP process whose stream fails while no turn runs (its box was
    deleted) settles `lost()` with why; closing the session never does."""
    from verifiers.v1.acp import ACPConfig, ACPHarnessSession

    class Process:
        def __init__(self, fails):
            self.out, self.fails = asyncio.Queue(), fails
            self.stdout, self.stderr = self.read(), self.read(stderr=True)

        async def read(self, stderr=False):
            if stderr:
                return
            if self.fails:
                raise RuntimeError("the sandbox has been terminated")
            await self.out.get()  # the runner exits once asked to shut down
            yield b""

        async def write(self, data):
            self.out.put_nowait(data)

        async def wait(self):
            return 0

        async def terminate(self):
            pass

        kill = terminate

    class Runtime:
        def __init__(self, fails):
            self.fails = fails

        async def prepare_uv_script(self, *args, **kwargs):
            return ["runner"]

        async def open_process(self, argv, env):
            return Process(self.fails)

    def session(fails):
        config = ACPConfig(env={}, command=["agent"], prompt=None)
        return ACPHarnessSession(
            None, None, None, Runtime(fails), "", "", {}, None, config
        )

    lost = session(fails=True)
    await lost._start()
    error = await asyncio.wait_for(lost.lost(), 1)
    assert "the sandbox has been terminated" in str(error)

    closed = session(fails=False)
    await closed._start()
    await closed.close()
    waiting = asyncio.ensure_future(closed.lost())
    await asyncio.sleep(0.05)
    assert not waiting.done()
    waiting.cancel()
