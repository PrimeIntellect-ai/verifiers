"""The rlm harness maps each typed `RLMHarnessConfig` knob to one `RLM_*` env var it injects
when launching the rlm binary. `summarize_at_tokens` is always injected (empty when `None`) so
the field, not an ambient host var the subprocess runtime inherits, is the source of truth."""

from verifiers.v1.clients import RolloutContext
from verifiers.v1.harnesses.rlm.harness import RLMHarness, RLMHarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import Sampling


class _CaptureRuntime(Runtime):
    """A no-op runtime that records the env `launch` hands to `run_program`."""

    def __init__(self) -> None:
        super().__init__()
        self.launched_env: dict[str, str] = {}

    async def start(self) -> None:  # pragma: no cover - unused
        pass

    async def run(self, argv, env):  # pragma: no cover - unused
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def read(self, path):  # pragma: no cover - unused
        return b""

    async def write(self, path, data):  # pragma: no cover - unused
        pass

    async def run_program(self, argv, env):
        self.launched_env = env
        return ProgramResult(exit_code=0, stdout="", stderr="")


async def _launch_env(config: RLMHarnessConfig) -> dict[str, str]:
    runtime = _CaptureRuntime()
    ctx = RolloutContext(model="gpt-x", client=None, sampling=Sampling())  # type: ignore[arg-type]
    trace = Trace(task=Task(idx=0, prompt="solve it"))
    await RLMHarness(config).launch(
        ctx, trace, runtime, endpoint="http://ep", secret="sk", mcp_urls={}
    )
    return runtime.launched_env


async def test_summarize_none_injects_empty_disabling_value() -> None:
    # `summarize_at_tokens=None` injects "" (which rlm reads as "off"), so it reliably *disables*
    # auto-compaction rather than letting an ambient host value leak through the subprocess
    # runtime and silently re-enable it.
    env = await _launch_env(RLMHarnessConfig(id="rlm"))
    assert env["RLM_SUMMARIZE_AT_TOKENS"] == ""
