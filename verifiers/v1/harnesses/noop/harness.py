from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


class NoopHarnessConfig(HarnessConfig):
    pass


class NoopHarness(Harness[NoopHarnessConfig]):
    """A seat that runs no program and never calls the model.

    For rollouts whose work happens entirely in task hooks — setup, finalize,
    scoring — with no agent phase at all: e.g. the harbor env's verifier seat,
    where grading is the task's own `test.sh`. `launch` returns a synthetic
    success, which the harness contract allows; the trace records what the task
    did, with zero turns."""

    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False
    SUPPORTS_RESUME = False
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        return ProgramResult(exit_code=0, stdout="", stderr="")
