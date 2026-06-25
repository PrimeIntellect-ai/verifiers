"""A verifier-only harness that does not call a model or mutate the runtime."""

from verifiers.v1.clients import RolloutContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace


class NoopHarnessConfig(HarnessConfig):
    """The no-op harness."""


class NoopHarness(Harness[NoopHarnessConfig]):
    SUPPORTS_MCP = False

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        return ProgramResult(exit_code=0, stdout="", stderr="")
