"""Fixture harness providing a browser-control capability."""

import verifiers.v1 as vf


class CapabilityHarnessConfig(vf.HarnessConfig):
    id: str = "capability-harness-v1"


class CapabilityHarness(vf.Harness[CapabilityHarnessConfig]):
    CAPABILITIES = frozenset({vf.HarnessCapability.BROWSER_CONTROL})

    async def launch(
        self,
        ctx: vf.RolloutContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        del ctx, trace, runtime, endpoint, secret, mcp_urls
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")


__all__ = ["CapabilityHarness"]
