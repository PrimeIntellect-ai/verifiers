"""Fixture external harness package for loader regression tests."""

import verifiers.v1 as vf


class ExternalHarnessConfig(vf.HarnessConfig):
    custom_flag: bool = False


class ExternalHarness(vf.Harness[ExternalHarnessConfig]):
    SUPPORTS_MCP = False

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls):
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")


__all__ = ["ExternalHarness"]
