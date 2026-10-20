"""fetch (v1, MCP tool): a tool server on a Prime VM of its own, under its own egress.

The integration test places a `FetchToolset` on a Prime runtime of its own
(`tools.runtime`) beside an offline workspace. The server's VM joins the workspace over
the rollout's private network and keeps the egress policy of its own runtime.
"""

import urllib.request

import verifiers.v1 as vf


class FetchToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "fetch"

    @vf.tool
    def fetch(self, url: str) -> str:
        """Fetch a URL; return its HTTP status, or the type of the error."""
        try:
            return str(urllib.request.urlopen(url, timeout=15).status)
        except (OSError, ValueError) as error:  # URLError and timeouts are OSErrors
            return type(error).__name__


class FetchToolTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class FetchToolTask(vf.Task[vf.TaskData, vf.State, FetchToolTaskConfig]):
    @classmethod
    def toolsets(cls, config: FetchToolTaskConfig) -> list[vf.Toolset]:
        return [FetchToolset(config.tools)]


if __name__ == "__main__":
    FetchToolset.run()
