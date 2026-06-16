"""deepwiki tool server: a remote (streamable-HTTP) MCP server, declared as a `vf.Toolset`.

It's a `vf.Toolset` with no `@tool` methods: setting `url` on its config makes the framework
connect to the remote directly (no launch, placement ignored), and the harness exposes its
tools as `deepwiki_<tool>`. The remote is DeepWiki (https://mcp.deepwiki.com/mcp), which
answers questions about GitHub repos.
"""

import verifiers.v1 as vf

DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"


class DeepWikiToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "deepwiki"  # a remote server (config.url) — no @tool methods; model sees `deepwiki_<tool>`


if __name__ == "__main__":
    DeepWikiToolset.run()
