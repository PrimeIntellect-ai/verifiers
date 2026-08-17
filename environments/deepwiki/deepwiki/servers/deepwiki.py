import verifiers.v1 as vf

DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"


class DeepWikiToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "deepwiki"  # a remote server (config.url) — no @tool methods; model sees `deepwiki_<tool>`


if __name__ == "__main__":
    DeepWikiToolset.run()
