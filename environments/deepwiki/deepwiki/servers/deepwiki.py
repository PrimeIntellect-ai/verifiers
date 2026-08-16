import verifiers.v1 as vf

DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"


class DeepWikiToolset(vf.Toolset[vf.ToolsetConfig]):
    # A remote server (config.url) with no locally registered @tool methods.
    TOOL_PREFIX = "deepwiki"


if __name__ == "__main__":
    DeepWikiToolset.run()
