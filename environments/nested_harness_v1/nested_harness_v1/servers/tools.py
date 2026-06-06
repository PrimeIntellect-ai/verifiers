from mcp.server.fastmcp import FastMCP

mcp = FastMCP("nested-harness")


@mcp.tool()
def call_harness(prompt: str) -> str:
    return prompt.upper()


if __name__ == "__main__":
    mcp.run(transport="stdio")
