# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp"]
# ///
"""glossary tool server — a single-file uv script (only runtime dep: uv).

Serves a `lookup` tool over a facts corpus passed in via the FACTS_JSON env var, on
streamable HTTP at `/mcp` on the port the harness sets in MCP_PORT.
"""

import json
import os

from mcp.server.fastmcp import FastMCP

FACTS: dict[str, str] = json.loads(os.environ.get("FACTS_JSON", "{}"))
mcp = FastMCP("facts", host="127.0.0.1", port=int(os.environ["MCP_PORT"]))


@mcp.tool()
def lookup(name: str) -> str:
    """Look up what a person or thing is known for."""
    return FACTS.get(name.strip().lower(), "no entry found")


mcp.run(transport="streamable-http")
