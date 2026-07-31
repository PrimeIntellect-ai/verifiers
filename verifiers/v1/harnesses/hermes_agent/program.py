# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["hermes-agent[acp,mcp]=={version}"]
# ///
"""Start Hermes Agent's native ACP server."""

from acp_adapter.entry import main

if __name__ == "__main__":
    main()
