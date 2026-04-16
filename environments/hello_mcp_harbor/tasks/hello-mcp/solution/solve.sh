#!/bin/bash
# Oracle solution (not used by `prime eval run` — kept so `harbor run -a oracle`
# works against this task too).

set -e

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv run --with mcp python3 <<'EOF'
import asyncio, os
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = os.environ.get("HARBOR_MCP_MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")


async def main():
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_secret", {})
            with open("/app/secret.txt", "w") as f:
                f.write(result.content[0].text)


asyncio.run(main())
EOF
