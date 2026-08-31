# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp==2.0.0", "httpx", "httpx2", "tenacity"]
# ///
"""Shared Null/Bash chat program; secrets use argv so tools do not inherit them."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from openai import AsyncOpenAI

if TYPE_CHECKING:
    # The harness bundles these modules into the generated script before execution.
    from verifiers.v1.harnesses.utils.compaction import (  # noqa: TC004
        Compactor,
        discover_threshold,
    )
    from verifiers.v1.harnesses.utils.core import (  # noqa: TC004
        BASH_TOOL,
        EDIT_TOOL,
        SEARCH_TOOL,
        run_chat_loop,
    )
    from verifiers.v1.harnesses.utils.mcp import connect_mcp  # noqa: TC004


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--initial-messages-file", default="")
    parser.add_argument("--mcp-config", default="")
    parser.add_argument("--tool-interception-url", default="")
    parser.add_argument("--bash", action="store_true")
    parser.add_argument("--compaction", action="store_true")
    parser.add_argument("--summarize-at-tokens", type=int)
    parser.add_argument("--edit", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--serper-key", default="")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    initial = []
    if args.initial_messages_file:
        path = Path(args.initial_messages_file)
        payload = path.read_bytes()
        path.unlink()
        initial = json.loads(payload)
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=httpx.Timeout(600.0 if args.bash else None, connect=5.0),
    )
    tool_client = (
        httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0))
        if args.tool_interception_url
        else None
    )
    config = json.loads(args.mcp_config or "{}")
    tools = [BASH_TOOL] if args.bash else []
    reserved = {"bash"} if args.bash else set()
    if args.edit:
        tools.append(EDIT_TOOL)
        reserved.add("edit")
    if args.search:
        tools.append(SEARCH_TOOL)
        reserved.add("search")
    if config.get("mcpServers"):
        mcp_tools, dispatch, servers = await asyncio.wait_for(
            connect_mcp(config, reserved), timeout=None if args.bash else 60
        )
    else:
        mcp_tools, dispatch, servers = [], {}, {}
    tools += mcp_tools
    messages = (
        [{"role": "system", "content": args.system_prompt}]
        if args.system_prompt
        else []
    )
    if initial:
        messages.extend(initial)
    elif args.prompt:
        messages.append({"role": "user", "content": args.prompt})
    compactor = Compactor(
        client,
        args.model,
        tools,
        args.compaction,
        args.summarize_at_tokens,
    )
    if compactor.enabled and compactor.threshold is None:
        compactor.threshold = await discover_threshold(client, args.model)
    # The initial conversation is the floor for checkpoint fallbacks: a first-turn
    # checkpoint must never retry from an empty base.
    compactor.note_good(messages)
    await run_chat_loop(args, compactor, messages, dispatch, servers, tool_client)
    if tool_client is not None:
        await tool_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
