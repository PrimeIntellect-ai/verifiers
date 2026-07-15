# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp", "httpx"]
# ///
"""Secrets arrive through argv so local tool subprocesses do not inherit them."""

import argparse
import asyncio
import json
import subprocess
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
from openai import AsyncOpenAI

SERPER_URL = "https://google.serper.dev/search"

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command and return its combined stdout and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."}
            },
            "required": ["command"],
        },
    },
}

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Replace a unique string in a file. old_str must appear exactly once in the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative to cwd or absolute).",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact string to find (must appear exactly once).",
                },
                "new_str": {"type": "string", "description": "Replacement string."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Run a web search via Serper (Google) and return the top organic results as title, "
            "URL, and snippet. Issue focused queries and call it several times to cover different "
            "angles; use the bash tool (e.g. curl) to read a result page in full."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}


def format_results(results, query: str) -> str:
    """Format Serper organic results as title/URL/snippet blocks."""
    sections = []
    for i, result in enumerate(results, 1):
        title = (result.get("title") or "").strip() or "Untitled"
        lines = [f"Result {i}: {title}"]
        link = (result.get("link") or "").strip()
        if link:
            lines.append(f"URL: {link}")
        snippet = (result.get("snippet") or "").strip()
        if snippet:
            lines.append(f"  - {snippet}")
        sections.append("\n".join(lines))
    if not sections:
        return f"No results returned for query: {query}"
    return "\n\n---\n\n".join(sections)


def run_search(query: str, api_key: str, num_results: int = 5) -> str:
    """Serper Google web search -> formatted organic results.

    The key arrives as an argument (handed in by the harness over argv, like the interception
    secret) instead of from `$SERPER_API_KEY`, so the agent's `bash` subprocesses never inherit it.
    The whole call is wrapped so a bad query or malformed payload becomes a tool error rather than
    raising out of the chat loop and killing the rollout."""
    if not api_key:
        return "Error: no Serper API key (SERPER_API_KEY was not set in the eval environment)"
    # num_results comes straight from model tool JSON, so it may be a non-int (e.g. "ten"); coerce
    # defensively — `organic[:num_results]` would otherwise raise on a bad slice.
    try:
        num_results = max(1, int(num_results))
    except (TypeError, ValueError):
        num_results = 5
    try:
        response = httpx.post(
            SERPER_URL,
            json={"q": query},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=45,
        )
        response.raise_for_status()
        organic = response.json().get("organic") or []
        return format_results(organic[:num_results], query)
    except Exception as e:
        return f"search failed ({e}). Try again or rephrase the query."


def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            ["bash", "-c", command], capture_output=True, text=True, timeout=3600
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"error: {e}"


def run_edit(path: str, old_str: str, new_str: str) -> str:
    if not isinstance(path, str) or not path:
        return "error: 'path' is required"
    if not isinstance(old_str, str) or not isinstance(new_str, str):
        return "error: 'old_str' and 'new_str' must be strings"
    if not old_str:
        # '' matches everywhere (''.count('') == 1 on an empty file), so it would insert
        # rather than replace — reject it to keep the "exactly once" contract honest.
        return "error: 'old_str' must be a non-empty string"
    filepath = Path(path)
    if not filepath.is_absolute():
        filepath = Path.cwd() / filepath
    if not filepath.exists():
        return f"error: {path} not found"
    # Reading/writing can fail on a directory, permissions, or non-text content; return the
    # error as a tool result instead of letting it abort the chat loop.
    try:
        content = filepath.read_text()
    except Exception as e:
        return f"error: could not read {path}: {e}"
    count = content.count(old_str)
    if count != 1:
        return f"error: old_str must appear exactly once in {path} (found {count})"
    try:
        filepath.write_text(content.replace(old_str, new_str, 1))
    except Exception as e:
        return f"error: could not write {path}: {e}"
    return f"Edited {path}"


async def chat(
    client: AsyncOpenAI, model: str, messages: list[dict], tools: list[dict]
):
    completion = await client.chat.completions.create(
        model=model, messages=messages, tools=tools or None
    )
    return completion.choices[0].message


async def connect_mcp(stack: AsyncExitStack, config: dict) -> tuple[list[dict], dict]:
    """Connect to each configured MCP server (a streamable-HTTP `url`); return
    (tool schemas, dispatch mapping `<server>_<tool>` -> (session, raw tool name))."""
    from mcp import ClientSession
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    tool_schemas: list[dict] = []
    dispatch: dict[str, tuple] = {}
    for name, spec in config.get("mcpServers", {}).items():
        http_client = await stack.enter_async_context(
            create_mcp_http_client(headers=spec.get("headers") or None)
        )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(spec["url"], http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        for tool in (await session.list_tools()).tools:
            full = f"{name}_{tool.name}"
            tool_schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": full,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )
            dispatch[full] = (session, tool.name)
    return tool_schemas, dispatch


def mcp_content_to_chat_content(blocks) -> str | list[dict]:
    parts = []
    for block in blocks:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.type == "image":
            url = f"data:{block.mimeType};base64,{block.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            parts.append({"type": "text", "text": str(block)})
    if not parts:
        return str(blocks)
    if all(part["type"] == "text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


async def call_mcp(dispatch: dict, name: str, arguments: dict) -> str | list[dict]:
    session, raw = dispatch[name]
    result = await session.call_tool(raw, arguments)
    return mcp_content_to_chat_content(result.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--initial-messages-file", default="")
    parser.add_argument("--mcp-config", default="")
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
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    config = json.loads(args.mcp_config or "{}")
    async with AsyncExitStack() as stack:
        mcp_tools, dispatch = (
            await connect_mcp(stack, config) if config.get("mcpServers") else ([], {})
        )
        tools = [BASH_TOOL]
        if args.edit:
            tools.append(EDIT_TOOL)
        if args.search:
            tools.append(SEARCH_TOOL)
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
        while True:
            message = await chat(client, args.model, messages, tools)
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                break
            for call in message.tool_calls:
                name = call.function.name
                try:
                    tool_args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": f"error: invalid JSON in tool arguments ({e}); resend the call with valid JSON",
                        }
                    )
                    continue
                # Valid JSON can still be a non-object (`[]`, `42`, `null`); the `.get(...)` calls
                # below assume a dict, so reject anything else as a tool error rather than crashing.
                if not isinstance(tool_args, dict):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": f"error: tool arguments must be a JSON object, got {type(tool_args).__name__}; resend as an object",
                        }
                    )
                    continue
                if name in dispatch:
                    content = await call_mcp(dispatch, name, tool_args)
                elif name == "bash":
                    content = await asyncio.to_thread(
                        run_bash, tool_args.get("command", "")
                    )
                elif name == "edit" and args.edit:
                    content = await asyncio.to_thread(
                        run_edit,
                        tool_args.get("path"),
                        tool_args.get("old_str"),
                        tool_args.get("new_str"),
                    )
                elif name == "search" and args.search:
                    content = await asyncio.to_thread(
                        run_search,
                        tool_args.get("query", ""),
                        args.serper_key,
                        tool_args.get("num_results", 5),
                    )
                else:
                    content = f"error: unknown tool {name!r}"
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )


if __name__ == "__main__":
    asyncio.run(main())
