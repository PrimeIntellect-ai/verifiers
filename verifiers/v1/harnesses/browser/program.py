# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "browser-harness==0.1.8",
#     "openai>=2,<3",
#     "mcp>=1.24.0,<2",
#     "httpx>=0.28,<1",
#     "tenacity>=9,<10",
# ]
# ///
"""A chat loop whose one local tool drives a browser the environment provides.

The tool executes Python through browser-use's `browser-harness`
(https://github.com/browser-use/browser-harness, MIT): each call pipes the
model's code to the harness CLI, which holds one CDP WebSocket to the browser
in a small daemon and pre-imports its page helpers. `browser-harness` is pinned
exactly because its CLI surface is what the tool contract wraps; it pins its
own CDP/websocket dependencies exactly in turn.

`--cdp-url` is the endpoint to attach to; the harness resolves it (an
environment's own browser, or one the harness launched in fallback mode) and
this program only ever attaches. It never launches or discovers a browser, so
there is no process management here -- the launcher and the harness own that.

Secrets arrive through argv so the browser and the model's snippets do not
inherit them from the environment.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

MCP_CALL_ATTEMPTS = 6
MCP_TIMEOUT = httpx.Timeout(600.0, connect=5.0)  # the OpenAI SDK client defaults

BROWSER_TOOL_TIMEOUT = 3600
"""Matches the bash harness's command timeout; helper calls inside the snippet
fail much sooner on their own (the harness IPC read times out in seconds)."""

BROWSER_TOOL = {
    "type": "function",
    "function": {
        "name": "browser",
        "description": (
            "Execute Python code that drives the browser through browser-harness. "
            "Helpers are pre-imported; use print() to see values. Each call runs in "
            "a fresh process: Python variables do not persist between calls, but the "
            "browser (tabs, cookies, page state) does."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute.",
                }
            },
            "required": ["code"],
        },
    },
}


def browser_environment(endpoint: str, state_dir: Path) -> dict[str, str]:
    """The environment every `browser-harness` invocation runs with.

    `BU_CDP_URL` is the documented override that makes the harness attach to
    exactly the endpoint the environment provided instead of discovering a local
    Chrome profile. `BH_HOME` keeps all harness state -- daemon socket, logs,
    screenshots, the agent-editable `agent_helpers.py` workspace -- under this
    trace's state directory, so cleanup can find and remove it. Telemetry is
    disabled through its documented opt-out: the rollout may have no route to the
    telemetry host, and phoning home per tool call is not this program's to
    decide anyway.
    """
    return {
        **os.environ,
        "BU_CDP_URL": endpoint,
        "BH_HOME": str(state_dir / "bh-home"),
        "BH_TELEMETRY": "0",
    }


def run_browser(code: str, env: dict[str, str]) -> str:
    """One browser-harness invocation: the model's code on stdin, exactly the
    CLI's own heredoc contract. The CLI ensures the daemon, pre-imports the
    helpers, and execs the code; stdout and stderr both go back to the model
    because a wrong selector explains itself through its traceback. Uncapped,
    like the bash harness's tool -- the system prompt tells the model to filter
    in Python before printing rather than this imposing a per-harness limit."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "browser_harness.run"],
            input=code,
            capture_output=True,
            text=True,
            timeout=BROWSER_TOOL_TIMEOUT,
            env=env,
            check=False,
        )
        return (result.stdout + result.stderr) or "(no output)"
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"error: {e}"


async def chat(
    client: AsyncOpenAI, model: str, messages: list[dict], tools: list[dict]
):
    completion = await client.chat.completions.create(
        model=model, messages=messages, tools=tools or None
    )
    return completion.choices[0].message


@asynccontextmanager
async def mcp_session(spec: dict):
    """One fresh streamable-HTTP session to an MCP server, opened and closed within the caller's
    task so AnyIO cancellation scopes stay correctly nested. A teardown failure after the body
    completed is swallowed — the result is already in hand, and closing noise must not fail (or
    replay) an already-answered call."""
    from mcp import ClientSession
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    stack = AsyncExitStack()
    try:
        http_client = await stack.enter_async_context(
            create_mcp_http_client(
                headers=spec.get("headers") or None, timeout=MCP_TIMEOUT
            )
        )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(spec["url"], http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        yield session
    finally:
        with suppress(Exception):
            await stack.aclose()


async def with_retry(call):
    """Run one session-scoped operation, retrying transient failures with backoff. A call whose
    response was lost may be replayed — MCP has no idempotency key, so tools should tolerate
    at-least-once delivery (a tool that fails reports through its result, not an exception)."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(MCP_CALL_ATTEMPTS),
        wait=wait_exponential_jitter(initial=0.5, max=30),
        reraise=True,
    ):
        with attempt:
            return await call()


async def connect_mcp(
    config: dict, reserved: set[str]
) -> tuple[list[dict], dict, dict]:
    """Enumerate each configured MCP server's tools (a streamable-HTTP `url`); return (tool schemas,
    dispatch mapping `<server>_<tool>` -> (server name, raw tool name), servers mapping name -> spec).
    No session is held — a stateless-HTTP server is reconnected per call."""
    tool_schemas: list[dict] = []
    dispatch: dict[str, tuple] = {}
    servers: dict[str, dict] = {}
    for name, spec in config.get("mcpServers", {}).items():
        servers[name] = spec

        async def list_tools(spec: dict = spec):
            async with mcp_session(spec) as session:
                return (await session.list_tools()).tools

        for tool in await with_retry(list_tools):
            # A server named "" (TOOL_PREFIX = None) advertises its tools bare.
            full = f"{name}_{tool.name}" if name else tool.name
            if full in reserved or full in dispatch:
                raise ValueError(
                    f"duplicate tool name {full!r}; keep MCP tool names qualified"
                )
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
            dispatch[full] = (name, tool.name)
    return tool_schemas, dispatch, servers


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


async def call_mcp(
    servers: dict, dispatch: dict, name: str, arguments: dict
) -> str | list[dict]:
    """Call a tool on a fresh session per attempt — see `with_retry` for the replay semantics.
    The result is converted outside the retry so a conversion failure fails once."""
    server_name, raw = dispatch[name]

    async def call():
        async with mcp_session(servers[server_name]) as session:
            return await session.call_tool(raw, arguments)

    result = await with_retry(call)
    return mcp_content_to_chat_content(result.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--cdp-url", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--initial-messages-file", default="")
    parser.add_argument("--mcp-config", default="")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    initial = []
    if args.initial_messages_file:
        path = Path(args.initial_messages_file)
        payload = path.read_bytes()
        path.unlink()
        initial = json.loads(payload)
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    tool_env = browser_environment(args.cdp_url, state_dir)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    config = json.loads(args.mcp_config or "{}")
    tools = [BROWSER_TOOL]
    reserved = {"browser"}
    mcp_tools, dispatch, servers = (
        await connect_mcp(config, reserved)
        if config.get("mcpServers")
        else ([], {}, {})
    )
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
                content = await call_mcp(servers, dispatch, name, tool_args)
            elif name == "browser":
                content = await asyncio.to_thread(
                    run_browser, tool_args.get("code", ""), tool_env
                )
            else:
                content = f"error: unknown tool {name!r}"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )


if __name__ == "__main__":
    asyncio.run(main())
