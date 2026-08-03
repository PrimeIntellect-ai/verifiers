# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "browser-harness==0.1.8",
#     "openai",
#     "mcp>=1.24.0,<2",
#     "httpx",
#     "tenacity",
#     "pydantic>=2.13.4",
# ]
# ///
"""A chat loop whose one local tool drives a real Chromium over CDP.

Each tool call pipes the model's code to browser-harness, whose daemon holds the
CDP connection and pre-imports its page helpers. `chromium` launches and owns a
local browser; `cdp` attaches to an HTTP or WebSocket endpoint it does not own.
Trace-scoped state preserves tabs across resume and records owned process IDs.
"""

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import httpx
from openai import AsyncOpenAI
from pydantic import TypeAdapter, ValidationError
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

MCP_CALL_ATTEMPTS = 6
MCP_TIMEOUT = 600.0

JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, Any])
MESSAGES_ADAPTER = TypeAdapter(list[dict[str, Any]])

BROWSER_TOOL_TIMEOUT = 3600
"""Matches the bash harness's command timeout."""

BROWSER_READY_TIMEOUT = 60
"""The documented Playwright image announced CDP in under one second from a
cold container; one minute leaves ample runtime startup headroom."""

_DEVTOOLS_LINE = re.compile(r"DevTools listening on ws://127\.0\.0\.1:(\d+)/")

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


def find_browser() -> str:
    """The Chromium binary to launch, where a machine keeps one.

    `PLAYWRIGHT_BROWSERS_PATH` before PATH: an image that installs browsers
    through Playwright puts nothing on PATH.
    """
    for key in ("BH_CHROME_PATH", "CHROME_PATH"):
        candidate = os.environ.get(key)
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    registry = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if registry:
        builds = sorted(Path(registry).glob("chromium-*/chrome-linux*/chrome"))
        if builds:
            return str(builds[-1])
    for name in (
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
    ):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit(
        "no Chromium/Chrome found; run on a browser-capable image "
        "(e.g. mcr.microsoft.com/playwright/python) or set BH_CHROME_PATH"
    )


def _endpoint_alive(endpoint: str) -> bool:
    try:
        # browser-harness uses five seconds for this same DevTools HTTP probe.
        urllib.request.urlopen(f"{endpoint}/json/version", timeout=5).close()
        return True
    except OSError:
        return False


def ensure_chromium(state_dir: Path) -> str:
    """Reuse this trace's live Chromium or launch it."""
    endpoint_file = state_dir / "cdp-endpoint"
    if endpoint_file.exists():
        endpoint = endpoint_file.read_text().strip()
        if endpoint and _endpoint_alive(endpoint):
            return endpoint

    profile = state_dir / "profile"
    profile.mkdir(parents=True, exist_ok=True)
    log = state_dir / "browser.log"
    with open(log, "wb") as stderr:
        browser = subprocess.Popen(
            [
                find_browser(),
                "--remote-debugging-port=0",
                f"--user-data-dir={profile}",
                "--headless",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + BROWSER_READY_TIMEOUT
        while time.monotonic() < deadline and browser.poll() is None:
            output = log.read_text(errors="replace")
            if match := _DEVTOOLS_LINE.search(output):
                endpoint = f"http://127.0.0.1:{match.group(1)}"
                endpoint_file.write_text(endpoint)
                return endpoint
            time.sleep(0.1)
        if browser.poll() is None:
            browser.kill()
        browser.wait()
    tail = log.read_text(errors="replace")[-2000:]
    raise SystemExit(
        f"browser did not announce a DevTools port within {BROWSER_READY_TIMEOUT}s: "
        f"{tail or '<no browser output>'}"
    )


def run_browser(code: str, env: dict[str, str]) -> str:
    """Run model code through the pinned browser-harness CLI."""
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
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
):
    completion = await client.chat.completions.create(
        model=model,
        messages=cast(Any, messages),
        tools=cast(Any, tools or None),
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
                headers=spec.get("headers") or None,
                timeout=httpx.Timeout(spec.get("timeout", MCP_TIMEOUT), connect=5.0),
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
        return "\n".join(str(part["text"]) for part in parts)
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
    parser.add_argument("--browser", default="chromium")
    parser.add_argument("--cdp-url", default="")
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
        initial = MESSAGES_ADAPTER.validate_json(payload)
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    endpoint = (
        ensure_chromium(state_dir) if args.browser == "chromium" else args.cdp_url
    )
    browser_env = {
        **os.environ,
        "BH_HOME": str(state_dir / "bh-home"),
        "BH_TELEMETRY": "0",
        (
            "BU_CDP_WS" if urlsplit(endpoint).scheme in {"ws", "wss"} else "BU_CDP_URL"
        ): endpoint,
    }
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    config = JSON_OBJECT_ADAPTER.validate_json(args.mcp_config or "{}")
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
                tool_args = JSON_OBJECT_ADAPTER.validate_json(
                    call.function.arguments or "{}"
                )
            except ValidationError as e:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": (
                            "error: invalid tool arguments "
                            f"({e.errors(include_url=False)[0]['msg']}); "
                            "resend as a JSON object"
                        ),
                    }
                )
                continue
            if name in dispatch:
                content = await call_mcp(servers, dispatch, name, tool_args)
            elif name == "browser":
                content = await asyncio.to_thread(
                    run_browser,
                    tool_args.get("code", ""),
                    browser_env,
                )
            else:
                content = f"error: unknown tool {name!r}"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )


if __name__ == "__main__":
    asyncio.run(main())
