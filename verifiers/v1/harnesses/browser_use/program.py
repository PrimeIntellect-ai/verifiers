# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "browser-harness==0.1.8",
#     "openai",
#     "mcp==2.0.0",
#     "httpx",
#     "httpx2",
#     "tenacity",
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
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from openai import AsyncOpenAI

# {tool_interception}

if TYPE_CHECKING:
    # The harness bundles this module into the generated script before execution.
    from verifiers.v1.harnesses.utils.mcp import call_mcp, connect_mcp  # noqa: TC004

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
    parser.add_argument("--tool-interception-url", default="")
    parser.add_argument("--tool-interception-secret-bytes", type=int, default=0)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    tool_secret = read_tool_secret(  # noqa: F821 - injected runtime client
        args.tool_interception_secret_bytes, "Browser"
    )
    tool_interceptor = (
        ToolInterceptionClient(args.tool_interception_url, tool_secret)  # noqa: F821 - injected runtime client
        if args.tool_interception_url
        else None
    )
    initial = []
    if args.initial_messages_file:
        path = Path(args.initial_messages_file)
        payload = path.read_bytes()
        path.unlink()
        initial = json.loads(payload)
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
            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": "",
                "name": name,
            }
            if tool_interceptor is not None:
                decision = await asyncio.to_thread(
                    tool_interceptor.call, "before", tool_message
                )
                if decision["action"] == "rewrite":
                    messages.append(decision["message"])
                    continue
            try:
                tool_args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                content = f"error: invalid JSON in tool arguments ({e}); resend the call with valid JSON"
            # Valid JSON can still be a non-object (`[]`, `42`, `null`); the `.get(...)` calls
            # below assume a dict, so reject anything else as a tool error rather than crashing.
            else:
                if not isinstance(tool_args, dict):
                    content = f"error: tool arguments must be a JSON object, got {type(tool_args).__name__}; resend as an object"
                elif name in dispatch:
                    content = await call_mcp(servers, dispatch, name, tool_args)
                elif name == "browser":
                    content = await asyncio.to_thread(
                        run_browser,
                        tool_args.get("code", ""),
                        browser_env,
                    )
                else:
                    content = f"error: unknown tool {name!r}"
            tool_message["content"] = content
            if tool_interceptor is not None:
                decision = await asyncio.to_thread(
                    tool_interceptor.call, "after", tool_message
                )
                if decision["action"] == "rewrite":
                    tool_message = decision["message"]
            messages.append(tool_message)
    if tool_interceptor is not None:
        tool_interceptor.close()


if __name__ == "__main__":
    asyncio.run(main())
