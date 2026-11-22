# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "browser-harness==0.1.13",
#     "openai",
#     "mcp==2.3.0",
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
import fcntl
import json
import locale
import os
import re
import select
import selectors
import shutil
import struct
import subprocess
import sys
import termios
import threading
import time
import urllib.request
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from openai import AsyncOpenAI

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


def _pending(fd: int) -> int:
    """Bytes waiting in a pipe right now."""
    return struct.unpack("i", fcntl.ioctl(fd, termios.FIONREAD, b"\0" * 4))[0]


def _at_end(fd: int) -> bool:
    """Whether every process has closed a pipe whose waiting bytes were just read. Bytes
    that arrived since are a leftover child's, written after the program exited, and
    dropped."""
    poller = select.poll()  # not select.select, which rejects fds >= FD_SETSIZE (1024)
    poller.register(fd, select.POLLIN)
    if not poller.poll(0):
        return False
    return not os.read(fd, 65536)


def _discard_until_eof(fd: int) -> None:
    """Read a pipe a leftover child still holds to its end, so the child's writes neither
    block on a full pipe nor fail on a closed one once the call has returned."""
    with open(fd, "rb", buffering=0) as pipe:
        while pipe.read(65536):
            pass


def _text(output: bytes | bytearray) -> str:
    """One stream's output as text, with its newlines translated on their own."""
    return output.decode(errors="replace").replace("\r\n", "\n").replace("\r", "\n")


HELD_OUTPUT_NOTE = "[a process this code started still holds its output; what it writes later is not shown]"
"""Ends a result whose output a child the program left running still holds open."""


def run_program(args: list[str], stdin: str, env: dict[str, str]) -> str:
    """Run a program with `stdin` as its input and return its stdout then stderr.

    Returns as soon as the program itself exits, with exactly what it wrote by then.
    Children it left running may still hold its pipes; what they write later is not
    included, and the pipes are drained in the background instead of being waited on.
    """
    held_open = False
    # The encoding text=True used for stdin; text it cannot encode fails the call.
    data = memoryview(stdin.encode(locale.getpreferredencoding(False)))
    with subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    ) as process:
        assert process.stdin and process.stdout and process.stderr
        stdout, stderr = bytearray(), bytearray()
        deadline = time.monotonic() + BROWSER_TOOL_TIMEOUT
        try:
            with selectors.DefaultSelector() as selector:
                if data:
                    selector.register(process.stdin, selectors.EVENT_WRITE)
                else:
                    process.stdin.close()
                selector.register(process.stdout, selectors.EVENT_READ, stdout)
                selector.register(process.stderr, selectors.EVENT_READ, stderr)
                try:
                    while selector.get_map():
                        if process.poll() is not None:
                            # The program has exited, so what it wrote is in the pipes
                            # already: read exactly that and return. A pipe still open
                            # is held by a child it left running.
                            if not process.stdin.closed:
                                selector.unregister(process.stdin)
                                process.stdin.close()
                            for key in list(selector.get_map().values()):
                                pending = _pending(key.fd)
                                while pending > 0:
                                    chunk = os.read(key.fd, min(pending, 65536))
                                    if not chunk:
                                        break
                                    pending -= len(chunk)
                                    key.data.extend(chunk)
                                if _at_end(key.fd):
                                    selector.unregister(key.fileobj)
                            held_open = bool(selector.get_map())
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(
                                process.args, BROWSER_TOOL_TIMEOUT
                            )
                        for key, _ in selector.select(min(remaining, 0.1)):
                            if key.fileobj is process.stdin:
                                try:
                                    written = os.write(key.fd, data[: select.PIPE_BUF])
                                except BrokenPipeError:  # it exits without reading all
                                    written = len(data)
                                data = data[written:]
                                if not data:
                                    selector.unregister(process.stdin)
                                    process.stdin.close()
                                continue
                            chunk = os.read(key.fd, 65536)
                            if chunk:
                                key.data.extend(chunk)
                            else:
                                selector.unregister(key.fileobj)
                finally:
                    # A pipe still open belongs to a child that outlives the call. Read
                    # it to its end off the call instead of closing it under the child.
                    for key in list(selector.get_map().values()):
                        if key.fileobj is not process.stdin:
                            threading.Thread(
                                target=_discard_until_eof,
                                args=(os.dup(key.fd),),
                                daemon=True,
                            ).start()
                # Both pipes can close while the program still runs (it redirected or
                # closed its stdout and stderr): let it finish within the deadline.
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    raise subprocess.TimeoutExpired(
                        process.args, BROWSER_TOOL_TIMEOUT
                    ) from None
        finally:
            # Do not communicate() here: leftover children may still hold the pipes.
            if process.returncode is None:
                process.kill()
                process.wait()
    text = _text(stdout) + _text(stderr)
    if held_open:
        text += ("\n" if text and not text.endswith("\n") else "") + HELD_OUTPUT_NOTE
    return text


def run_browser(code: str, env: dict[str, str]) -> str:
    """Run model code through the pinned browser-harness CLI."""
    try:
        return (
            run_program([sys.executable, "-m", "browser_harness.run"], code, env)
            or "(no output)"
        )
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


def read_staged(path: str) -> bytes:
    """Read and delete a file the harness staged for this launch; empty without one."""
    if not path:
        return b""
    file = Path(path)
    data = file.read_bytes()
    file.unlink()
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--browser", default="chromium")
    parser.add_argument("--cdp-url", default="")
    parser.add_argument("--system-prompt-file", default="")
    parser.add_argument("--initial-messages-file", default="")
    parser.add_argument("--mcp-config", default="")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    system_prompt = read_staged(args.system_prompt_file).decode()
    initial = json.loads(read_staged(args.initial_messages_file) or b"[]")
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    endpoint = (
        ensure_chromium(state_dir) if args.browser == "chromium" else args.cdp_url
    )
    browser_env = {
        **os.environ,
        "BH_HOME": str(state_dir / "bh-home"),
        "BH_TELEMETRY": "0",
        "BH_TAB_MARKER": "0",  # Keep task page titles unchanged.
        (
            "BU_CDP_WS" if urlsplit(endpoint).scheme in {"ws", "wss"} else "BU_CDP_URL"
        ): endpoint,
    }
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    config = json.loads(args.mcp_config or "{}")
    tools = [BROWSER_TOOL]
    reserved = {"browser"}
    async with AsyncExitStack() as mcp_stack:
        mcp_tools, dispatch, servers = await connect_mcp(config, mcp_stack, reserved)
        tools += mcp_tools
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        messages.extend(initial)
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
