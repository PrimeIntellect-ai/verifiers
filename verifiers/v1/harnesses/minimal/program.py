# /// script
# requires-python = ">=3.11"
# dependencies = ["openai", "mcp>=1.24.0,<2", "httpx", "tenacity"]
# ///
"""Shared null/bash program; secrets use argv so tool subprocesses do not inherit them."""

import argparse
import asyncio
import json
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

SERPER_URL = "https://google.serper.dev/search"

MCP_CALL_ATTEMPTS = 6
MCP_CANCELLED_CLOSE_GRACE = 1.0  # then re-cancel a hung stateful-session DELETE
MCP_ERROR_PREFIX = "VF_MCP_ERROR="
MCP_TIMEOUT = httpx.Timeout(600.0, connect=5.0)  # the OpenAI SDK client defaults


class MCPTransportFailure(Exception):
    """A lost MCP transport, with enough policy data for the parent harness to classify it."""

    def __init__(
        self,
        server: str,
        operation: str,
        initialized: bool,
        replay_safe: bool,
        cause: BaseException,
    ) -> None:
        leaf = cause
        while nested := getattr(leaf, "exceptions", ()):
            leaf = next(
                (e for e in nested if not isinstance(e, asyncio.CancelledError)),
                nested[0],
            )
        status = getattr(getattr(leaf, "response", None), "status_code", None)
        transient = (
            isinstance(
                leaf,
                (
                    asyncio.CancelledError,
                    BrokenResourceError,
                    ClosedResourceError,
                    EndOfStream,
                    httpx.TransportError,
                    OSError,
                    TimeoutError,
                ),
            )
            or (
                isinstance(leaf, McpError)
                and leaf.error.code == CONNECTION_CLOSED
                and leaf.error.message == "Connection closed"
            )
            or status in (408, 429)
            or (status is not None and status >= 500)
        )
        can_replay = not initialized or replay_safe
        self.session_retry_safe = transient and can_replay
        detail = {
            "cause_type": type(leaf).__name__,
            "delivery": "response_unknown" if initialized else "operation_not_started",
            "operation": operation,
            "phase": operation if initialized else "initialize",
            "replay_safe": can_replay,
            "server": server,
            "session_retry_safe": self.session_retry_safe,
            "status_code": status,
            "transient": transient,
        }
        super().__init__(MCP_ERROR_PREFIX + json.dumps(detail, separators=(",", ":")))


async def _mcp_transport(
    spec: dict, ready: asyncio.Future, stop: asyncio.Event
) -> BaseException | None:
    """Own the SDK transport task group separately so its failure cannot cancel the session owner."""
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    try:
        async with (
            create_mcp_http_client(
                headers=spec.get("headers") or None, timeout=MCP_TIMEOUT
            ) as http_client,
            streamable_http_client(spec["url"], http_client=http_client) as streams,
        ):
            if not ready.done():
                ready.set_result(streams)
            await stop.wait()
    except BaseException as error:  # noqa: BLE001 - preserve transport cancellation
        if not ready.done():
            ready.set_exception(error)
        return error
    return None


@asynccontextmanager
async def mcp_session(server: str, spec: dict, operation: str, replay_safe: bool):
    """Open one session while keeping the transport's AnyIO cancel scope in its owner task."""
    from mcp import ClientSession

    ready = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()
    transport = asyncio.create_task(
        _mcp_transport(spec, ready, stop), name=f"mcp-transport-{server}"
    )
    failure = None
    initialized = False
    operation_completed = False
    outer_cancel = None
    try:
        read, write, *_ = await ready
        async with ClientSession(read, write) as session:
            try:
                await session.initialize()
                initialized = True
                yield session
            except asyncio.CancelledError as error:
                outer_cancel = error
                raise
            operation_completed = True
    except BaseException as error:  # noqa: BLE001 - preserve caller cancellation
        task = asyncio.current_task()
        if outer_cancel is None and isinstance(error, asyncio.CancelledError):
            outer_cancel = error
        if task is not None and task.cancelling():
            failure = outer_cancel or asyncio.CancelledError()
        elif not operation_completed:
            failure = error
    finally:
        stop.set()
        task = asyncio.current_task()
        aborting = (
            isinstance(failure, asyncio.CancelledError)
            and task is not None
            and task.cancelling()
        )
        if aborting:
            transport.cancel()
        while not transport.done():
            cancel_count = task.cancelling() if task is not None else 0
            try:
                if aborting:
                    await asyncio.wait_for(
                        asyncio.shield(transport), MCP_CANCELLED_CLOSE_GRACE
                    )
                else:
                    await asyncio.shield(transport)
            except TimeoutError:
                transport.cancel()
            except asyncio.CancelledError as error:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > cancel_count:
                    failure = error
                    aborting = True
                    transport.cancel()
        try:
            transport_failure = transport.result()
        except asyncio.CancelledError as error:
            transport_failure = error
        failure_leaf = failure
        while nested := getattr(failure_leaf, "exceptions", ()):
            failure_leaf = next(
                (e for e in nested if not isinstance(e, asyncio.CancelledError)),
                nested[0],
            )
        connection_closed = (
            isinstance(failure_leaf, McpError)
            and failure_leaf.error.code == CONNECTION_CLOSED
            and failure_leaf.error.message == "Connection closed"
        )
        if transport_failure is None and (
            connection_closed
            or isinstance(
                failure_leaf,
                (
                    BrokenResourceError,
                    ClosedResourceError,
                    EndOfStream,
                    httpx.TransportError,
                    OSError,
                    TimeoutError,
                ),
            )
        ):
            transport_failure = failure

    if failure is None:
        return
    task = asyncio.current_task()
    if isinstance(failure, asyncio.CancelledError) and (
        task is None or task.cancelling() or transport_failure is None
    ):
        raise failure
    if transport_failure is None:
        raise failure
    raise MCPTransportFailure(
        server, operation, initialized, replay_safe, transport_failure
    ) from transport_failure


async def with_retry(call):
    """Retry only transient failures before a replay-safe operation has completed."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(MCP_CALL_ATTEMPTS),
        wait=wait_exponential_jitter(initial=0.5, max=30),
        retry=retry_if_exception(
            lambda error: (
                isinstance(error, MCPTransportFailure) and error.session_retry_safe
            )
        ),
        reraise=True,
    ):
        with attempt:
            return await call()


async def connect_mcp(
    config: dict, reserved: set[str] | None = None
) -> tuple[list[dict], dict, dict]:
    """Enumerate MCP tools and build advertised-name dispatch metadata."""
    reserved = reserved or set()
    tool_schemas: list[dict] = []
    dispatch: dict[str, tuple] = {}
    servers: dict[str, dict] = {}
    for name, spec in config.get("mcpServers", {}).items():
        servers[name] = spec

        async def list_tools(name: str = name, spec: dict = spec):
            async with mcp_session(
                name, spec, "list_tools", replay_safe=True
            ) as session:
                return (await session.list_tools()).tools

        for tool in await with_retry(list_tools):
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
    """Call once after initialization; a lost response is never replayed in this session."""
    server, raw = dispatch[name]

    async def call():
        async with mcp_session(
            server, servers[server], "call_tool", replay_safe=False
        ) as session:
            return await session.call_tool(raw, arguments)

    result = await with_retry(call)
    return mcp_content_to_chat_content(result.content)


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
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"search failed ({e}). Try again or rephrase the query."


def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        return result.stdout + result.stderr
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
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
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"error: could not read {path}: {e}"
    count = content.count(old_str)
    if count != 1:
        return f"error: old_str must appear exactly once in {path} (found {count})"
    try:
        filepath.write_text(content.replace(old_str, new_str, 1))
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"error: could not write {path}: {e}"
    return f"Edited {path}"


async def chat(
    client: AsyncOpenAI, model: str, messages: list[dict], tools: list[dict]
):
    completion = await client.chat.completions.create(
        model=model, messages=messages, tools=tools or None
    )
    return completion.choices[0].message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--initial-messages-file", default="")
    parser.add_argument("--mcp-config", default="")
    parser.add_argument("--bash", action="store_true")
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
        # Null keeps its historical enumeration bound; bash leaves it unset.
        async with asyncio.timeout(None if args.bash else 60):
            mcp_tools, dispatch, servers = await connect_mcp(config, reserved)
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
            elif name == "bash" and args.bash:
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
