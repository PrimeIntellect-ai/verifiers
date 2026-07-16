"""NeMo Gym resources-server tasks driven by Verifiers harnesses.

NeMo Gym remains authoritative for per-rollout state, tools, and verification. The
resources server is external because Gym does not expose a stable resource-only
launcher; Verifiers owns the model loop, trace, runtime, and harness.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import httpx
from pydantic import Field

from verifiers.v1.decorators import reward
from verifiers.v1.dialects.responses import ResponsesDialect, messages_to_wire
from verifiers.v1.mcp import SharedToolsetConfig, Toolset
from verifiers.v1.state import State
from verifiers.v1.task import Task, TaskConfig, TaskData
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, ToolMessage, content_text

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult, Tool as MCPTool


class NeMoGymTaskConfig(TaskConfig):
    resources_url: str = "http://127.0.0.1:8000"
    """Base URL of an already-running NeMo Gym resources server."""

    headers: dict[str, str] = Field(default_factory=dict)
    """Headers added to seed, direct-tool, MCP, and verification requests."""

    request_timeout: float = Field(60.0, gt=0)


class NeMoGymConfig(TasksetConfig):
    dataset_path: Path
    """JSONL rows containing ``responses_create_params`` and verifier metadata."""

    task: NeMoGymTaskConfig = NeMoGymTaskConfig()


class NeMoGymData(TaskData):
    row: dict[str, Any]
    """The exact source row sent back to ``/seed_session`` and ``/verify``."""


class NeMoGymState(State):
    resources_url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    request_timeout: float = 60.0
    cookies: dict[str, str] = Field(default_factory=dict)
    mcp_url: str | None = None
    mcp_headers: dict[str, str] = Field(default_factory=dict)
    direct_tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)


async def _post(state: NeMoGymState, path: str, body: dict[str, Any]) -> httpx.Response:
    """POST to Gym while carrying the rollout's cookie session forward."""
    async with httpx.AsyncClient(
        headers=state.headers,
        cookies=state.cookies,
        timeout=state.request_timeout,
    ) as client:
        response = await client.post(f"{state.resources_url}/{path}", json=body)
    state.cookies.update(response.cookies)
    return response


class _NeMoGymToolset(Toolset[SharedToolsetConfig, NeMoGymState]):
    """Expose one rollout's Gym tools through the standard V1 MCP boundary.

    Gym's MCP endpoint is stateless HTTP; its signed seed header restores the hidden
    rollout session on every request.
    """

    TOOL_PREFIX = None

    def _register(self, mcp: FastMCP) -> None:
        server = mcp._mcp_server
        server.list_tools()(self._with_state(self.list_tools))
        server.call_tool(validate_input=False)(self._with_state(self.call_tool))

    @asynccontextmanager
    async def _upstream(self) -> AsyncIterator[ClientSession]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        url = self.state.mcp_url
        if url is None:
            raise RuntimeError("NeMo Gym did not provide an MCP URL")
        async with (
            httpx.AsyncClient(
                headers=self.state.mcp_headers,
                timeout=self.state.request_timeout,
            ) as client,
            streamable_http_client(url, http_client=client) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session

    async def list_tools(self) -> list[MCPTool]:
        from mcp.types import Tool

        if self.state.mcp_url is not None:
            async with self._upstream() as session:
                tools = (await session.list_tools()).tools
        else:
            tools = [
                Tool(
                    name=spec["name"],
                    description=spec.get("description") or None,
                    inputSchema=spec.get("parameters") or {},
                )
                for spec in self.state.direct_tools
                if spec.get("type") == "function"
            ]
        self.state.tool_names = [tool.name for tool in tools]
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        from mcp.types import CallToolResult, TextContent

        if self.state.mcp_url is not None:
            async with self._upstream() as session:
                return await session.call_tool(name, arguments)

        response = await _post(self.state, name, arguments)
        return CallToolResult(
            content=[TextContent(type="text", text=response.text)],
            isError=not response.is_success,
        )


def _trace_to_nemo_response(
    trace: Trace,
    responses_create_params: dict[str, Any],
    tool_names: list[str],
) -> dict[str, Any]:
    """Convert the one completed V1 branch into a Gym Responses object."""

    if trace.num_branches != 1:
        raise ValueError(
            f"NeMo Gym scoring requires exactly one trace branch, got {trace.num_branches}"
        )

    known_names = set(tool_names) | {
        spec["name"]
        for spec in responses_create_params.get("tools") or []
        if spec.get("type") == "function" and isinstance(spec.get("name"), str)
    }
    response_item_types = {"reasoning", "message", "function_call"}
    output: list[dict[str, Any]] = []
    started = False

    for node in trace.nodes:
        message = node.message
        if isinstance(message, AssistantMessage) and node.sampled:
            started = True
            if message.provider_state and not all(
                isinstance(item, dict) and item.get("type") in response_item_types
                for item in message.provider_state
            ):
                message = message.model_copy(update={"provider_state": None})
            if message.reasoning_content and not message.provider_state:
                output.append(
                    {
                        "id": f"rs_{trace.id}_{len(output)}",
                        "type": "reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": message.reasoning_content,
                            }
                        ],
                    }
                )
            output.extend(dict(item) for item in messages_to_wire([message]))
        elif started and isinstance(message, ToolMessage):
            output.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": content_text(message.content),
                }
            )

    for item in output:
        name = str(item.get("name", ""))
        bare_name = name.removeprefix("_")
        if item.get("type") == "function_call" and bare_name in known_names:
            item["name"] = bare_name

    return {
        "id": f"resp_{trace.id}",
        "created_at": trace.nodes[-1].timestamp,
        "model": str(responses_create_params.get("model") or "verifiers"),
        "object": "response",
        "output": output,
        "parallel_tool_calls": responses_create_params.get("parallel_tool_calls", True),
        "tool_choice": responses_create_params.get("tool_choice", "auto"),
        "tools": responses_create_params.get("tools") or [],
        "status": "completed",
    }


class NeMoGymTask(Task[NeMoGymData, NeMoGymState, NeMoGymTaskConfig]):
    async def setup(self, trace: Trace) -> None:
        state = trace.state
        state.resources_url = self.config.resources_url.rstrip("/")
        state.headers = dict(self.config.headers)
        state.request_timeout = self.config.request_timeout
        state.direct_tools = self.data.row["responses_create_params"].get("tools") or []
        state.cookies.clear()

        response = await _post(state, "seed_session", self.data.row)
        response.raise_for_status()
        metadata = response.json().get("mcp")
        if metadata is None:
            return
        if metadata.get("transport") != "http":
            raise ValueError(
                f"unsupported NeMo Gym MCP transport: {metadata.get('transport')!r}"
            )
        state.mcp_url = urljoin(
            f"{state.resources_url}/", metadata.get("url_path", "/mcp")
        )
        state.mcp_headers = state.headers | (metadata.get("headers") or {})

    @reward(weight=1.0)
    async def nemo_gym(self, trace: Trace) -> float:
        state = trace.state
        response_body = _trace_to_nemo_response(
            trace,
            self.data.row["responses_create_params"],
            state.tool_names,
        )
        response = await _post(
            state, "verify", {**self.data.row, "response": response_body}
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise TypeError("NeMo Gym /verify response must be an object")

        details = {
            key: value
            for key, value in result.items()
            if key not in {"responses_create_params", "response", "reward"}
        }
        trace.info["nemo_gym"] = details
        trace.record_metrics(
            {
                key: float(value)
                for key, value in details.items()
                if isinstance(value, (bool, int, float))
            }
        )
        return float(result["reward"])


class NeMoGymTaskset(Taskset[NeMoGymTask, NeMoGymConfig]):
    tools = (_NeMoGymToolset,)

    def load(self) -> Iterator[NeMoGymTask]:
        path = self.config.dataset_path.expanduser().resolve()
        dialect = ResponsesDialect()
        count = 0

        with path.open(encoding="utf-8") as dataset:
            for count, line in enumerate(filter(str.strip, dataset), start=1):
                row = json.loads(line)
                prompt, _ = dialect.parse_request(row["responses_create_params"])
                yield NeMoGymTask(
                    NeMoGymData(
                        idx=count - 1,
                        name=f"{path.stem}:{count - 1}",
                        prompt=prompt,
                        row=row,
                    ),
                    self.config.task,
                )

        if not count:
            raise ValueError(f"NeMo Gym dataset is empty: {path}")


if __name__ == "__main__":
    _NeMoGymToolset.run()
