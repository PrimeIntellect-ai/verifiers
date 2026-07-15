"""NeMo Gym resources-server tasks driven by Verifiers harnesses.

NeMo Gym remains authoritative for per-rollout state, tools, and verification. The
resources server is external because Gym does not expose a stable resource-only
launcher; Verifiers owns the model loop, trace, runtime, and harness.
"""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import httpx
from pydantic import Field

from verifiers.v1.decorators import reward
from verifiers.v1.dialects.responses import ResponsesDialect
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
    dataset_path: Path | None = None
    """JSONL rows containing ``responses_create_params`` and verifier metadata."""

    task: NeMoGymTaskConfig = NeMoGymTaskConfig()


class NeMoGymData(TaskData):
    row: dict[str, Any]
    """The exact source row sent back to ``/seed_session`` and ``/verify``."""

    @property
    def responses_create_params(self) -> dict[str, Any]:
        return self.row["responses_create_params"]


class NeMoGymState(State):
    resources_url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    request_timeout: float = 60.0
    cookies: dict[str, str] = Field(default_factory=dict)
    mcp_url: str | None = None
    mcp_headers: dict[str, str] = Field(default_factory=dict)
    direct_tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)


class _NeMoGymToolset(Toolset[SharedToolsetConfig, NeMoGymState]):
    """Expose one rollout's Gym tools through the standard V1 MCP boundary.

    Gym's MCP endpoint is stateless HTTP; its signed seed header restores the hidden
    rollout session on every request.
    """

    TOOL_PREFIX = "nemo_gym"

    def _register(self, mcp: FastMCP) -> None:
        server = mcp._mcp_server
        server.list_tools()(self._with_state(self.list_tools))
        server.call_tool(validate_input=False)(self._with_state(self.call_tool))

    @asynccontextmanager
    async def _upstream(self) -> AsyncIterator[ClientSession]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        if self.state.mcp_url is None:
            raise RuntimeError("NeMo Gym did not provide an MCP URL")
        async with httpx.AsyncClient(
            headers=self.state.mcp_headers,
            timeout=httpx.Timeout(self.state.request_timeout),
        ) as client:
            async with streamable_http_client(
                self.state.mcp_url, http_client=client
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
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

        if name not in self.state.tool_names:
            raise ValueError(f"unknown NeMo Gym tool: {name}")
        if self.state.mcp_url is not None:
            async with self._upstream() as session:
                return await session.call_tool(name, arguments)

        async with httpx.AsyncClient(timeout=self.state.request_timeout) as client:
            response = await client.post(
                f"{self.state.resources_url.rstrip('/')}/{name}",
                headers=self.state.headers,
                cookies=self.state.cookies,
                json=arguments,
            )
        self.state.cookies.update(dict(response.cookies))
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

    branches = trace.branches
    if len(branches) != 1:
        raise ValueError(
            f"NeMo Gym scoring requires exactly one trace branch, got {len(branches)}"
        )

    known_names = set(tool_names)
    known_names.update(
        spec["name"]
        for spec in responses_create_params.get("tools") or []
        if spec.get("type") == "function" and isinstance(spec.get("name"), str)
    )
    aliases = {
        name: {
            name,
            f"nemo_gym_{name}",
            f"nemo_gym__{name}",
            f"mcp__nemo_gym__{name}",
        }
        for name in known_names
    }
    response_item_types = {
        "reasoning",
        "message",
        "function_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
    }
    output: list[dict[str, Any]] = []
    started = False

    for node in branches[0].nodes:
        message = node.message
        if isinstance(message, AssistantMessage) and node.sampled:
            started = True
            provider_items = message.provider_state or []
            if provider_items and all(
                isinstance(item, dict) and item.get("type") in response_item_types
                for item in provider_items
            ):
                items = copy.deepcopy(provider_items)
                for item in items:
                    if item.get("type") != "function_call":
                        continue
                    emitted_name = str(item.get("name"))
                    for raw_name in sorted(known_names, key=len, reverse=True):
                        if emitted_name in aliases[raw_name]:
                            item["name"] = raw_name
                            break
                output.extend(items)
                continue

            if message.reasoning_content:
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
            for call in message.tool_calls or []:
                name = call.name
                for raw_name in sorted(known_names, key=len, reverse=True):
                    if name in aliases[raw_name]:
                        name = raw_name
                        break
                output.append(
                    {
                        "id": call.id,
                        "type": "function_call",
                        "call_id": call.id,
                        "name": name,
                        "arguments": call.arguments,
                        "status": "completed",
                    }
                )
            if message.content:
                output.append(
                    {
                        "id": f"msg_{trace.id}_{len(output)}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": message.content,
                                "annotations": [],
                            }
                        ],
                    }
                )
        elif started and isinstance(message, ToolMessage):
            output.append(
                {
                    "id": f"fco_{trace.id}_{len(output)}",
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": content_text(message.content),
                    "status": "completed",
                }
            )

    model = responses_create_params.get("model") or "verifiers"
    return {
        "id": f"resp_{trace.id}",
        "created_at": branches[0].nodes[-1].timestamp,
        "model": str(model),
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
        state.direct_tools = list(self.data.responses_create_params.get("tools") or [])

        async with httpx.AsyncClient(timeout=state.request_timeout) as client:
            response = await client.post(
                f"{state.resources_url}/seed_session",
                headers=state.headers,
                json=self.data.row,
            )
        response.raise_for_status()
        state.cookies = dict(response.cookies)
        seed = response.json()
        metadata = seed.get("mcp")
        if metadata is None:
            return
        if metadata.get("transport") != "http":
            raise ValueError(
                f"unsupported NeMo Gym MCP transport: {metadata.get('transport')!r}"
            )
        state.mcp_url = urljoin(
            f"{state.resources_url}/", metadata.get("url_path", "/mcp")
        )
        state.mcp_headers = {
            **state.headers,
            **dict(metadata.get("headers") or {}),
        }

    @reward(weight=1.0)
    async def nemo_gym(self, trace: Trace) -> float:
        state = trace.state
        response_body = _trace_to_nemo_response(
            trace,
            self.data.responses_create_params,
            state.tool_names,
        )
        async with httpx.AsyncClient(timeout=state.request_timeout) as client:
            response = await client.post(
                f"{state.resources_url}/verify",
                headers=state.headers,
                cookies=state.cookies,
                json={**self.data.row, "response": response_body},
            )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise TypeError("NeMo Gym /verify response must be an object")

        omitted = {"responses_create_params", "response", "reward"}
        trace.info["nemo_gym"] = {
            key: value for key, value in result.items() if key not in omitted
        }
        trace.record_metrics(
            {
                key: float(value)
                for key, value in result.items()
                if key not in omitted and isinstance(value, (bool, int, float))
            }
        )
        return float(result["reward"])


class NeMoGymTaskset(Taskset[NeMoGymTask, NeMoGymConfig]):
    tools = (_NeMoGymToolset,)

    def load(self) -> Iterator[NeMoGymTask]:
        if self.config.dataset_path is None:
            raise ValueError("NeMoGymConfig.dataset_path is required")
        path = self.config.dataset_path.expanduser().resolve()
        dialect = ResponsesDialect()
        count = 0

        with path.open(encoding="utf-8") as dataset:
            for line_number, line in enumerate(dataset, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(row, dict) or not isinstance(
                    row.get("responses_create_params"), dict
                ):
                    raise ValueError(
                        f"{path} line {line_number} must be an object with "
                        "responses_create_params"
                    )
                params = row["responses_create_params"]
                prompt, _ = dialect.parse_request(params)
                yield NeMoGymTask(
                    NeMoGymData(
                        idx=count,
                        name=f"{path.stem}:{count}",
                        prompt=prompt,
                        row=row,
                    ),
                    self.config.task,
                )
                count += 1

        if count == 0:
            raise ValueError(f"NeMo Gym dataset is empty: {path}")


if __name__ == "__main__":
    _NeMoGymToolset.run()
