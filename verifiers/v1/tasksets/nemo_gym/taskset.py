"""NeMo Gym resource-server tasks driven by Verifiers harnesses."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast
from urllib.parse import urljoin

import httpx
from pydantic import Field

from verifiers.v1.decorators import reward
from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.envs.single_agent import SingleAgentEnv
from verifiers.v1.mcp import SharedToolsetConfig, Toolset
from verifiers.v1.mcp.launch import mcp_session
from verifiers.v1.runtimes import SubprocessConfig, make_runtime
from verifiers.v1.state import State
from verifiers.v1.task import Task, TaskConfig, TaskData
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, ToolMessage
from verifiers.utils.serve_utils import get_free_port

NEMO_GYM_INSTALL_HINT = "uv sync --python 3.12 --extra nemo-gym"

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult, Tool as MCPTool

    from verifiers.v1.runtimes import Runtime


class NeMoGymTaskConfig(TaskConfig):
    resources_url: str | None = None
    """Base URL of an existing server; managed tasksets fill this automatically."""

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
    """Bridge rollout-specific Gym tools into the standard V1 MCP boundary."""

    TOOL_PREFIX = None

    def _register(self, mcp: FastMCP) -> None:
        server = mcp._mcp_server
        server.list_tools()(self._with_state(self.list_tools))
        server.call_tool(validate_input=False)(self._with_state(self.call_tool))

    async def list_tools(self) -> list[MCPTool]:
        from mcp.types import Tool

        if self.state.mcp_url is not None:
            async with mcp_session(
                self.state.mcp_url,
                headers=self.state.mcp_headers,
                timeout=self.state.request_timeout,
            ) as session:
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
            async with mcp_session(
                self.state.mcp_url,
                headers=self.state.mcp_headers,
                timeout=self.state.request_timeout,
            ) as session:
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
            if message.provider_state and all(
                item.get("type") in response_item_types
                for item in message.provider_state
            ):
                output.extend(dict(item) for item in message.provider_state)
                continue
            if message.reasoning_content:
                output.append(
                    {
                        "id": f"rs_{trace.id}_{len(output)}",
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": message.reasoning_content}
                        ],
                    }
                )
            if message.content:
                output.append({"role": "assistant", "content": message.content})
            output.extend(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in message.tool_calls or []
            )
        elif started and isinstance(message, ToolMessage):
            content = (
                message.content
                if isinstance(message.content, str)
                else json.dumps(
                    [part.model_dump(mode="json") for part in message.content]
                )
            )
            output.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": content,
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
    async def setup(self, trace: Trace, runtime: Runtime) -> None:
        state = trace.state
        if self.config.resources_url is None:
            raise ValueError("set resources_url or use a managed NeMo Gym taskset")
        state.resources_url = self.config.resources_url.rstrip("/")
        state.headers = dict(self.config.headers)
        state.request_timeout = self.config.request_timeout
        state.direct_tools = self.data.row["responses_create_params"].get("tools") or []

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
    resource_server: ClassVar[str | None] = None
    """Import reference for a package-provided resource server, if managed."""

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


class NeMoGymEnv(SingleAgentEnv):
    """Start a taskset's NeMo resource server once per environment worker."""

    _nemo_runtime: Runtime | None = None

    async def start(self) -> None:
        taskset = cast(NeMoGymTaskset, self.taskset)
        config = taskset.config.task
        if config.resources_url is not None:
            return
        if importlib.util.find_spec("nemo_gym") is None:
            raise RuntimeError(
                "Managed NeMo Gym tasksets require the `nemo-gym` extra. "
                f"Install it with: `{NEMO_GYM_INSTALL_HINT}`"
            )
        entrypoint = taskset.resource_server
        if entrypoint is None:
            raise ValueError("set --env.taskset.task.resources-url")

        runtime = self._nemo_runtime = make_runtime(SubprocessConfig())
        await runtime.start()
        port = get_free_port()
        await runtime.run_background(
            [sys.executable, "-m", "verifiers.v1.tasksets.nemo_gym.server"],
            {
                "NEMO_GYM_PORT": str(port),
                "NEMO_GYM_RESOURCE_SERVER": entrypoint,
            },
            "nemo_gym.log",
        )
        config.resources_url = f"http://127.0.0.1:{port}"

        async with httpx.AsyncClient(timeout=1) as client:
            for _ in range(60):
                try:
                    if (await client.get(config.resources_url)).is_success:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        log = (await runtime.read("nemo_gym.log")).decode(errors="replace")[-2000:]
        raise RuntimeError(f"NeMo Gym server did not start:\n{log}")

    async def stop(self) -> None:
        if self._nemo_runtime is None:
            return
        runtime, self._nemo_runtime = self._nemo_runtime, None
        cast(NeMoGymTaskset, self.taskset).config.task.resources_url = None
        await runtime.stop()


if __name__ == "__main__":
    _NeMoGymToolset.run()
