"""NeMo Gym resource-server tasks driven by Verifiers harnesses."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, cast

import httpx
from pydantic import Field

from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.envs.single_agent import SingleAgentEnv
from verifiers.v1.mcp import SharedToolsetConfig, Toolset
from verifiers.v1.runtimes import Runtime, SubprocessConfig, make_runtime
from verifiers.v1.task import Task, TaskConfig, TaskData
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.utils.decorators import reward

from .response import trace_to_nemo_response
from .toolset import NeMoGymState, NeMoGymToolset


class NeMoGymTaskConfig(TaskConfig):
    resources_url: str | None = None
    """Base URL of an existing server; managed tasksets fill this automatically."""

    headers: dict[str, str] = Field(default_factory=dict)
    """Headers added to seed, direct-tool, MCP, and verification requests."""

    request_timeout: float = Field(60.0, gt=0)
    """Per-request timeout for this task's Gym HTTP and MCP calls."""


class NeMoGymConfig(TasksetConfig):
    dataset: Path
    """JSONL rows containing ``responses_create_params`` and verifier metadata."""

    tools: SharedToolsetConfig = SharedToolsetConfig()
    task: NeMoGymTaskConfig = NeMoGymTaskConfig()


class NeMoGymData(TaskData):
    row: dict[str, Any]
    """The exact source row sent back to ``/seed_session`` and ``/verify``."""


class NeMoGymTask(Task[NeMoGymData, NeMoGymState, NeMoGymTaskConfig]):
    async def setup(self, trace: Trace, runtime: Runtime) -> None:
        state = trace.state
        if self.config.resources_url is None:
            raise ValueError("set resources_url or use a managed NeMo Gym taskset")
        state.resources_url = self.config.resources_url.rstrip("/")
        state.headers = dict(self.config.headers)
        state.request_timeout = self.config.request_timeout
        tools = self.data.row["responses_create_params"].get("tools") or []
        state.direct_tools = {
            tool["name"]: tool for tool in tools if tool.get("type") == "function"
        }

        response = await state.post("seed_session", self.data.row)
        response.raise_for_status()
        metadata = response.json().get("mcp")
        if metadata is None:
            return
        if metadata.get("transport") != "http":
            raise ValueError(
                f"unsupported NeMo Gym MCP transport: {metadata.get('transport')!r}"
            )
        mcp_path = metadata.get("url_path", "/mcp").lstrip("/")
        state.mcp_url = f"{state.resources_url}/{mcp_path}"
        state.mcp_headers = state.headers | (metadata.get("headers") or {})

    @reward(weight=1.0)
    async def nemo_gym(self, trace: Trace) -> float:
        state = trace.state
        params = self.data.row["responses_create_params"]
        response = await state.post(
            "verify",
            self.data.row | {"response": trace_to_nemo_response(trace, params)},
        )
        response.raise_for_status()
        result = response.json()
        details = {
            key: value
            for key, value in result.items()
            if key not in {"responses_create_params", "response", "reward"}
        }
        trace.info["nemo_gym"] = details
        for key, value in details.items():
            if isinstance(value, (bool, int, float)):
                trace.record_metric(key, float(value))
        return float(result["reward"])


class NeMoGymTaskset(Taskset[NeMoGymTask, NeMoGymConfig]):
    resource_server: ClassVar[str | None] = None
    """Import reference for a package-provided resource server, if managed."""

    @classmethod
    def toolsets(cls, config: NeMoGymConfig) -> list[Toolset]:
        return [NeMoGymToolset(config.tools)]

    def load(self) -> Iterator[NeMoGymTask]:
        path = self.config.dataset.expanduser().resolve()
        dialect = ResponsesDialect()
        found = False

        with path.open(encoding="utf-8") as lines:
            for idx, line in enumerate(filter(str.strip, lines)):
                found = True
                row = json.loads(line)
                prompt, _ = dialect.parse_request(row["responses_create_params"])
                yield NeMoGymTask(
                    NeMoGymData(
                        idx=idx,
                        name=f"{path.stem}:{idx}",
                        prompt=prompt,
                        row=row,
                    ),
                    self.config.task,
                )

        if not found:
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
                "Install it with: `uv sync --python 3.12 --extra nemo-gym`"
            )
        entrypoint = taskset.resource_server
        if entrypoint is None:
            raise ValueError("set --env.taskset.task.resources-url")

        runtime = self._nemo_runtime = make_runtime(SubprocessConfig())
        await runtime.start()
        await runtime.run_background(
            [sys.executable, "-m", "verifiers.v1.tasksets.nemo_gym.server"],
            {"NEMO_GYM_RESOURCE_SERVER": entrypoint},
            "nemo_gym.log",
        )

        async with httpx.AsyncClient(timeout=1) as client:
            for _ in range(60):
                try:
                    port = int((await runtime.read("nemo_gym.port")).decode())
                    resources_url = f"http://127.0.0.1:{port}"
                    if (await client.get(resources_url)).is_success:
                        config.resources_url = resources_url
                        return
                except (FileNotFoundError, ValueError, httpx.HTTPError):
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
