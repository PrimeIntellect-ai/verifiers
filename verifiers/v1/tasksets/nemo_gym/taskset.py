"""NeMo Gym resource-server tasks driven by Verifiers harnesses."""

import asyncio
import json
import math
import secrets
from collections.abc import Iterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, ClassVar, cast

import httpx
from pydantic import Field

from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.envs.single_agent import SingleAgentEnv
from verifiers.v1.interception.tunnel import PrimeTunnel
from verifiers.v1.interception.tunnel.base import HEALTH_PATH
from verifiers.v1.mcp import SharedToolsetConfig, Toolset
from verifiers.v1.mcp.launch import log_tail, read_port
from verifiers.v1.runtimes import (
    PrimeConfig,
    Runtime,
    SubprocessConfig,
    provision_runtime,
)
from verifiers.v1.task import Task, TaskConfig, TaskData
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.utils.decorators import reward

from .response import trace_to_nemo_response
from .toolset import NeMoGymState, NeMoGymToolset

RESOURCE_TOKEN_HEADER = "X-Verifiers-Nemo-Token"


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

    tools: SharedToolsetConfig = SharedToolsetConfig(runtime=PrimeConfig())
    """The shared tool server runs on Prime by default."""
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
        state.tool_names = list(state.direct_tools)

        response = await state.post("seed_session", self.data.row)
        response.raise_for_status()
        state.cookies.update(response.cookies)
        if metadata := response.json().get("mcp"):
            state.mcp_url = f"{state.resources_url}/{metadata['url_path'].lstrip('/')}"
            state.mcp_headers = state.headers | metadata["headers"]

    @reward(weight=1.0)
    async def nemo_gym(self, trace: Trace) -> float:
        state = trace.state
        params = self.data.row["responses_create_params"]
        response = await state.post(
            "verify",
            self.data.row
            | {"response": trace_to_nemo_response(trace, params, state.tool_names)},
        )
        response.raise_for_status()
        result = response.json()
        reward = result.pop("reward")
        del result["responses_create_params"], result["response"]
        trace.info["nemo_gym"] = result
        for key, value in result.items():
            if isinstance(value, (bool, int, float)):
                trace.record_metric(key, float(value))
        return float(reward)


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
                request = dialect.parse_request(row["responses_create_params"])
                yield NeMoGymTask(
                    NeMoGymData(
                        name=f"{path.stem}:{idx}",
                        prompt=request.messages,
                        row=row,
                    ),
                    self.config.task,
                )

        if not found:
            raise ValueError(f"NeMo Gym dataset is empty: {path}")


class NeMoGymEnv(SingleAgentEnv):
    """Start a taskset's NeMo resource server once per environment worker."""

    _nemo_stack: AsyncExitStack | None = None

    async def start(self) -> None:
        taskset = cast(NeMoGymTaskset, self.taskset)
        config = taskset.config.task
        if config.resources_url is not None:
            return
        entrypoint = taskset.resource_server
        if entrypoint is None:
            raise ValueError("set --env.taskset.task.resources-url")

        token = secrets.token_urlsafe(32)
        async with AsyncExitStack() as stack:
            runtime = await stack.enter_async_context(
                provision_runtime(SubprocessConfig())
            )
            server = Path(__file__).with_name("server.py")
            program = await runtime.prepare_uv_script(server.read_bytes())
            await runtime.run_background(
                program,
                {"NEMO_GYM_RESOURCE_SERVER": entrypoint, "NEMO_GYM_TOKEN": token},
                "nemo_gym.log",
            )
            timeout = taskset.config.tools.timeout.startup
            try:
                async with (
                    asyncio.timeout(timeout),
                    httpx.AsyncClient(timeout=1) as client,
                ):
                    port = await read_port(runtime, "nemo_gym.port", math.inf)
                    resources_url = await stack.enter_async_context(
                        PrimeTunnel().expose(port)
                    )
                    while True:
                        try:
                            response = await client.get(f"{resources_url}{HEALTH_PATH}")
                            if response.status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(0.5)
            except TimeoutError as exc:
                log = await log_tail(runtime, "nemo_gym.log")
                raise RuntimeError(f"NeMo Gym server did not start:\n{log}") from exc

            stack.callback(setattr, config, "headers", config.headers)
            stack.callback(setattr, config, "resources_url", config.resources_url)
            config.headers = {
                key: value
                for key, value in config.headers.items()
                if key.lower() != RESOURCE_TOKEN_HEADER.lower()
            } | {RESOURCE_TOKEN_HEADER: token}
            config.resources_url = resources_url
            self._nemo_stack = stack.pop_all()

    async def stop(self) -> None:
        if self._nemo_stack is None:
            return
        stack, self._nemo_stack = self._nemo_stack, None
        await stack.aclose()
