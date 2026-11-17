"""NeMo Gym resource-server tasks driven by Verifiers harnesses."""

import asyncio
import json
import math
import secrets
from collections.abc import Iterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, ClassVar, cast
from urllib.parse import urlsplit

import httpx
from pydantic import Field

from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.envs.single_agent import SingleAgentEnv
from verifiers.v1.interception.tunnel import PrimeTunnel
from verifiers.v1.mcp import SharedToolsetConfig, Toolset
from verifiers.v1.mcp.launch import _read_back_port, log_tail, tool_runtime
from verifiers.v1.runtimes import Runtime, provision_backend
from verifiers.v1.runtimes.base import SERVICE_PORT
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
    """Placement of the shared tool and managed resource servers; the harness's
    provider by default."""
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
    _nemo_headers: dict[str, str] | None = None

    async def start(self) -> None:
        taskset = cast(NeMoGymTaskset, self.taskset)
        config = taskset.config.task
        if config.resources_url is not None:
            return
        entrypoint = taskset.resource_server
        if entrypoint is None:
            raise ValueError("set --env.taskset.task.resources-url")
        runtime_config = tool_runtime(taskset.config.tools, self.tool_provider())
        if getattr(runtime_config, "network_restricted", False):
            raise ValueError(
                "managed NeMo Gym resource servers need an unrestricted runtime"
            )

        token = secrets.token_urlsafe(32)
        async with AsyncExitStack() as stack:
            backend = await stack.enter_async_context(provision_backend(runtime_config))
            runtime = backend.bind()
            server = Path(__file__).with_name("server.py")
            program = await runtime.prepare_uv_script(server.read_bytes())
            port_file, log_file = "/tmp/nemo_gym.port", "/tmp/nemo_gym.log"
            await runtime.run_background(
                program,
                {
                    "NEMO_GYM_RESOURCE_SERVER": entrypoint,
                    "NEMO_GYM_PORT": str(SERVICE_PORT),
                    "NEMO_GYM_TOKEN": token,
                },
                log_file,
            )
            timeout = taskset.config.tools.timeout.startup
            loop = asyncio.get_running_loop()
            deadline = loop.time() + (math.inf if timeout is None else timeout)
            port = await _read_back_port(runtime, port_file, deadline)
            resources_url = await runtime.expose(port)
            if runtime.is_local:
                # The shared NeMo tool server reaches this from a container of its own,
                # where host loopback is not this machine: publish it.
                resources_url = await stack.enter_async_context(
                    PrimeTunnel().expose(urlsplit(resources_url).port or 80)
                )
            async with httpx.AsyncClient(
                timeout=1, headers={"X-Verifiers-Nemo-Token": token}
            ) as client:
                while loop.time() < deadline:
                    try:
                        response = await client.get(resources_url)
                        if response.status_code < 500 and response.status_code not in {
                            401,
                            403,
                        }:
                            self._nemo_headers = dict(config.headers)
                            config.headers = {
                                key: value
                                for key, value in config.headers.items()
                                if key.lower() != "x-verifiers-nemo-token"
                            } | {"X-Verifiers-Nemo-Token": token}
                            config.resources_url = resources_url
                            self._nemo_stack = stack.pop_all()
                            return
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.5)
            raise RuntimeError(
                f"NeMo Gym server did not start:\n{await log_tail(runtime, log_file)}"
            )

    async def stop(self) -> None:
        if self._nemo_stack is None:
            return
        stack, self._nemo_stack = self._nemo_stack, None
        config = cast(NeMoGymTaskset, self.taskset).config.task
        config.resources_url = None
        assert self._nemo_headers is not None
        config.headers, self._nemo_headers = self._nemo_headers, None
        await stack.aclose()
