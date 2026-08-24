from typing import cast

import pytest
from pydantic import ValidationError

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.harnesses.cline import ClineHarness, ClineHarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Sampling


class RecordingRuntime:
    def __init__(self) -> None:
        self.commands: list[tuple[list[str], dict[str, str]]] = []
        self.programs: list[tuple[list[str], dict[str, str]]] = []
        self.files: dict[str, bytes] = {}

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.commands.append((argv, env))
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.programs.append((argv, env))
        return ProgramResult(exit_code=0, stdout="", stderr="")


def test_cline_config_validates_pinned_version_and_retries() -> None:
    config = ClineHarnessConfig(id="cline")
    assert config.version == "3.0.57"
    assert config.compaction == "basic"
    assert config.max_retries == 6

    with pytest.raises(ValidationError):
        ClineHarnessConfig(id="cline", version="3.0.57; unsafe")
    with pytest.raises(ValidationError):
        ClineHarnessConfig(id="cline", max_retries=0)


@pytest.mark.asyncio
async def test_cline_launch_wires_interception_and_restricts_tools() -> None:
    runtime = RecordingRuntime()
    harness = ClineHarness(
        ClineHarnessConfig(id="cline", disabled_tools=["custom_external_tool"])
    )
    ctx = ModelContext(
        model="openai/example",
        client=EvalClientConfig(),
        sampling=Sampling(reasoning_effort="medium"),
    )
    trace = Trace.model_construct(id="trace-id")

    result = await harness.launch(
        ctx,
        trace,
        cast(Runtime, runtime),
        "http://interception.test/v1",
        "ephemeral-secret",
        {},
        TaskData(prompt="solve the task", system_prompt="follow the rules"),
    )

    assert result.exit_code == 0
    settings = runtime.files[
        "/tmp/vf-cline/trace-id/settings/global-settings.json"
    ].decode()
    assert '"fetch_web_content"' in settings
    assert '"spawn_agent"' in settings
    assert '"custom_external_tool"' in settings
    assert (
        runtime.files["/tmp/vf-cline/trace-id/settings/cline_mcp_settings.json"]
        == b'{"mcpServers":{}}'
    )

    auth, _ = runtime.commands[-1]
    assert auth[auth.index("--baseurl") + 1] == "http://interception.test/v1"
    assert auth[auth.index("--modelid") + 1] == "openai/example"

    argv, env = runtime.programs[-1]
    assert argv[argv.index("--key") + 1] == "ephemeral-secret"
    assert argv[argv.index("--model") + 1] == "openai/example"
    assert argv[argv.index("--thinking") + 1] == "medium"
    assert argv[-1] == "follow the rules\n\nsolve the task"
    assert env["CLINE_TELEMETRY_DISABLED"] == "1"
    assert env["CLINE_MCP_SETTINGS_PATH"].endswith("cline_mcp_settings.json")


@pytest.mark.asyncio
async def test_cline_rejects_mcp_before_starting_program() -> None:
    runtime = RecordingRuntime()
    harness = ClineHarness(ClineHarnessConfig(id="cline"))
    ctx = ModelContext(model="openai/example", client=EvalClientConfig())

    with pytest.raises(ValueError, match="does not support MCP"):
        await harness.launch(
            ctx,
            Trace.model_construct(id="trace-id"),
            cast(Runtime, runtime),
            "http://interception.test/v1",
            "ephemeral-secret",
            {"server": "http://mcp.test"},
            TaskData(prompt="solve the task"),
        )

    assert runtime.programs == []
