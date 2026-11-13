"""Prompt text never reaches a process command line. An agent's `pkill -f <word>`
matches every process whose argv holds the word, so a harness that passes the task or
system prompt on argv can be killed by the task it runs. Each harness delivers the
text by file (`runtime.write`) or over its ACP channel instead."""

from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.acp import ACPHarness, ACPHarnessSession
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.harnesses import (
    BashHarness,
    BashHarnessConfig,
    BrowserUseHarness,
    BrowserUseHarnessConfig,
    ClaudeCodeHarness,
    ClaudeCodeHarnessConfig,
    CodexHarness,
    CodexHarnessConfig,
    DeepSeekHarness,
    DeepSeekHarnessConfig,
    HermesAgentHarness,
    HermesAgentHarnessConfig,
    KimiCodeHarness,
    KimiCodeHarnessConfig,
    MiniSWEAgentHarness,
    MiniSWEAgentHarnessConfig,
    NullHarness,
    NullHarnessConfig,
    OpenClawHarness,
    OpenClawHarnessConfig,
    OpenCodeHarness,
    OpenCodeHarnessConfig,
    PiHarness,
    PiHarnessConfig,
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
    RLMHarness,
    RLMHarnessConfig,
    StirrupHarness,
    StirrupHarnessConfig,
    Terminus2Harness,
    Terminus2HarnessConfig,
)
from verifiers.v1.runtimes import ProgramResult

TASK = "VF_SENTINEL_TASK_7f3a"
SYSTEM = "VF_SENTINEL_SYS_7f3a"

HARNESSES = {
    "null": (NullHarness, NullHarnessConfig),
    "bash": (BashHarness, BashHarnessConfig),
    "browser_use": (BrowserUseHarness, BrowserUseHarnessConfig),
    "terminus_2": (Terminus2Harness, Terminus2HarnessConfig),
    "mini_swe_agent": (MiniSWEAgentHarness, MiniSWEAgentHarnessConfig),
    "stirrup": (StirrupHarness, StirrupHarnessConfig),
    "pi": (PiHarness, PiHarnessConfig),
    "prime_agent": (PrimeAgentHarness, PrimeAgentHarnessConfig),
    "claude_code": (ClaudeCodeHarness, ClaudeCodeHarnessConfig),
    "codex": (CodexHarness, CodexHarnessConfig),
    "deepseek": (DeepSeekHarness, DeepSeekHarnessConfig),
    "hermes_agent": (HermesAgentHarness, HermesAgentHarnessConfig),
    "kimi_code": (KimiCodeHarness, KimiCodeHarnessConfig),
    "openclaw": (OpenClawHarness, OpenClawHarnessConfig),
    "opencode": (OpenCodeHarness, OpenCodeHarnessConfig),
    "rlm": (RLMHarness, RLMHarnessConfig),
}


class RecordingRuntime:
    """Accepts every command and records each argv and written file."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(type="docker", workdir="/app")
        self.env: dict[str, str] = {}
        self.user = None
        self.argvs: list[list[str]] = []
        self.files: dict[str, bytes] = {}

    def host_url(self, url: str) -> str:
        return url

    async def prepare_uv_script(self, script, env=None, *, activate=True):
        return ["python", "/scripts/program.py"]

    async def run(self, argv, env):
        self.argvs.append(list(argv))
        return ProgramResult(exit_code=0, stdout="/app\n", stderr="")

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        return self.files.get(path, b"")


@pytest.mark.parametrize("name", HARNESSES)
async def test_prompt_text_stays_off_argv(name):
    harness_cls, config_cls = HARNESSES[name]
    harness = harness_cls(config_cls())
    runtime = RecordingRuntime()
    data = vf.TaskData(idx=0, prompt=TASK, system_prompt=SYSTEM)
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=data),
    )
    ctx = ModelContext(model="openai/test-model", client=EvalClientConfig())
    session = await harness.session(
        ctx,
        trace,
        runtime,
        "http://127.0.0.1:1/v1",
        "secret",
        {},
        data,
        tool_interception_url="http://127.0.0.1:1/tool",
    )
    protocol = ""
    if isinstance(harness, ACPHarness):
        assert isinstance(session, ACPHarnessSession)
        config = session.config
        argvs = [*runtime.argvs, config.command]
        protocol = f"{config.prompt} {config.system_prompt} {config.session_meta}"
    else:
        await session.turn()
        argvs = runtime.argvs

    # A script the harness writes is exec'd with its contents as argv.
    scripts = [data for data in runtime.files.values() if data.startswith(b"#!")]
    for sentinel in (TASK, SYSTEM):
        assert not any(sentinel in arg for argv in argvs for arg in argv)
        assert not any(sentinel.encode() in script for script in scripts)
        written = any(sentinel.encode() in data for data in runtime.files.values())
        assert written or sentinel in protocol, f"{sentinel} never reached the program"
