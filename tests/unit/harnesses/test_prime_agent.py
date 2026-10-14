"""The prime-agent harness installs and finds its kernel venv in one shared path."""

import copy

import verifiers.v1 as vf
from verifiers.v1.harnesses.prime_agent import harness as prime_agent_harness
from verifiers.v1.harnesses.prime_agent.harness import (
    PRIME_AGENT_DIR,
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
)
from verifiers.v1.runtimes import ProgramResult


class RecordingRuntime:
    """Runtime whose views record each run's env, like an agent_user seat."""

    def __init__(self, user: str | None = "agent") -> None:
        self.user = user
        self.runs: list[tuple[str | None, dict[str, str]]] = []

    def with_user(
        self, user: str | None, home: str | None = None
    ) -> "RecordingRuntime":
        view = copy.copy(self)
        view.user = user
        return view

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.runs.append((self.user, env))
        return ProgramResult(0, "", "")

    async def prepare_uv_script(
        self,
        script: str | bytes,
        env: dict[str, str] | None = None,
        *,
        activate: bool = True,
    ) -> list[str]:
        return ["python", "acp.py"]


async def test_install_and_session_environments_share_one_kernel_venv(monkeypatch):
    async def skip_node(runtime: RecordingRuntime) -> None:
        return None

    monkeypatch.setattr(prime_agent_harness, "ensure_node", skip_node)
    runtime = RecordingRuntime()
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))
    await harness.setup(runtime)
    install_env = runtime.runs[-1][1]

    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
    )
    session_env = harness._env(trace, "secret")

    venv = install_env["PRIME_AGENT_KERNEL_VENV"]
    assert venv == session_env["PRIME_AGENT_KERNEL_VENV"]
    assert venv.startswith(f"{PRIME_AGENT_DIR}/")
