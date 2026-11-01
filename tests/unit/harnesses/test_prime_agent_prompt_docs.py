"""Prime Agent's `prompt_docs`: a replacement system prompt with its reference files."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.sampling import Sampling
from verifiers.v1.harnesses.prime_agent import (
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
)
from verifiers.v1.harnesses.prime_agent.harness import REFERENCE_DIR


class RecordingRuntime:
    """Records each write with the user it was made as."""

    def __init__(self, files=None, user: str | None = "agent") -> None:
        self.files: dict[str, tuple[str | None, bytes]] = {} if files is None else files
        self.user = user

    def with_user(self, user: str | None) -> "RecordingRuntime":
        return RecordingRuntime(self.files, user)

    async def run(self, command, env):
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = (self.user, data)


async def written(**config) -> dict[str, tuple[str | None, bytes]]:
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(**config))
    runtime = RecordingRuntime()
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")),
    )
    ctx = ModelContext(
        model="openai/gpt-oss-20b",
        client=vf.EvalClientConfig(),
        sampling=Sampling(reasoning_effort="low"),
    )
    await harness.prepare_acp(
        ctx, trace, runtime, "http://proxy.invalid/v1", "secret", {}, trace.task.data
    )
    return runtime.files


async def test_default_writes_no_prompt():
    files = await written()
    assert not [p for p in files if p.endswith(("/SYSTEM.md", "/settings.json"))]
    assert not [p for p in files if p.startswith(REFERENCE_DIR)]


async def test_prompt_docs(tmp_path):
    (tmp_path / "SYSTEM.md").write_text("tiny")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "rules.md").write_text("rules")
    files = await written(prompt_docs=tmp_path)
    agent_dir = next(p for p in files if p.endswith("/models.json")).removesuffix(
        "/models.json"
    )
    assert files[f"{agent_dir}/SYSTEM.md"] == ("agent", b"tiny")
    user, settings = files[f"{agent_dir}/settings.json"]
    assert (user, json.loads(settings)) == ("agent", {"skillsInPrompt": False})
    # The reference directory is the system's, so its files are written as root.
    assert files[f"{REFERENCE_DIR}/rules.md"] == (None, b"rules")


def test_prompt_docs_checked_at_config(tmp_path):
    (tmp_path / "docs" / "nested").mkdir(parents=True)
    (tmp_path / "SYSTEM.md").write_text("tiny")
    with pytest.raises(ValidationError):
        PrimeAgentHarnessConfig(prompt_docs=tmp_path)
    with pytest.raises(ValidationError):
        PrimeAgentHarnessConfig(prompt_docs=tmp_path / "missing")
