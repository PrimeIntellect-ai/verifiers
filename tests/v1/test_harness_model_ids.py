"""Offline model-ID launch contracts for native agent harnesses."""

import json
from types import SimpleNamespace

import pytest

from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.harnesses.openclaw.harness import (
    OpenClawHarness,
    OpenClawHarnessConfig,
)
from verifiers.v1.harnesses.pi.harness import PiHarness, PiHarnessConfig
from verifiers.v1.task import TaskData


class _Runtime:
    def __init__(self) -> None:
        self.writes: dict[str, bytes] = {}
        self.runs: list[list[str]] = []
        self.background: list[list[str]] = []

    async def write(self, path: str, data: bytes) -> None:
        self.writes[path] = data

    async def run(self, command: list[str], environment: dict[str, str]):
        del environment
        self.runs.append(command)
        return SimpleNamespace(exit_code=0, stdout="12345", stderr="")

    async def run_background(
        self,
        command: list[str],
        environment: dict[str, str],
        log_path: str,
    ) -> None:
        del environment, log_path
        self.background.append(command)


async def _capture_acp(*args, **kwargs):
    del args, kwargs
    return SimpleNamespace(exit_code=0, stdout="", stderr="")


def _context(model: str) -> ModelContext:
    return ModelContext(model=model, client=EvalClientConfig())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "provider", "primary"),
    [
        ("gpt-5.6-luna", "openai", "openai/gpt-5.6-luna"),
        (
            "openrouter/meta-llama/llama-3.3-70b",
            "openrouter",
            "openrouter/meta-llama/llama-3.3-70b",
        ),
    ],
)
async def test_openclaw_launch_normalizes_bare_model_ids(
    monkeypatch: pytest.MonkeyPatch, model: str, provider: str, primary: str
) -> None:
    import verifiers.v1.harnesses.openclaw.harness as openclaw

    monkeypatch.setattr(openclaw.OPENCLAW_ACP, "run", _capture_acp)
    runtime = _Runtime()
    await OpenClawHarness(OpenClawHarnessConfig()).launch(
        _context(model),
        SimpleNamespace(id="trace"),
        runtime,
        "http://intercept",
        "secret",
        {},
        TaskData(prompt="hello"),
    )

    config = json.loads(runtime.writes[".vf-openclaw/trace/openclaw.json"])
    assert config["agents"]["defaults"]["model"]["primary"] == primary
    assert config["models"]["providers"] == {
        provider: {"baseUrl": "http://intercept", "apiKey": "${OPENCLAW_INTERCEPT_KEY}"}
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "provider", "remainder"),
    [
        ("gpt-5.6-luna", "openai", "gpt-5.6-luna"),
        (
            "openrouter/meta-llama/llama-3.3-70b",
            "openrouter",
            "meta-llama/llama-3.3-70b",
        ),
    ],
)
async def test_pi_launch_normalizes_bare_model_ids(
    monkeypatch: pytest.MonkeyPatch, model: str, provider: str, remainder: str
) -> None:
    import verifiers.v1.harnesses.pi.harness as pi

    monkeypatch.setattr(pi.PI_ACP, "run", _capture_acp)
    runtime = _Runtime()
    await PiHarness(PiHarnessConfig()).launch(
        _context(model),
        SimpleNamespace(id="trace"),
        runtime,
        "http://intercept",
        "secret",
        {},
        TaskData(prompt="hello"),
    )

    models = json.loads(runtime.writes[".vf-pi-agent-trace/models.json"])
    assert models["providers"] == {
        provider: {"baseUrl": "http://intercept", "apiKey": "$PI_INTERCEPT_KEY"}
    }
    wrapper = runtime.writes[".vf-pi-agent-trace/pi"].decode()
    assert f"--provider {provider} --model {remainder}" in wrapper
