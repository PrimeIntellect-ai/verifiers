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
    "model",
    [
        "gpt-5.6-luna",
        "openai/gpt-5.6-luna",
        "internal/glm-5.2-fast",
        "custom/non-catalog-model-v1",
        "openrouter/meta-llama/llama-3.3-70b",
    ],
)
async def test_openclaw_launch_isolates_every_model_id(
    monkeypatch: pytest.MonkeyPatch, model: str
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
    assert config["agents"]["defaults"]["model"]["primary"] == f"intercept/{model}"
    assert config["models"]["mode"] == "replace"
    assert config["models"]["providers"] == {
        "intercept": {
            "baseUrl": "http://intercept",
            "apiKey": "${OPENCLAW_INTERCEPT_KEY}",
            "api": "openai-responses",
            "authHeader": True,
            "models": [
                {
                    "id": model,
                    "name": model,
                    "reasoning": False,
                    "input": ["text", "image"],
                }
            ],
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.6-luna",
        "openai/gpt-5.6-luna",
        "internal/glm-5.2-fast",
        "custom/non-catalog-model-v1",
        "openrouter/meta-llama/llama-3.3-70b",
    ],
)
async def test_pi_launch_isolates_every_model_id(
    monkeypatch: pytest.MonkeyPatch, model: str
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
        "intercept": {
            "baseUrl": "http://intercept",
            "api": "openai-completions",
            "apiKey": "$PI_INTERCEPT_KEY",
            "models": [
                {
                    "id": model,
                    "reasoning": model.rsplit("/", 1)[-1].startswith(
                        ("gpt-5", "o1", "o3", "o4")
                    ),
                    "input": ["text", "image"],
                }
            ],
        }
    }
    wrapper = runtime.writes[".vf-pi-agent-trace/pi"].decode()
    assert f"--provider intercept --model {model}" in wrapper
