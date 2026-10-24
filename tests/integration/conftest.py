"""Shared fixtures for the integration suite: REAL eval runs (a live model endpoint, real
runtimes) with the smallest settings that still exercise the path, asserting on the
resulting `Trace`(s). They need a model API key (`PRIME_API_KEY`); without one the
model-driven tests skip.

Each behavior module fixes one task and fans it across the harness x runtime grid in
`grid.py`, which also says why a cell is skipped. `run` drives the `Env` API in-process,
the calls the eval CLI makes. The placement fixtures below are indirect-only: they
translate a parametrized value, and using one without `indirect=True` fails loudly.
"""

import asyncio
import os
from pathlib import Path

import pytest
from grid import agent_seats, runtime_config

from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.episode import Episode
from verifiers.v1.utils.loaders import harness_config_type, load_environment

CI_MODEL = "internal/deepseek-v4.1-flash"
CI_REASONING_EFFORT = "low"
# Judge config fields that keep config-plugged judges on the CI model and effort.
CI_JUDGE = {"model": CI_MODEL, "sampling": {"reasoning_effort": CI_REASONING_EFFORT}}
CI_DECISION_JUDGE = {"model": "cloudflare/clef"}

# Fixture tasksets (echo, echo-agentic, ...) live in tests/integration/fixtures,
# added to the path via `pythonpath` in pyproject so the loader resolves them by id.

LIVE_FIXTURES = {"run", "live_ctx"}


@pytest.fixture
def harness_runtime(request) -> dict:
    """The runtime config placing the harness: a config built by the grid, or a
    runtime type from a runtime-only grid."""
    param = request.param
    return dict(param) if isinstance(param, dict) else runtime_config(param)


@pytest.fixture
def tool_runtime(request) -> dict:
    """A `taskset.task.tools` override placing the tool server: `colocated` (inside the harness's
    runtime), `service` (in the task's `tools` Compose service), `service-vm` (in the task's
    `tools-vm` service VM) or its own runtime, by type."""
    if request.param == "colocated":
        return {"colocated": True}
    if request.param == "service":
        return {"service": "tools"}
    if request.param == "service-vm":
        return {"service": "tools-vm"}
    return {"runtime": runtime_config(request.param)}


# Built-in harnesses are bundled in the `harnesses` package; the agent CLIs (`rlm` /
# `kimi-code` / `openclaw` / `codex` / `claude-code` / `hermes-agent`) install their
# dependencies at rollout.
# `compact` (an example harness) and `terminus-2` (drives the host tmux) are outside the grid.
@pytest.fixture
def harness(request) -> HarnessConfig:
    config = {"id": request.param} if isinstance(request.param, str) else request.param
    return harness_config_type(config["id"]).model_validate(config)


def pytest_configure(config) -> None:
    """Self-launching tool servers run `python -m <module>` in a fresh subprocess, which
    inherits `PYTHONPATH` but not pytest's in-process `pythonpath`. Put the fixture dir on
    `PYTHONPATH` so a fixture server module (e.g. `tool_response_image`)
    resolves there too — an installed example package (e.g. `glossary`) already would."""
    config.addinivalue_line(
        "markers", "opencode: integration cells on the OpenCode harness"
    )
    fixtures = str(Path(__file__).parent / "fixtures")
    existing = os.environ.get("PYTHONPATH", "")
    if fixtures not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            f"{fixtures}{os.pathsep}{existing}" if existing else fixtures
        )


def pytest_collection_modifyitems(config, items) -> None:
    """Skip the model-driven tests when no model endpoint is configured; the rest
    (scripted harness behaviors, the examples' own key check) decide for themselves."""
    if os.environ.get("PRIME_API_KEY"):
        return
    skip = pytest.mark.skip(reason="needs PRIME_API_KEY")
    for item in items:
        if LIVE_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(skip)


def _label_prime_runtimes(config: dict) -> None:
    """Tag every prime runtime config (nested: harness / tool) with a `vf-ci` label, so
    leftover sandboxes can be found and cleaned up."""
    if isinstance(config, dict):
        if config.get("type") == "prime":
            config.setdefault("labels", ["vf-ci"])
        for value in config.values():
            _label_prime_runtimes(value)


def _eval_config(
    taskset: str,
    *,
    harness: str | HarnessConfig | None = "null",
    n: int = 1,
    num_tasks: int = 1,
    max_tokens: int = 2048,
    max_turns: int | None = 4,
    rollout_timeout: float = 180,
    taskset_overrides: dict | None = None,
    runtime: dict | None = None,
    env: dict | None = None,
    reasoning_effort: str | None = CI_REASONING_EFFORT,
    model: str = CI_MODEL,
    client: dict | None = None,
) -> EvalConfig:
    """Build the smallest `EvalConfig` that still exercises the path, used by the `run`
    fixture. `taskset_overrides` merges onto the
    `{id: ...}` config; `runtime` places the `agent` seat's harness (an agent field, not a
    harness one).

    `harness=None` leaves every seat on its own story — the multi-agent case: there
    is no run-level harness, so a single-agent test's `harness` lands on the `agent`
    seat and a multi-agent test pins its seats through `env` role fields instead."""
    taskset_cfg = {"id": taskset, **(taskset_overrides or {})}
    env_cfg = dict(env or {})
    _label_prime_runtimes(taskset_cfg)
    if harness:
        env_cfg.setdefault("agent", {})["harness"] = (
            harness_config_type(harness)(id=harness)
            if isinstance(harness, str)
            else harness
        )
    if runtime:
        runtime_cfg = dict(runtime)
        _label_prime_runtimes(runtime_cfg)
        env_cfg.setdefault("agent", {})["runtime"] = runtime_cfg
    retries = {
        "max_retries": 0,
        "rules": [
            {"type": error_type, "max_retries": 2}
            for error_type in ("ProviderError", "InterceptionError", "HarnessError")
        ],
    }
    env_cfg.setdefault("retries", retries)
    # Per-run caps live on the seats: cap each role the env declares (a test's own
    # seat dict wins over the shared defaults).
    for seat in agent_seats(taskset, env_cfg.get("id", "")):
        seat_cfg = env_cfg.setdefault(seat, {})
        seat_cfg.setdefault("max_turns", max_turns)
        seat_cfg.setdefault("max_output_tokens", max_tokens)
        seat_cfg.setdefault("timeout", {"rollout": rollout_timeout, "scoring": 60})
        # Agent runs retry locally; interactions retry with their whole episode.
        seat_cfg.setdefault("retries", retries)
    return EvalConfig(
        env={
            "taskset": taskset_cfg,
            **env_cfg,
        },
        select={"limit": num_tasks},
        num_rollouts=n,
        sampling={
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        },
        model=model,
        **({"client": client} if client else {}),
    )


@pytest.fixture
def run():
    """Run a taskset's episodes in-process through the `Env` API: the calls `vf-eval`
    makes (`serving`, `slots`, `run_slot`, retries included), without its run
    directory, platform push, or dashboard. The Examples job covers that CLI layer."""

    async def _episodes(taskset: str, **kwargs) -> list[Episode]:
        config = _eval_config(taskset, **kwargs)
        env = load_environment(config.env)
        tasks = list(env.taskset.select(config.select))
        ctx = ModelContext(
            client=config.client, model=config.model, sampling=config.sampling
        )
        async with env.serving():
            return list(
                await asyncio.gather(
                    *(
                        env.run_slot(slot, ctx)
                        for task in tasks
                        for slot in env.slots(task, config.num_rollouts)
                    )
                )
            )

    return _episodes


@pytest.fixture
async def live_ctx():
    """The live `ModelContext` (default model + endpoint config, provider-default sampling)
    for driving `Agent` directly — the agent-surface counterpart of `run`."""
    # Endpoint config only — each rollout builds and closes its own client.
    yield ModelContext(
        model=CI_MODEL,
        client=EvalClientConfig(),
        sampling=SamplingConfig(max_tokens=2048, reasoning_effort=CI_REASONING_EFFORT),
    )


@pytest.fixture
async def scripted_model():
    """A deterministic upstream model: every call gets a plain "OK" in the caller's own
    dialect (chat completions, Responses, or Anthropic messages), streamed through that
    dialect's own encoder when asked. Yields a client config pointing at it."""
    from aiohttp import web

    from verifiers.v1.dialects.anthropic import AnthropicDialect
    from verifiers.v1.dialects.chat import ChatDialect
    from verifiers.v1.dialects.responses import ResponsesDialect

    def chat(model: str) -> dict:
        message = {"role": "assistant", "content": "OK"}
        return {
            "id": "chatcmpl-scripted",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def responses(model: str) -> dict:
        text = {"type": "output_text", "text": "OK", "annotations": []}
        return {
            "id": "resp_scripted",
            "object": "response",
            "created_at": 0,
            "model": model,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_scripted",
                    "status": "completed",
                    "role": "assistant",
                    "content": [text],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

    def anthropic(model: str) -> dict:
        return {
            "id": "msg_scripted",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    routes = [
        ("/chat/completions", ChatDialect(), chat),
        ("/responses", ResponsesDialect(), responses),
        ("/messages", AnthropicDialect(), anthropic),
    ]

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        dialect, build = next(
            (dialect, build)
            for suffix, dialect, build in routes
            if request.path.endswith(suffix)
        )
        raw = build(str(body.get("model", "scripted")))
        if not dialect.streaming(body):
            return web.json_response(raw)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in dialect.stream_events(raw):
            await response.write(event)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield {"base_url": f"http://127.0.0.1:{port}/v1"}
    finally:
        await runner.cleanup()
