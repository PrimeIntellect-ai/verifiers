"""Shared fixtures + helpers for the v1 end-to-end eval tests.

These tests run REAL eval runs (a live model endpoint, real runtimes) with the smallest
settings that still exercise the path, then assert on the resulting `Trace`(s) — they are
not unit tests of individual components. They need a model API key (`PRIME_API_KEY`);
without one the `e2e`-marked tests skip (config parsing still runs).

`run_v1` mirrors the eval CLI's in-process path (`run_eval` with `--no-serve`). Placement coverage (harness x harness runtime x tool
server runtime) is PAIRWISE, not a full cross product: each test carries a curated list of
combinations (in test_e2e.py) that hits every axis value and the cross-boundary pairs with
distinct networking. The full cross bought flake exposure and CI minutes, not coverage — add
a combination to a test's list when it exercises a genuinely new reachability pair. The
placement fixtures below are indirect-only: they translate a parametrized value, and using
one without `indirect=True` fails loudly.

Every combination carries its axes' pytest marks, so subsets select with `-m`:

    uv run pytest tests/v1 -n auto                                # everything (needs modal setup)
    uv run pytest tests/v1 -n auto -m "not e2e"                   # deterministic CI matrix
    uv run pytest tests/v1 -n auto -m "e2e and not prime and not modal"  # live CI job
    uv run pytest tests/v1 -n auto -m docker                      # any case touching the docker runtime
    uv run pytest tests/v1 -n auto -m bash                        # only the bash harness
    uv run pytest tests/v1 -n auto -m prime                       # only prime (real sandboxes; local)
    uv run pytest tests/v1 -n auto -m modal                       # only modal (needs local setup)

Marks: runtimes `subprocess` / `docker` / `prime` / `modal`, placement `colocated`,
harnesses `null` / `bash` / `rlm` / `kimi_code` / `pi` / `pool` / `openclaw` / `codex` /
`claude_code` / `hermes_agent`.
A mark is applied per axis, so it selects every case touching that value on ANY axis; for one exact
combination use `-k` on the test id (e.g. `-k "harness-in-docker-with-tool-in-subprocess"`).
prime/modal provision real remote sandboxes (slow, infra-flaky, need setup), so they're local-only.
CI runs deterministic tests across the Python matrix and the remaining live E2Es once.
"""

import os
from pathlib import Path

import pytest

import verifiers.v1 as vf
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.trace import Trace
from verifiers.v1.utils.loaders import harness_config_type

CI_MODEL = "openai/gpt-5.6-luna"

# Fixture tasksets (echo-v1, echo-agentic-v1) live in tests/v1/fixtures, added to the
# path via `pythonpath` in pyproject so the loader resolves them by id (no install).

# The placement fixtures translate one parametrized value each; the combinations live on
# the tests (`indirect=True`), so coverage is a visible, curated list — never an implicit
# cross product.


@pytest.fixture
def harness_runtime(request) -> str:
    return request.param


@pytest.fixture
def tool_runtime(request) -> dict:
    """A `taskset.task.tools` override placing the tool server: `colocated` (inside the harness's
    runtime) or its own runtime, by type."""
    if request.param == "colocated":
        return {"colocated": True}
    if request.param == "docker":
        return {"runtime": {"type": "docker", "allow": ["*"]}}
    return {"runtime": {"type": request.param}}


# Built-in harnesses are bundled in the `harnesses` package; the agent CLIs (`rlm` /
# `kimi-code` / `openclaw` / `codex` / `claude-code` / `hermes-agent`) install their
# dependencies at rollout.
# `compact` (an example harness) and `terminus-2` (drives the host tmux) are excluded from e2e.
@pytest.fixture
def harness(request) -> HarnessConfig:
    config = {"id": request.param} if isinstance(request.param, str) else request.param
    return harness_config_type(config["id"]).model_validate(config)


def pytest_configure(config) -> None:
    """Self-launching tool servers run `python -m <module>` in a fresh subprocess, which
    inherits `PYTHONPATH` but not pytest's in-process `pythonpath`. Put the fixture dir on
    `PYTHONPATH` so a fixture server module (e.g. `tool_response_image_v1`)
    resolves there too — an installed example package (e.g. `glossary`) already would."""
    fixtures = str(Path(__file__).parent / "fixtures")
    existing = os.environ.get("PYTHONPATH", "")
    if fixtures not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            f"{fixtures}{os.pathsep}{existing}" if existing else fixtures
        )


def pytest_collection_modifyitems(config, items) -> None:
    """Skip the live-model tests (marked `e2e`) when no model endpoint is configured, so the
    rest of the suite (e.g. config parsing) still runs in a keyless environment."""
    if os.environ.get("PRIME_API_KEY"):
        return
    skip = pytest.mark.skip(reason="needs PRIME_API_KEY")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


def _configure_prime_runtimes(config: dict) -> None:
    """Configure every prime runtime config (nested — harness / tool): tag a `vf-ci` label
    for optional cleanup, and pin a region that supports port exposure."""
    if isinstance(config, dict):
        if config.get("type") == "prime":
            config.setdefault("labels", ["vf-ci"])
            # `us` is required for prime's port exposure, which a tool server hosted in a
            # sandbox needs to be reachable from outside it.
            config.setdefault("region", "us")
        for value in config.values():
            _configure_prime_runtimes(value)


def _eval_config(
    taskset: str,
    *,
    output_dir: Path,
    harness: str | HarnessConfig | None = "null",
    n: int = 1,
    num_tasks: int = 1,
    max_tokens: int = 2048,
    max_turns: int | None = 4,
    rollout_timeout: float = 180,
    taskset_overrides: dict | None = None,
    runtime: dict | None = None,
    env: dict | None = None,
    pool: dict | None = None,
    reasoning_effort: str | None = None,
    server: bool = False,
) -> EvalConfig:
    """Build the smallest `EvalConfig` that still exercises the path, shared by the in-process
    (`run_v1`) and env-server (`run_v1_server`) fixtures. `taskset_overrides` merges onto the
    `{id: ...}` config; `runtime` places the `agent` seat's harness (an agent field, not a
    harness one).

    `harness=None` leaves every seat on its own story — the multi-agent case: there
    is no run-level harness, so a single-agent test's `harness` lands on the `agent`
    seat and a multi-agent test pins its seats through `env` role fields instead."""
    taskset_cfg = {"id": taskset, **(taskset_overrides or {})}
    env_cfg = dict(env or {})
    _configure_prime_runtimes(taskset_cfg)
    if harness:
        env_cfg.setdefault("agent", {})["harness"] = (
            harness_config_type(harness)(id=harness)
            if isinstance(harness, str)
            else harness
        )
    if runtime:
        runtime_cfg = dict(runtime)
        _configure_prime_runtimes(runtime_cfg)
        env_cfg.setdefault("agent", {})["runtime"] = runtime_cfg
    retries = {"max_retries": 2, "include": ["ProviderError", "HarnessError"]}
    env_cfg.setdefault("retries", retries)
    # Per-run caps live on the seats: resolve the env's declared roles and cap
    # each one (a test's own seat dict wins over the shared defaults).
    config_cls = vf.env_config_type(taskset, env_cfg.get("id", ""))
    seats = [
        name
        for name, field in config_cls.model_fields.items()
        if isinstance(field.default, vf.AgentConfig)
    ]
    for seat in seats:
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
        num_tasks=num_tasks,
        num_rollouts=n,
        sampling={
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        },
        rich=None,
        serve=({"pool": pool} if pool else {}) if server else None,
        output_dir=output_dir.parent,
        run={"dir": output_dir.name},
        model=CI_MODEL,
    )


@pytest.fixture
def run_v1():
    """Run a v1 taskset end-to-end in-process (`run_eval` with `--no-serve`) and return
    its traces."""

    async def _run(taskset: str, **kwargs) -> list[Trace]:
        config = _eval_config(taskset, **kwargs)
        records = await run_eval(config)
        # The runner answers durability envelopes; the tests assert on traces.
        return [t for r in records for t in r.traces]

    return _run


@pytest.fixture
def run_v1_server():
    """Run a v1 taskset through the env-server worker pool (`run_eval`'s default path) —
    the path a CLI run and prime-rl training both take. Spawns the broker + a worker, so
    it's the only fixture that exercises serving resources (shared tool servers,
    interception pool) being stood up by the *server* rather than the in-process runner.
    Pinned to a single static worker for determinism."""

    async def _run(taskset: str, **kwargs) -> list[Trace]:
        kwargs.setdefault("pool", {"type": "static", "num_workers": 1})
        config = _eval_config(taskset, server=True, **kwargs)
        records = await run_eval(config)
        return [t for r in records for t in r.traces]

    return _run


@pytest.fixture
async def live_ctx():
    """The e2e `ModelContext` (default model + endpoint config, provider-default sampling)
    for driving `Agent` directly — the agent-surface counterpart of `run_v1`."""
    from verifiers.v1.clients import EvalClientConfig, ModelContext
    from verifiers.v1.types import SamplingConfig

    # Endpoint config only — each rollout builds and closes its own client.
    yield ModelContext(
        model=CI_MODEL,
        client=EvalClientConfig(),
        sampling=SamplingConfig(max_tokens=2048),
    )
