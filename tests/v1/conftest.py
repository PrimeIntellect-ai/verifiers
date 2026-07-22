"""Shared fixtures + helpers for the v1 end-to-end eval tests.

These tests run REAL eval runs (a live model endpoint, real runtimes) with the smallest
settings that still exercise the path, then assert on the resulting `Trace`(s) — they are
not unit tests of individual components. They need a model API key (`PRIME_API_KEY`);
without one the `e2e`-marked tests skip (config parsing still runs).

`run_v1` / `run_v0` mirror the eval CLI's two paths (`run_eval` for a v1 taskset,
`run_legacy_eval` for a v0 env). Placement coverage (harness x harness runtime x user/tool
server runtime) is PAIRWISE, not a full cross product: each test carries a curated list of
combinations (in test_e2e.py) that hits every axis value and the cross-boundary pairs with
distinct networking. The full cross bought flake exposure and CI minutes, not coverage — add
a combination to a test's list when it exercises a genuinely new reachability pair. The
placement fixtures below are indirect-only: they translate a parametrized value, and using
one without `indirect=True` fails loudly.

Every combination carries its axes' pytest marks, so subsets select with `-m`:

    uv run pytest tests/v1 -n auto                                # everything (needs modal setup)
    uv run pytest tests/v1 -n auto -m "not prime and not modal"  # the CI set (host + docker only)
    uv run pytest tests/v1 -n auto -m docker                      # any case touching the docker runtime
    uv run pytest tests/v1 -n auto -m bash                        # only the bash harness
    uv run pytest tests/v1 -n auto -m prime                       # only prime (real sandboxes; local)
    uv run pytest tests/v1 -n auto -m modal                       # only modal (needs local setup)

Marks: runtimes `subprocess` / `docker` / `prime` / `modal`, placement `colocated`,
harnesses `null` / `bash` / `rlm` / `kimi_code` / `codex` / `claude_code`. A mark is applied per
axis, so it selects every case touching that value on ANY axis; for one exact combination use `-k`
on the test id (e.g. `-k "harness-in-docker-with-tool-in-subprocess"`). prime/modal provision real
remote sandboxes (slow, infra-flaky, need setup), so they're local-only — CI runs
`-m "not prime and not modal"`.
"""

import os
from pathlib import Path

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.loaders import load_environment
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.trace import Trace

# Fixture tasksets/envs (echo-v1, echo-agentic-v1, echo-v0, echo-multi-v0) live in
# tests/v1/fixtures, added to the path via `pythonpath` in pyproject so the v1 loader and the
# v0 legacy bridge both resolve them by id (no install).

# The placement fixtures translate one parametrized value each; the combinations live on
# the tests (`indirect=True`), so coverage is a visible, curated list — never an implicit
# cross product.


@pytest.fixture
def harness_runtime(request) -> str:
    return request.param


@pytest.fixture
def user_runtime(request) -> dict:
    """A `taskset.task.user` override placing the user simulator: `colocated` (inside the harness's
    runtime) or its own runtime, by type."""
    if request.param == "colocated":
        return {"colocated": True}
    return {"colocated": False, "runtime": {"type": request.param}}


@pytest.fixture
def tool_runtime(request) -> dict:
    """A `taskset.task.tools` override placing the tool server: `colocated` (inside the harness's
    runtime) or its own runtime, by type."""
    if request.param == "colocated":
        return {"colocated": True}
    return {"runtime": {"type": request.param}}


# Built-in harnesses are bundled in the `harnesses` package; the agent CLIs (`rlm` /
# `kimi-code` / `codex` / `claude-code`) install their dependencies at rollout. `compact` (an
# example harness) and `terminus-2` (drives the host tmux) are excluded from e2e.
@pytest.fixture
def harness(request) -> str:
    return request.param


def pytest_configure(config) -> None:
    """Self-launching tool/user servers run `python -m <module>` in a fresh subprocess, which
    inherits `PYTHONPATH` but not pytest's in-process `pythonpath`. Put the fixture dir on
    `PYTHONPATH` so a fixture server module (e.g. `echo_user_sim_v1`, `tool_response_image_v1`)
    resolves there too — an installed example package (e.g. `glossary_v1`) already would."""
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
    """Configure every prime runtime config (nested — harness / tool / user): tag a `vf-ci` label
    for optional cleanup, and pin a region that supports port exposure."""
    if isinstance(config, dict):
        if config.get("type") == "prime":
            config.setdefault("labels", ["vf-ci"])
            # `us` is required for prime's port exposure, which a tool/user server hosted in a
            # sandbox needs to be reachable from outside it.
            config.setdefault("region", "us")
        for value in config.values():
            _configure_prime_runtimes(value)


def _eval_config(
    taskset: str,
    *,
    output_dir: Path,
    harness: str | None = "null",
    n: int = 1,
    num_tasks: int = 1,
    max_tokens: int = 2048,
    max_turns: int | None = 4,
    rollout_timeout: float = 180,
    taskset_overrides: dict | None = None,
    harness_overrides: dict | None = None,
    env: dict | None = None,
    pool: dict | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> EvalConfig:
    """Build the smallest `EvalConfig` that still exercises the path, shared by the in-process
    (`run_v1`) and env-server (`run_v1_server`) fixtures. `taskset_overrides` / `harness_overrides`
    are merged onto the `{id: ...}` config (placement, runtime, etc.); `model` overrides the default
    text model (e.g. a VLM for an image task).

    `temperature=0` (greedy) makes the run reproducible; `max_tokens` is generous headroom,
    not a target — these trivial tasks finish in a few hundred tokens, so capping tighter only
    risks truncating the reasoning before the answer (which tanks the reward).

    `harness=None` leaves every seat on its own story — the multi-agent case: there
    is no run-level harness, so a single-agent test's `harness` lands on the `agent`
    seat and a multi-agent test pins its seats through `env` role fields instead."""
    taskset_cfg = {"id": taskset, **(taskset_overrides or {})}
    env_cfg = dict(env or {})
    _configure_prime_runtimes(taskset_cfg)
    if harness:
        harness_cfg = {"id": harness, **(harness_overrides or {})}
        _configure_prime_runtimes(harness_cfg)
        env_cfg.setdefault("agent", {})["harness"] = harness_cfg
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
        # Flake resilience: retries are per-agent now (flat RetryConfig).
        seat_cfg.setdefault("retries", {"max_retries": 2, "include": ["ProviderError"]})
    return EvalConfig(
        env={
            "taskset": taskset_cfg,
            **env_cfg,
        },
        num_tasks=num_tasks,
        num_rollouts=n,
        sampling={
            "max_tokens": max_tokens,
            "temperature": 0,
            "reasoning_effort": reasoning_effort,
        },
        rich=False,
        output_dir=output_dir,
        **({"pool": pool} if pool else {}),
        **({"model": model} if model else {}),
    )


@pytest.fixture
def run_v1():
    """Run a v1 taskset end-to-end in-process (`run_eval`, the `--rich` CLI path) and return
    its traces."""

    async def _run(taskset: str, **kwargs) -> list[Trace]:
        config = _eval_config(taskset, **kwargs)
        records = await run_eval(load_environment(config.env), config)
        # The runner answers durability envelopes; the tests assert on traces.
        return [t for r in records for t in r.traces]

    return _run


@pytest.fixture
def run_v1_server():
    """Run a v1 taskset through the env-server worker pool (`run_eval_server`) — the path a
    `--server` CLI run and prime-rl training both take. Spawns the broker + a worker, so it's
    the only fixture that exercises serving resources (shared tool servers, interception pool)
    being stood up by the *server* rather than the in-process runner. Pinned to a single static
    worker for determinism."""
    from verifiers.v1.cli.eval.runner import run_eval_server

    async def _run(taskset: str, **kwargs) -> list[Trace]:
        kwargs.setdefault("pool", {"type": "static", "num_workers": 1})
        config = _eval_config(taskset, **kwargs)
        records = await run_eval_server(config)
        return [t for r in records for t in r.traces]

    return _run


@pytest.fixture
def run_v0():
    """Run a legacy v0 env through the v1 bridge (the eval CLI's `--id` path)."""
    from verifiers.v1.legacy import run_legacy_eval

    async def _run(
        env_id: str,
        *,
        output_dir: Path,
        n: int = 1,
        max_tokens: int = 2048,
        args: dict | None = None,
    ) -> list[Trace]:
        config = EvalConfig(
            id=env_id,
            args=args or {},
            num_tasks=1,
            num_rollouts=n,
            sampling={"max_tokens": max_tokens, "temperature": 0},
            rich=False,
            output_dir=output_dir,
        )
        records = await run_legacy_eval(config)
        return [t for r in records for t in r.traces]

    return _run
