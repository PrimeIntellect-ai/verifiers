"""Shared fixtures + helpers for the v1 end-to-end eval tests.

These tests run REAL eval runs (a live model endpoint, real runtimes) with the smallest
settings that still exercise the path, then assert on the resulting `Trace`(s) — they are
not unit tests of individual components. They need a model API key (`PRIME_API_KEY`);
Claude Code additionally uses `OPENROUTER_API_KEY`.

`run_v1` / `run_v0` mirror the eval CLI's two paths (`run_eval` for a v1 taskset,
`run_legacy_eval` for a v0 env). The `runtime` fixture fans a test out across the built-in
runtimes (modal excluded for now); docker/prime are marked so they deselect cleanly
(`-m "not slow"`).
"""

import os
from pathlib import Path

import pytest

from verifiers.v1.clients import EvalClientConfig
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.env import Environment
from verifiers.v1.cli.runner import run_eval
from verifiers.v1.trace import Trace

# Fixture tasksets/envs (echo-v1, echo-agentic-v1, echo-v0, echo-multi-v0) live in
# tests/v1/fixtures, added to the path via `pythonpath` in pyproject so the v1 loader and the
# v0 legacy bridge both resolve them by id (no install).

# Built-in runtimes, modal excluded. docker needs the daemon; prime provisions real
# sandboxes + tunnels (network + PRIME credentials), so both are marked to deselect easily.
RUNTIMES = [
    "subprocess",
    pytest.param("docker", marks=pytest.mark.slow),
    pytest.param("prime", marks=[pytest.mark.slow, pytest.mark.prime]),
]


@pytest.fixture(params=RUNTIMES)
def runtime(request) -> str:
    return request.param


# The runtime a tool / user-sim server runs in (its OWN, not colocated in the agent's), so the
# task-tools and user-sim tests can cover every runtime independent of the agent's.
@pytest.fixture(params=RUNTIMES)
def server_runtime(request) -> str:
    return request.param


@pytest.fixture
def skip_if_unexposable():
    """Skip when a trace failed because the server's runtime couldn't publish its port to the
    host — a prime sandbox whose region doesn't support port exposure (a known infra limit, not
    a code bug). subprocess/docker share the host network, so they never hit this.

    TODO: re-enable the prime cases once prime supports port exposure in all regions (or the
    runtime publishes the port via an in-sandbox tunnel)."""

    def _skip(trace) -> None:
        if any("port exposure" in str(e) for e in trace.errors):
            pytest.skip(
                "runtime can't publish a port to the host (pin a prime region that supports port exposure)"
            )

    return _skip


# Built-in harnesses (bundled in the `harnesses` package), composed with `runtime` for the
# harness x runtime matrix. compact is an example harness, not built-in, so it's excluded. rlm
# and the CLI harnesses install an agent binary at rollout, so they're marked slow.
@pytest.fixture(
    params=[
        "default",
        pytest.param("rlm", marks=pytest.mark.slow),
        pytest.param("codex", marks=pytest.mark.slow),
        pytest.param("claude-code", marks=pytest.mark.slow),
    ]
)
def harness(request) -> str:
    return request.param


@pytest.fixture
def harness_supports():
    """Read a capability flag (e.g. `SUPPORTS_TASK_TOOLS`, `SUPPORTS_USER_SIM`) off an harness
    by id — the matrix tests use it to decide whether a harness/task pairing should run, be
    rejected, or be skipped."""
    from verifiers.v1.loaders import load_harness

    def _supports(harness_id: str, flag: str) -> bool:
        harness = load_harness(
            EvalConfig.model_validate({"harness": {"id": harness_id}}).harness
        )
        return getattr(harness, flag)

    return _supports


def pytest_collection_modifyitems(config, items) -> None:
    """Skip the live-model tests (marked `e2e`) when no model endpoint is configured, so the
    rest of the suite (e.g. config parsing) still runs in a keyless environment."""
    has_default_key = bool(
        os.environ.get("PRIME_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )
    has_openrouter_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    for item in items:
        if "e2e" not in item.keywords:
            continue
        callspec = getattr(item, "callspec", None)
        is_claude = callspec and callspec.params.get("harness") == "claude-code"
        if is_claude and not has_openrouter_key:
            item.add_marker(pytest.mark.skip(reason="needs OPENROUTER_API_KEY"))
        elif not is_claude and not has_default_key:
            item.add_marker(
                pytest.mark.skip(
                    reason="needs a model API key (PRIME_API_KEY / OPENAI_API_KEY)"
                )
            )


@pytest.fixture
def run_v1():
    """Run a v1 taskset end-to-end (the eval CLI's native path) and return its traces.

    `temperature=0` (greedy) makes the run reproducible; `max_tokens` is generous headroom,
    not a target — these trivial tasks finish in a few hundred tokens, so capping tighter only
    risks truncating the reasoning before the answer (which tanks the reward)."""

    async def _run(
        taskset: str,
        *,
        runtime: str,
        output_dir: Path,
        harness: str = "default",
        n: int = 1,
        max_tokens: int = 2048,
        max_turns: int | None = 4,
        rollout_timeout: float = 180,
        enable_bash: bool = False,
        taskset_overrides: dict | None = None,
    ) -> list[Trace]:
        harness_config: dict = {"id": harness, "runtime": {"type": runtime}}
        if enable_bash:
            harness_config["enable_bash"] = True
        eval_overrides = {}
        if harness == "claude-code":
            eval_overrides["client"] = EvalClientConfig(
                base_url="https://openrouter.ai/api",
                api_key_var="OPENROUTER_API_KEY",
            )
        config = EvalConfig(
            **eval_overrides,
            taskset={"id": taskset, **(taskset_overrides or {})},
            harness=harness_config,
            num_tasks=1,
            num_rollouts=n,
            max_turns=max_turns,
            max_output_tokens=max_tokens,
            sampling={"max_tokens": max_tokens, "temperature": 0},
            timeout={"rollout": rollout_timeout, "scoring": 60},
            rich=False,
            output_dir=output_dir,
        )
        return await run_eval(Environment(config), config)

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
        return await run_legacy_eval(config)

    return _run
