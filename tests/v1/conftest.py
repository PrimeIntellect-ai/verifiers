"""Shared fixtures + helpers for the v1 end-to-end eval tests.

These tests run REAL eval runs (a live model endpoint, real runtimes) with the smallest
settings that still exercise the path, then assert on the resulting `Trace`(s) — they are
not unit tests of individual components. They need a model API key (`PRIME_API_KEY`);
without one the `e2e`-marked tests skip (config parsing still runs).

`run_v1` / `run_v0` mirror the eval CLI's two paths (`run_eval` for a v1 taskset,
`run_legacy_eval` for a v0 env). The `runtime` fixture fans a test out across the built-in
runtimes (modal excluded for now); docker/prime are marked so they deselect cleanly
(`-m "not slow"`).
"""

import os
from pathlib import Path

import pytest

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.env import Environment
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.trace import Trace

# Fixture tasksets/envs (echo-v1, echo-agentic-v1, echo-v0, echo-multi-v0) live in
# tests/v1/fixtures, added to the path via `pythonpath` in pyproject so the v1 loader and the
# v0 legacy bridge both resolve them by id (no install).

# The harness runtime, modal excluded. docker needs the daemon; prime provisions real
# sandboxes + tunnels (network + PRIME credentials), so both are marked to deselect easily. The
# `id`s make a test read like `<harness>-harness-in-<rt>` / `in-<rt>-with-<user|tool>-...`.
HARNESS_RUNTIMES = [
    pytest.param("subprocess", id="in-subprocess"),
    pytest.param("docker", marks=pytest.mark.slow, id="in-docker"),
    pytest.param("prime", marks=[pytest.mark.slow, pytest.mark.prime], id="in-prime"),
]


@pytest.fixture(params=HARNESS_RUNTIMES)
def harness_runtime(request) -> str:
    return request.param


# The user simulator's runtime: inside the harness's runtime (`colocated`) or its own runtime; this
# fans the user test across both (reusing the runtime markers for the own-runtime cases).
USER_RUNTIMES = [
    pytest.param("colocated", id="with-user-colocated"),
    pytest.param("subprocess", id="with-user-in-subprocess"),
    pytest.param("docker", marks=pytest.mark.slow, id="with-user-in-docker"),
    pytest.param(
        "prime",
        marks=[pytest.mark.slow, pytest.mark.prime],
        id="with-user-in-prime",
    ),
]


@pytest.fixture(params=USER_RUNTIMES)
def user_runtime(request) -> dict:
    """A `taskset.user` override placing the user simulator: `colocated` (inside the harness's
    runtime) or its own runtime, by type."""
    if request.param == "colocated":
        return {"colocated": True}
    return {"colocated": False, "runtime": {"type": request.param}}


# The tool server's runtime: inside the harness's runtime (`colocated`), shared once per eval, or its
# own runtime per rollout; this fans the tool test across all of them (runtime markers for the
# own-runtime cases — colocated/shared use the host subprocess runtime).
TOOL_RUNTIMES = [
    pytest.param("colocated", id="with-tool-colocated"),
    pytest.param("shared", id="with-tool-shared"),
    pytest.param("subprocess", id="with-tool-in-subprocess"),
    pytest.param("docker", marks=pytest.mark.slow, id="with-tool-in-docker"),
    pytest.param(
        "prime",
        marks=[pytest.mark.slow, pytest.mark.prime],
        id="with-tool-in-prime",
    ),
]


@pytest.fixture(params=TOOL_RUNTIMES)
def tool_runtime(request) -> dict:
    """A `taskset.tools` override placing the tool server: `colocated` (inside the harness's
    runtime), `shared` (one instance for the whole eval), or its own runtime, by type."""
    if request.param == "colocated":
        return {"colocated": True}
    if request.param == "shared":
        return {"shared": True}
    return {"runtime": {"type": request.param}}


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


# Built-in harnesses (bundled in the `harnesses` package), composed with `harness_runtime` for the
# plain-task harness x runtime matrix. compact is an example harness, not built-in, so it's
# excluded. Agent CLI harnesses install their dependencies at rollout, so they're marked slow.
@pytest.fixture(
    params=[
        pytest.param("default", id="default-harness"),
        pytest.param("rlm", marks=pytest.mark.slow, id="rlm-harness"),
        pytest.param("kimi-code", marks=pytest.mark.slow, id="kimi-code-harness"),
    ]
)
def harness(request) -> str:
    return request.param


# `bash` (and `codex`) live here, not in `harness`: a bash/CLI agent on a no-op chat task
# (`test_single_turn`'s echo) may run a shell command or complete its loop without replying, which
# is flaky; on a task with a concrete action (writing a file) it's reliable. `bash` is the built-in
# chat loop + a bash tool, so it installs nothing and runs fast (not marked slow).
@pytest.fixture(
    params=[
        pytest.param("bash", id="bash-harness"),
        pytest.param(
            "mini-swe-agent", marks=pytest.mark.slow, id="mini-swe-agent-harness"
        ),
        pytest.param("codex", marks=pytest.mark.slow, id="codex-harness"),
    ]
)
def agentic_harness(request) -> str:
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
    if os.environ.get("PRIME_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return
    skip = pytest.mark.skip(
        reason="needs a model API key (PRIME_API_KEY / OPENAI_API_KEY)"
    )
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


def _eval_config(
    taskset: str,
    *,
    output_dir: Path,
    harness: str = "default",
    n: int = 1,
    num_tasks: int = 1,
    max_tokens: int = 2048,
    max_turns: int | None = 4,
    rollout_timeout: float = 180,
    taskset_overrides: dict | None = None,
    harness_overrides: dict | None = None,
    pool: dict | None = None,
    model: str | None = None,
) -> EvalConfig:
    """Build the smallest `EvalConfig` that still exercises the path, shared by the in-process
    (`run_v1`) and env-server (`run_v1_server`) fixtures. `taskset_overrides` / `harness_overrides`
    are merged onto the `{id: ...}` config (placement, runtime, etc.); `model` overrides the default
    text model (e.g. a VLM for an image task).

    `temperature=0` (greedy) makes the run reproducible; `max_tokens` is generous headroom,
    not a target — these trivial tasks finish in a few hundred tokens, so capping tighter only
    risks truncating the reasoning before the answer (which tanks the reward)."""
    return EvalConfig(
        taskset={"id": taskset, **(taskset_overrides or {})},
        harness={"id": harness, **(harness_overrides or {})},
        num_tasks=num_tasks,
        num_rollouts=n,
        max_turns=max_turns,
        max_output_tokens=max_tokens,
        sampling={"max_tokens": max_tokens, "temperature": 0},
        timeout={"rollout": rollout_timeout, "scoring": 60},
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
        return await run_eval(Environment(config), config)

    return _run


@pytest.fixture
def run_v1_server():
    """Run a v1 taskset through the env-server worker pool (`run_eval_server`) — the path a
    non-`--rich` CLI run and prime-rl training both take. Spawns the broker + a worker, so it's
    the only fixture that exercises serving resources (shared tool servers, interception pool)
    being stood up by the *server* rather than the in-process runner. Pinned to a single static
    worker for determinism."""
    from verifiers.v1.cli.eval.runner import run_eval_server

    async def _run(taskset: str, **kwargs) -> list[Trace]:
        kwargs.setdefault("pool", {"type": "static", "num_workers": 1})
        config = _eval_config(taskset, **kwargs)
        return await run_eval_server(config)

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
