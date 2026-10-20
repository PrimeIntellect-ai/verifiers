"""The integration grid: which harnesses and runtimes a behavior runs across, and why a
cell is skipped.

Each behavior module fixes one task and fans it over `grid(...)`: every bundled harness
on Prime, the production runtime. The basic single- and multi-turn behaviors also run
on the host (`DEBUG_RUNTIMES`), which needs no container. Docker cells run on the CI
runner itself and make the jobs too slow, so the Docker runtime is covered by
`tests/unit/test_runtimes.py` instead. Every reason a cell cannot run derives
from a capability the code declares, never from a list of names, so the grid follows
the code.

A harness without a capability the behavior needs is collected as a skip with its
reason (`Harness.EXECUTES_CODE`, `SUPPORTS_MCP`, `SUPPORTS_SKILLS`, resume support via
`SUPPORTS_RESUME` or a native session).

A runtime without a feature the behavior needs (`RuntimeBackend.SUPPORTS_*`, or a
`NetworkPolicyConfig` for a network policy) skips the cell the same way. That a runtime
refuses an undeclared feature, with a message naming it, is tested in
`tests/unit/test_runtimes.py`.

Every cell carries its axes' marks, so `-m` selects subsets:

    uv run pytest tests/integration -n 8 -m "prime and codex"
    uv run pytest tests/integration -n 8 -m prime

`terminus-2` (drives the host tmux) and `browser-use` (needs its Playwright image, which
prime cannot pull) stay outside the grid.
"""

from collections.abc import Sequence
from typing import get_args

import pytest
from pydantic import TypeAdapter

import verifiers.v1 as vf
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import RuntimeConfig, backend_cls
from verifiers.v1.utils.loaders import harness_class, harness_config_type

HARNESSES: list[str | dict] = [
    "null",
    "bash",
    "rlm",
    "codex",
    "claude-code",
    "hermes-agent",
    "deepseek",
    "kimi-code",
    "pi",
    "openclaw",
    "opencode",
    "mini-swe-agent",
    "prime-agent",
]
DEBUG_RUNTIMES = ("subprocess",)
RUNTIMES = (*DEBUG_RUNTIMES, "prime")

# The runtime features a behavior can need, as `needs` names them.
RUNTIME_FEATURES = {
    "container": "task images",
    "user": "running as another user",
    "network": "network policies",
    "expose": "port exposure",
    "compose": "Compose services",
}

# Harnesses that cannot run against the CI model yet.
# TODO: drop each entry once its fix lands.
CI_MODEL_GAPS = {
    "kimi-code": "Prime chat completions rejects kimi-code's prompt_cache_key",
}


# Harnesses whose program rewrites earlier turns by design, so a rollout can record
# several branches. Each test names the ones it lets fork. Recheck on version bumps.
FORKS = {
    # Acts only through tool calls: it drops a tool-less reply from its history and
    # resends only its format error.
    "mini-swe-agent": "https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/agents/default.py#L114",
    # Moves its runtime-context message to the tail of every request and drops the
    # earlier one, unless the provider's replay policy is append-only.
    "openclaw": "https://github.com/openclaw/openclaw/blob/v2026.9.5/src/agents/internal-runtime-context.ts#L402-L408",
}


def assert_one_branch(
    trace: vf.Trace, harness: vf.HarnessConfig, forks: Sequence[str] = ()
) -> None:
    """A rollout records one branch, unless its harness is one of `forks` (see `FORKS`)."""
    assert set(forks) <= FORKS.keys(), (
        f"no FORKS reason for {set(forks) - FORKS.keys()}"
    )
    if harness.id not in forks:
        assert trace.num_branches == 1, f"{trace.num_branches} branches"


def agent_seats(taskset: str, env_id: str = "") -> list[str]:
    """The agent seats the env declares for `taskset`: `agent` for a plain taskset, its
    own roles for a recipe env."""
    config_cls = vf.env_config_type(taskset, env_id)
    return [
        name
        for name, field in config_cls.model_fields.items()
        if isinstance(field.default, vf.AgentConfig)
    ]


def harness_id(harness: str | dict) -> str:
    return harness if isinstance(harness, str) else harness["id"]


def mark(name: str) -> pytest.MarkDecorator:
    return getattr(pytest.mark, name.replace("-", "_"))


def runtime_config(runtime: str, harness: str | dict | None = None) -> dict:
    config: dict = {"type": runtime}
    program = harness is not None and harness_class(harness_id(harness)).NEEDS_CONTAINER
    # Prime's default 1 CPU / 2 GB is too small to install and run an agent CLI.
    if runtime == "prime" and program:
        config.update(cpu=2.0, memory=4.0)
    return config


def runtime_cls(runtime: str) -> type:
    return backend_cls(
        TypeAdapter(RuntimeConfig).validate_python(runtime_config(runtime))
    )


def supports(runtime: str, feature: str) -> bool:
    """Whether the runtime declares `feature`: a backend flag, or, for a network policy,
    a config that inherits `NetworkPolicyConfig`."""
    if feature == "network":
        config = TypeAdapter(RuntimeConfig).validate_python(runtime_config(runtime))
        return isinstance(config, NetworkPolicyConfig)
    flag = {
        "container": "SUPPORTS_TASK_IMAGES",
        "user": "SUPPORTS_USER_SWITCHING",
        "expose": "SUPPORTS_PORT_EXPOSURE",
        "compose": "SUPPORTS_COMPOSE",
    }[feature]
    return getattr(runtime_cls(runtime), flag)


def resumes(cls: type[Harness]) -> bool:
    """The same rule as `Agent.interaction`: transcript replay, a native resume, or a
    rollout-scoped session."""
    return (
        cls.SUPPORTS_RESUME
        or cls.session is not Harness.session
        or cls.resume is not Harness.resume
    )


def unsupported(
    harness: str | dict, runtime: str, needs: tuple[str, ...]
) -> str | None:
    """Why this cell cannot run, or None when it can: a capability the behavior needs
    that the harness or the runtime does not declare."""
    name = harness_id(harness)
    cls = harness_class(name)
    if "shell" in needs and not cls.EXECUTES_CODE:
        return f"{name} gives the model no shell"
    if "mcp" in needs and not cls.SUPPORTS_MCP:
        return f"{name} does not support MCP tools"
    if "skills" in needs and not cls.SUPPORTS_SKILLS:
        return f"{name} has no native skill support"
    if "resume" in needs and not resumes(cls):
        return f"{name} cannot resume an exchange"
    runtime_needs = (*(("container",) if cls.NEEDS_CONTAINER else ()), *needs)
    for feature in runtime_needs:
        if feature in RUNTIME_FEATURES and not supports(runtime, feature):
            return f"{runtime} does not support {RUNTIME_FEATURES[feature]}"
    return CI_MODEL_GAPS.get(name)


def cell_marks(
    harness: str | dict, runtime: str, needs: tuple[str, ...]
) -> list[pytest.MarkDecorator]:
    if reason := unsupported(harness, runtime, needs):
        return [pytest.mark.skip(reason=reason)]
    return []


def runtimes_grid(
    runtimes: Sequence[str] = ("prime",), *, needs: tuple[str, ...] = ()
) -> list:
    """`harness_runtime` cells for a behavior with no harness axis."""
    cells = []
    for runtime in runtimes:
        marks = [mark(runtime), *cell_marks("null", runtime, needs)]
        cells.append(pytest.param(runtime, marks=marks, id=f"harness-in-{runtime}"))
    return cells


def grid(
    harnesses: Sequence[str | dict] = HARNESSES,
    runtimes: Sequence[str] = ("prime",),
    *,
    needs: tuple[str, ...] = (),
) -> list:
    """`(harness, harness_runtime)` cells, each marked by its axes, skipped where the
    harness or the runtime rules the behavior out."""
    cells = []
    for harness in harnesses:
        for runtime in runtimes:
            marks = [
                mark(harness_id(harness)),
                mark(runtime),
                *cell_marks(harness, runtime, needs),
            ]
            cells.append(
                pytest.param(
                    harness,
                    runtime_config(runtime, harness),
                    marks=marks,
                    id=f"{harness_id(harness)}-in-{runtime}",
                )
            )
    return cells


def dialects(harness: str) -> tuple[str, ...]:
    """The dialects the harness config can select besides its default. Empty when the
    program speaks one dialect or picks its own."""
    field = harness_config_type(harness).model_fields["dialect"]
    return tuple(
        value
        for arg in get_args(field.annotation)
        for value in get_args(arg) or (arg,)
        if value != field.default
    )


def dialect_grid(
    runtimes: Sequence[str] = ("prime",), *, needs: tuple[str, ...] = ()
) -> list:
    """`grid` cells for every non-default `dialect` each harness config can select."""
    cells = []
    for harness in HARNESSES:
        name = harness_id(harness)
        for dialect in dialects(name):
            for cell in grid([{"id": name, "dialect": dialect}], runtimes, needs=needs):
                cells.append(
                    pytest.param(
                        *cell.values,
                        marks=cell.marks,
                        id=cell.id.replace(f"{name}-in-", f"{name}-{dialect}-in-"),
                    )
                )
    return cells


def skip_harness(cells: list, harness: str, reason: str) -> list:
    """Skip one harness's cells for a limit no capability flag declares. Each use names
    why, so a reader sees the gap and when it can go."""
    return [
        pytest.param(
            *cell.values,
            marks=[*cell.marks, pytest.mark.skip(reason=reason)],
            id=cell.id,
        )
        if harness_id(cell.values[0]) == harness
        else cell
        for cell in cells
    ]


def placements(
    harness_runtimes: tuple[str, ...] = ("prime",),
    tool_runtimes: tuple[str, ...] = ("colocated", "subprocess", "prime"),
    *,
    shared: bool = False,
) -> list:
    """`(harness_runtime, tool_runtime)` cells for a tool server placed colocated in the
    harness's box, in a Compose service beside it (`service`), or in its own runtime
    (on Prime beside a Prime harness, a VM on the rollout's private network). A
    `subprocess` tool runtime is a server on the host, reached from the sandbox through
    the tunnel. A `shared` server runs once per eval, so it joins no rollout's network."""
    cells = []
    for harness_runtime in harness_runtimes:
        for tool_runtime in tool_runtimes:
            marks = [mark(harness_runtime), mark(tool_runtime)]
            reason = None
            if tool_runtime == "service":
                reason = unsupported("null", harness_runtime, ("compose",))
            # Beside a Prime harness, a rollout's Prime server joins its private network.
            elif tool_runtime != "colocated" and not (
                not shared and tool_runtime == harness_runtime == "prime"
            ):
                reason = unsupported("null", tool_runtime, ("expose",))
            if reason:
                marks.append(pytest.mark.skip(reason=reason))
            cells.append(
                pytest.param(
                    harness_runtime,
                    tool_runtime,
                    marks=marks,
                    id=f"harness-in-{harness_runtime}-with-tool-"
                    + (
                        "colocated"
                        if tool_runtime == "colocated"
                        else f"in-{tool_runtime}"
                    ),
                )
            )
    return cells
