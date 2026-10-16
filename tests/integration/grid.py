"""The integration grid: which harnesses and runtimes a behavior runs across, and why a
cell is skipped.

Each behavior module fixes one task and fans it over `grid(...)`: every bundled harness
on Prime, the production runtime. The basic single- and multi-turn behaviors also run
on the local debug runtimes (`DEBUG_RUNTIMES`). Every reason a cell cannot run derives
from a capability the code declares, never from a list of names, so the grid follows
the code.

A harness without a capability the behavior needs is collected as a skip with its
reason (`Harness.EXECUTES_CODE`, `SUPPORTS_MCP`, `SUPPORTS_SKILLS`, resume support via
`SUPPORTS_RESUME` or a native session).

A runtime without a capability the behavior needs still runs, marked `refused`: the run
must fail with the framework's message for that gap, which the `run` fixture checks.
The runtime gaps are `RuntimeBackend.SUPPORTS_TASK_IMAGES` (for `Harness.NEEDS_CONTAINER`) on
the debug runtimes, and `SUPPORTS_PORT_EXPOSURE` for a tool server in its own Prime sandbox.

Every cell carries its axes' marks, so `-m` selects subsets:

    uv run pytest tests/integration -n 8 -m "docker and codex"
    uv run pytest tests/integration -n 8 -m prime

`terminus-2` (drives the host tmux) and `browser-use` (needs its Playwright image, which
prime cannot pull) stay outside the grid.
"""

from collections.abc import Sequence

import pytest
from pydantic import TypeAdapter

import verifiers.v1 as vf
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import RuntimeConfig, backend_cls
from verifiers.v1.utils.loaders import harness_class

HARNESSES: list[str | dict] = [
    "null",
    "bash",
    "rlm",
    "codex",
    "claude-code",
    "hermes-agent",
    "kimi-code",
    "pi",
    "openclaw",
    "opencode",
    "mini-swe-agent",
    "prime-agent",
]
DEBUG_RUNTIMES = ("subprocess", "docker")
RUNTIMES = (*DEBUG_RUNTIMES, "prime")

# The framework's message for each runtime gap; a refused cell must fail with it.
REFUSALS = {
    "container": "needs a container runtime",
    "user": "cannot run as another user",
    "network": "does not enforce network policies",
    "expose": "cannot expose a port",
}

# Harnesses that cannot run against the CI model yet.
# TODO: drop each entry once its fix lands.
CI_MODEL_GAPS = {
    "kimi-code": "Prime chat completions rejects kimi-code's prompt_cache_key",
    "openclaw": "openclaw cannot load the custom `internal/` model provider",
}


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


def runtime_supports_network_policy(runtime: str) -> bool:
    config = TypeAdapter(RuntimeConfig).validate_python(runtime_config(runtime))
    return isinstance(config, NetworkPolicyConfig)


def resumes(cls: type[Harness]) -> bool:
    """The same rule as `Agent.interaction`: transcript replay, a native resume, or a
    rollout-scoped session."""
    return (
        cls.SUPPORTS_RESUME
        or cls.session is not Harness.session
        or cls.resume is not Harness.resume
    )


def unsupported(harness: str | dict, needs: tuple[str, ...]) -> str | None:
    """Why the harness cannot run this behavior, or None when it can."""
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
    return CI_MODEL_GAPS.get(name)


def refused(harness: str | dict, runtime: str, needs: tuple[str, ...]) -> list[str]:
    """The gaps the runtime may refuse this cell with; the first check to run wins."""
    cls = runtime_cls(runtime)
    gaps = []
    if (
        harness_class(harness_id(harness)).NEEDS_CONTAINER
        and not cls.SUPPORTS_TASK_IMAGES
    ):
        gaps.append("container")
    if "user" in needs and not cls.SUPPORTS_USER_SWITCHING:
        gaps.append("user")
    if "network" in needs and not runtime_supports_network_policy(runtime):
        gaps.append("network")
    return gaps


def cell_marks(
    harness: str | dict, runtime: str, needs: tuple[str, ...]
) -> list[pytest.MarkDecorator]:
    if reason := unsupported(harness, needs):
        return [pytest.mark.skip(reason=reason)]
    if gaps := refused(harness, runtime, needs):
        return [pytest.mark.refused("|".join(REFUSALS[gap] for gap in gaps))]
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
    harness rules the behavior out, and refused where the runtime does."""
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
) -> list:
    """`(harness_runtime, tool_runtime)` cells for a tool server placed colocated in the
    harness's box or in its own runtime. A `subprocess` tool runtime is a server on the
    host, reached from the sandbox through the tunnel."""
    cells = []
    for harness_runtime in harness_runtimes:
        for tool_runtime in tool_runtimes:
            marks = [mark(harness_runtime), mark(tool_runtime)]
            if (
                tool_runtime != "colocated"
                and not runtime_cls(tool_runtime).SUPPORTS_PORT_EXPOSURE
            ):
                marks.append(pytest.mark.refused(REFUSALS["expose"]))
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
