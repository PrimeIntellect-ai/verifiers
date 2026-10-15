"""The integration grid: which harnesses and runtimes a behavior runs across, and why a
cell is skipped.

Each behavior module fixes one task and fans it over `grid(...)`: every bundled harness
x every runtime kind. The harness grid runs one local container engine (`docker`);
`podman` / `apptainer` share its container runtime code and are exercised by the
runtime-only grids (`placements`, `runtimes_grid`), where the harness is fixed. A cell
the stack cannot run is collected as a skip with its reason. Every reason derives from a
capability the code declares, never from a list of names, so the grid follows the code:

- `Harness.NEEDS_CONTAINER`: the harness never runs on the host (`subprocess`)
- `Harness.EXECUTES_CODE`, `SUPPORTS_MCP`, `SUPPORTS_SKILLS`, resume support
  (`SUPPORTS_RESUME` or a native session): the behavior needs a shell, MCP tools,
  skills, or a resumable exchange
- `RuntimeBackend.supports_user`, `RuntimeBackend.supports_expose`, `NetworkPolicyConfig`: the
  behavior needs a user switch, a tool server reachable from another runtime, or a
  network policy
- `podman` / `apptainer` need their engine on the host

CI deselects `podman`, `apptainer`, `modal` and `e2b` by mark: local engines and external
sandboxes.

Every cell carries its axes' marks, so `-m` selects subsets:

    uv run pytest tests/integration -n 8 -m "not podman and not apptainer and not modal and not e2b"  # CI
    uv run pytest tests/integration -n 8 -m "docker and codex"
    uv run pytest tests/integration -n 8 -m e2b                                                      # local

`terminus-2` (drives the host tmux) and `browser-use` (needs its Playwright image, which
prime cannot pull) stay outside the grid.
"""

import shutil
from collections.abc import Sequence

import pytest
from pydantic import TypeAdapter

import verifiers.v1 as vf
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import RuntimeConfig, _backend_cls
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
    "mini-swe-agent",
    "prime-agent",
]
RUNTIMES = ["subprocess", "docker", "podman", "apptainer", "prime", "modal", "e2b"]
ENGINES = ("podman", "apptainer")
HARNESS_RUNTIMES = [runtime for runtime in RUNTIMES if runtime not in ENGINES]

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
    # The default sandboxes are too small to install and run an agent CLI: e2b's box
    # starves rlm's ACP startup, and prime's 1 CPU / 2 GB gets openclaw terminated.
    if runtime == "e2b":
        config["memory"] = 2.0
    if runtime == "prime" and program:
        config.update(cpu=2.0, memory=4.0)
    return config


def runtime_cls(runtime: str) -> type:
    return _backend_cls(TypeAdapter(RuntimeConfig).validate_python({"type": runtime}))


def runtime_supports_user(runtime: str) -> bool:
    return runtime_cls(runtime).supports_user


def runtime_supports_network_policy(runtime: str) -> bool:
    config = TypeAdapter(RuntimeConfig).validate_python({"type": runtime})
    return isinstance(config, NetworkPolicyConfig)


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
    """Why this cell cannot run, or None when it can."""
    name = harness_id(harness)
    cls = harness_class(name)
    if runtime in ENGINES and shutil.which(runtime) is None:
        return f"needs {runtime} on the host"
    if runtime == "subprocess" and cls.NEEDS_CONTAINER:
        return f"{name} needs a container runtime"
    if "shell" in needs and not cls.EXECUTES_CODE:
        return f"{name} gives the model no shell"
    if "mcp" in needs and not cls.SUPPORTS_MCP:
        return f"{name} does not support MCP tools"
    if "skills" in needs and not cls.SUPPORTS_SKILLS:
        return f"{name} has no native skill support"
    if "resume" in needs and not resumes(cls):
        return f"{name} cannot resume an exchange"
    if "user" in needs and not runtime_supports_user(runtime):
        return f"{runtime} cannot run commands as another user"
    if "network" in needs and not runtime_supports_network_policy(runtime):
        return f"{runtime} has no network policy"
    return None


def runtimes_grid(
    runtimes: Sequence[str] = RUNTIMES, *, needs: tuple[str, ...] = ()
) -> list:
    """`harness_runtime` cells for a behavior with no harness axis."""
    cells = []
    for runtime in runtimes:
        marks = [mark(runtime)]
        if reason := unsupported("null", runtime, needs):
            marks.append(pytest.mark.skip(reason=reason))
        cells.append(pytest.param(runtime, marks=marks, id=f"harness-in-{runtime}"))
    return cells


def grid(
    harnesses: Sequence[str | dict] = HARNESSES,
    runtimes: Sequence[str] = HARNESS_RUNTIMES,
    *,
    needs: tuple[str, ...] = (),
) -> list:
    """`(harness, harness_runtime)` cells, each marked by its axes and skipped with a
    reason where the behavior's `needs` or the runtime rule it out."""
    cells = []
    for harness in harnesses:
        for runtime in runtimes:
            marks = [mark(harness_id(harness)), mark(runtime)]
            reason = unsupported(harness, runtime, needs) or CI_MODEL_GAPS.get(
                harness_id(harness)
            )
            if reason:
                marks.append(pytest.mark.skip(reason=reason))
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
    harness_runtimes: tuple[str, ...] = tuple(RUNTIMES),
    tool_runtimes: tuple[str, ...] = ("colocated", *RUNTIMES),
) -> list:
    """`(harness_runtime, tool_runtime)` cells for a tool server placed colocated in the
    harness's box or in its own runtime."""
    cells = []
    for harness_runtime in harness_runtimes:
        for tool_runtime in tool_runtimes:
            marks = [mark(harness_runtime), mark(tool_runtime)]
            missing = [
                engine
                for engine in {harness_runtime, tool_runtime} & set(ENGINES)
                if shutil.which(engine) is None
            ]
            if missing:
                marks.append(pytest.mark.skip(reason=f"needs {missing[0]} on the host"))
            elif (
                tool_runtime != "colocated"
                and not runtime_cls(tool_runtime).supports_expose
            ):
                marks.append(
                    pytest.mark.skip(
                        reason=f"{tool_runtime} cannot expose a port; a tool server "
                        "there is reachable only colocated"
                    )
                )
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
