"""The integration grid: which harnesses and runtimes a behavior runs across, and why a
cell is skipped.

Each behavior module fixes one task and fans it over `grid(...)`: every bundled harness
x every runtime kind. The harness grid runs one local container engine (`docker`);
`podman` / `apptainer` share its container runtime code and are exercised by the
runtime-only grids (`placements`, `runtimes_grid`), where the harness is fixed. A cell
the stack cannot run is collected as a skip with its reason, so the suite states what is
supported, what is not, and why:

- a `NEEDS_CONTAINER` harness (every agent program) never runs on the host (`subprocess`)
- `podman` / `apptainer` need their engine on the host; CI deselects their marks
- `modal` / `e2b` are external sandboxes: local-only, CI deselects their marks
- `prime` provisions real sandboxes and runs in CI with its secrets; it exposes no
  ports, so a tool server lives there only colocated with its harness
- a behavior's own needs: a reply to a chat task (`mini-swe-agent` only submits from a
  shell task), a shell (`EXECUTES_CODE`; `browser-use` drives a browser instead), MCP
  tools (`SUPPORTS_MCP`), a user that can be switched to (`supports_user`), or an
  exchange that resumes (`SUPPORTS_RESUME` or a native session)
- `browser-use` needs a browser image, which the grid pins on docker only

Every cell carries its axes' marks, so `-m` selects subsets:

    uv run pytest tests/integration -n 8 -m "not podman and not apptainer and not modal and not e2b"  # CI
    uv run pytest tests/integration -n 8 -m "docker and codex"
    uv run pytest tests/integration -n 8 -m e2b                                                      # local

`terminus-2` (drives the host tmux) stays outside the grid.
"""

import shutil
from collections.abc import Sequence

import pytest
from pydantic import TypeAdapter

from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import RuntimeConfig, _runtime_cls
from verifiers.v1.utils.loaders import harness_class

HARNESSES: list[str | dict] = [
    "null",
    "bash",
    "rlm",
    "codex",
    "claude-code",
    "hermes-agent",
    {"id": "kimi-code", "transport": "responses"},
    {"id": "pi", "transport": "responses"},
    "openclaw",
    "mini-swe-agent",
    "prime-agent",
    "browser-use",
]
BROWSER_IMAGE = (
    "mcr.microsoft.com/playwright/python:v1.61.0-noble@"
    "sha256:a9731514f24121d1dcd25d58d0a38146646d290a5998fd80d3e533e7b5e21c69"
)
RUNTIMES = ["subprocess", "docker", "podman", "apptainer", "prime", "modal", "e2b"]
ENGINES = ("podman", "apptainer")
EXTERNAL = ("modal", "e2b")
HARNESS_RUNTIMES = [runtime for runtime in RUNTIMES if runtime not in ENGINES]


def harness_id(harness: str | dict) -> str:
    return harness if isinstance(harness, str) else harness["id"]


def mark(name: str) -> pytest.MarkDecorator:
    return getattr(pytest.mark, name.replace("-", "_"))


def runtime_config(runtime: str, harness: str | dict | None = None) -> dict:
    config: dict = {"type": runtime}
    if runtime == "e2b":
        config["memory"] = 2.0  # the default box is too small to install an agent CLI
    if harness is not None and harness_id(harness) == "browser-use":
        config["image"] = BROWSER_IMAGE
    return config


def runtime_supports_user(runtime: str) -> bool:
    config = TypeAdapter(RuntimeConfig).validate_python({"type": runtime})
    return _runtime_cls(config).supports_user


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
    if name == "browser-use" and runtime != "docker":
        return f"{name} needs a browser image, pinned on docker only"
    if "reply" in needs and name == "mini-swe-agent":
        return f"{name} finishes by submitting a shell task, not by replying"
    if "shell" in needs and not cls.EXECUTES_CODE:
        return f"{name} gives the model no shell"
    if "shell" in needs and name == "browser-use":
        return f"{name} drives a browser, not a shell"
    if "mcp" in needs and not cls.SUPPORTS_MCP:
        return f"{name} does not support MCP tools"
    if "resume" in needs and not resumes(cls):
        return f"{name} cannot resume an exchange"
    if "user" in needs and not runtime_supports_user(runtime):
        return f"{runtime} cannot run commands as another user"
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
            if reason := unsupported(harness, runtime, needs):
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
            elif tool_runtime == "prime":
                marks.append(
                    pytest.mark.skip(
                        reason="prime VM sandboxes expose no ports; a tool server "
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
