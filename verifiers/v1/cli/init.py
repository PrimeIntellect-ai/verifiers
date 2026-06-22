"""The init entrypoint: `uv run init <name> [--add-tool] [--add-user] [--add-harness]`.

Registered as the `init` console script — the v1 sibling of v0's `vf-init`. It scaffolds a new
environment package under `--path` (default `./environments`), following the layout of the
shipped `environments/*_v1` examples: a `pyproject.toml`, a package whose `__init__.py` re-exports
the plugin via `__all__`, and a `taskset.py` that runs out of the box (replace `load_tasks` and
the `@reward`). The optional flags add more scaffolding — a `vf.Toolset` (`--add-tool`), a
`vf.User` simulator (`--add-user`), and a custom `vf.Harness` (`--add-harness`). `--v0` scaffolds
a legacy v0 `load_environment` package instead (via `verifiers.scripts.init`).
"""

import sys
from pathlib import Path

from pydantic_config import cli

from verifiers.v1.configs.init import InitConfig

USAGE = (
    "usage: uv run init <name> [--path ./environments] [-T/--add-tool] [-U/--add-user] "
    "[-H/--add-harness] [--v0]\n"
    "       scaffold a new v1 environment package (use --v0 for a legacy v0 environment)"
)


def _names(name: str) -> tuple[str, str, str, str]:
    """`(dash, pkg, stem, prefix)` derived from a raw name: the hyphenated id, the importable
    package (underscores), the `_v1`-less stem (for tool prefixes), and the CamelCase class
    prefix (e.g. `my-task-v1` -> `my-task-v1`, `my_task_v1`, `my_task`, `MyTask`)."""
    dash = name.strip().strip("/").replace("_", "-").lower()
    pkg = dash.replace("-", "_")
    stem = pkg[:-3] if pkg.endswith("_v1") else pkg
    prefix = "".join(part[:1].upper() + part[1:] for part in stem.split("_") if part)
    if not prefix or not prefix[0].isalpha():
        prefix = f"Env{prefix}"
    return dash, pkg, stem, prefix


def _write(path: Path, content: str, force: bool) -> bool:
    """Write `content` to `path` unless it exists (and `force` is off). Returns whether it wrote."""
    if path.exists() and not force:
        print(f"  skip   {path} (exists)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  write  {path}")
    return True


def _pyproject(dash: str, pkg: str) -> str:
    return f"""\
[project]
name = "{dash}"
version = "0.1.0"
description = "{dash} — <one-line description>."
requires-python = ">=3.11"
dependencies = ["verifiers"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["{pkg}"]
"""


def _init_py(pkg: str, prefix: str, add_harness: bool) -> str:
    lines = [f"from {pkg}.taskset import {prefix}Taskset"]
    exports = [f"{prefix}Taskset"]
    if add_harness:
        lines.append(f"from {pkg}.harness import {prefix}Harness")
        exports.append(f"{prefix}Harness")
    exports_repr = ", ".join(f'"{e}"' for e in exports)
    return "\n".join(lines) + f"\n\n__all__ = [{exports_repr}]\n"


def _taskset_py(pkg: str, prefix: str, *, add_tool: bool, add_user: bool) -> str:
    """The taskset module skeleton — a `Task`/`Config`/`Taskset` shell with `load_tasks` and a
    `@reward` to fill in. Each enabled piece (tool/user) adds its import, config field, and
    wiring method."""
    imports = "import verifiers.v1 as vf"
    local_imports: list[str] = []
    config_extra = ""
    methods: list[str] = []
    # a user simulator carries per-rollout state and a stop condition, so the taskset is typed with
    # its `State` subclass; without one it stays on the default `State`.
    state_param = ""
    if add_tool:
        local_imports.append(f"from {pkg}.servers.tool import {prefix}Toolset")
        config_extra += "\n    tools: vf.ToolsetConfig = vf.ToolsetConfig()"
        methods.append(
            f"    def tools(self, task: {prefix}Task) -> list[vf.Toolset]:\n"
            f"        return [{prefix}Toolset(self.config.tools)]"
        )
    if add_user:
        local_imports.append(
            f"from {pkg}.servers.user import {prefix}State, {prefix}User"
        )
        config_extra += "\n    user: vf.UserConfig = vf.UserConfig()"
        state_param = f", {prefix}State"
        methods.append(
            f"    def user(self, task: {prefix}Task) -> vf.User:\n"
            f"        return {prefix}User(self.config.user)"
        )
        methods.append(
            "    @vf.stop\n"
            "    async def user_done(self, trace: vf.Trace) -> bool:\n"
            "        return trace.state.done"
        )
    if local_imports:
        imports += "\n\n" + "\n".join(local_imports)
    methods_block = "".join(f"\n{m}\n" for m in methods)
    return f'''\
{imports}


class {prefix}Task(vf.Task):
    """A single task. Add task-specific fields here (e.g. a reference answer)."""


class {prefix}Config(vf.TasksetConfig):
    num_tasks: int = 5
    """How many tasks to build."""{config_extra}


class {prefix}Taskset(vf.Taskset[{prefix}Task, {prefix}Config{state_param}]):
    def load_tasks(self) -> list[{prefix}Task]:
        raise NotImplementedError(
            "Return this taskset's tasks, e.g. "
            "[{prefix}Task(idx=i, prompt=...) for i in range(self.config.num_tasks)]."
        )
{methods_block}
    @vf.reward(weight=1.0)
    async def reward(self, task: {prefix}Task, trace: vf.Trace) -> float:
        raise NotImplementedError("Score the rollout and return a float (e.g. in [0, 1]).")
'''


def _tool_py(stem: str, prefix: str) -> str:
    return f'''\
import verifiers.v1 as vf


class {prefix}Toolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "{stem}"

    @vf.tool
    def echo(self, text: str) -> str:
        """Return the given text unchanged (replace with a real tool)."""
        return text


if __name__ == "__main__":
    {prefix}Toolset.run()
'''


def _user_py(prefix: str) -> str:
    return f"""\
import verifiers.v1 as vf


class {prefix}State(vf.State):
    done: bool = False


class {prefix}User(vf.User[vf.UserConfig, {prefix}State]):
    async def respond(self, message: str) -> vf.Messages:
        if self.state.done:
            return []
        self.state.done = True
        return [vf.UserMessage(content="Thanks - anything else?")]


if __name__ == "__main__":
    {prefix}User.run()
"""


def _harness_py(prefix: str) -> str:
    return f'''\
import verifiers.v1 as vf


class {prefix}HarnessConfig(vf.HarnessConfig):
    """Run knobs for this harness. Add fields here (e.g. a CLI version to install)."""


class {prefix}Harness(vf.Harness[{prefix}HarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True

    async def launch(
        self,
        ctx: vf.RolloutContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        raise NotImplementedError(
            "Implement launch: run your agent in `runtime`, pointing its model calls at "
            "`endpoint` with bearer token `secret`."
        )
'''


def _readme(
    dash: str, pkg: str, *, add_tool: bool, add_user: bool, add_harness: bool
) -> str:
    layout = [
        f"- `{pkg}/taskset.py` — the taskset: `load_tasks` (data + prompts) and `@reward` (scoring)."
    ]
    if add_tool:
        layout.append(
            f"- `{pkg}/servers/tool.py` — a `vf.Toolset` tool server, wired in via `tools(task)`."
        )
    if add_user:
        layout.append(
            f"- `{pkg}/servers/user.py` — a `vf.User` simulator, wired in via `user(task)`."
        )
    if add_harness:
        layout.append(
            f"- `{pkg}/harness.py` — a custom harness, selectable with `--harness.id {dash}`."
        )
    layout_block = "\n".join(layout)
    return f"""\
# {dash}

A v1 verifiers environment, scaffolded with `init`.

## Develop

1. Implement `load_tasks` and the `@reward` in `{pkg}/taskset.py` (see `environments/*_v1`).
2. Install + run:

```bash
uv pip install -e .        # install this package (or register it in your project)
uv run eval {dash} -n 3    # evaluate a few tasks with the default harness
```

## Layout

{layout_block}

Tune knobs from the CLI: `--taskset.num-tasks 10`, `--model <id>`, `-n`/`-r`/`-t`/`-T`.
"""


def scaffold(config: InitConfig) -> Path:
    """Scaffold a v1 environment package from `config` and return its directory."""
    dash, pkg, stem, prefix = _names(config.name)
    env_dir = Path(config.path) / pkg
    pkg_dir = env_dir / pkg
    if env_dir.exists() and not config.force:
        raise SystemExit(
            f"error: {env_dir} already exists - refusing to overwrite (pass --force to overwrite)"
        )
    print(f"scaffolding v1 environment {dash!r} in {env_dir}")

    _write(env_dir / "pyproject.toml", _pyproject(dash, pkg), config.force)
    _write(
        env_dir / "README.md",
        _readme(
            dash,
            pkg,
            add_tool=config.add_tool,
            add_user=config.add_user,
            add_harness=config.add_harness,
        ),
        config.force,
    )
    _write(
        pkg_dir / "__init__.py", _init_py(pkg, prefix, config.add_harness), config.force
    )
    _write(
        pkg_dir / "taskset.py",
        _taskset_py(pkg, prefix, add_tool=config.add_tool, add_user=config.add_user),
        config.force,
    )
    if config.add_harness:
        _write(pkg_dir / "harness.py", _harness_py(prefix), config.force)
    if config.add_tool or config.add_user:
        _write(pkg_dir / "servers" / "__init__.py", "", config.force)
    if config.add_tool:
        _write(pkg_dir / "servers" / "tool.py", _tool_py(stem, prefix), config.force)
    if config.add_user:
        _write(pkg_dir / "servers" / "user.py", _user_py(prefix), config.force)

    print(f"\ndone. next:\n  uv pip install -e {env_dir}\n  uv run eval {dash} -n 3")
    return env_dir


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if argv and not argv[0].startswith(("-", "@")):  # leading bare token is the name
        argv = ["--name", argv[0], *argv[1:]]

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        cli(InitConfig)
        return

    sys.argv = [sys.argv[0], *argv]  # let prime-pydantic-config render help/errors
    config = cli(InitConfig)
    if not config.name:
        raise SystemExit(USAGE)
    if config.v0:
        if config.add_tool or config.add_user or config.add_harness:
            raise SystemExit(
                "--add-* flags are v1-only and can't be combined with --v0"
            )
        from verifiers.scripts.init import init_environment

        env_dir = init_environment(config.name, config.path, multi_file=True)
        print(f"scaffolded v0 environment in {env_dir}")
        return
    scaffold(config)


if __name__ == "__main__":
    main()
