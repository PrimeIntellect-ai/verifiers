"""Scaffold v1 environment packages."""

import sys
from pathlib import Path

from pydantic_config import cli

from verifiers.v1.configs.init import InitConfig

USAGE = (
    "usage: uv run init <name> [--path ./environments] [-T/--add-tool] "
    "[-H/--add-harness] [--v0]\n"
    "       scaffold a new v1 environment package (use --v0 for a legacy v0 environment)"
)


def _names(name: str) -> tuple[str, str, str, str]:
    dash = name.strip().strip("/").replace("_", "-").lower()
    pkg = dash.replace("-", "_")
    stem = pkg[:-3] if pkg.endswith("_v1") else pkg
    prefix = "".join(part[:1].upper() + part[1:] for part in stem.split("_") if part)
    if not prefix or not prefix[0].isalpha():
        prefix = f"Env{prefix}"
    return dash, pkg, stem, prefix


def _write(path: Path, content: str, force: bool) -> bool:
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


def _taskset_py(pkg: str, prefix: str, *, add_tool: bool) -> str:
    imports = "import verifiers.v1 as vf"
    local_imports: list[str] = []
    task_config_fields = ""
    task_decls = ""
    state = "vf.State"
    if add_tool:
        local_imports.append(f"from {pkg}.servers.tool import {prefix}Toolset")
        task_config_fields += "\n    tools: vf.ToolsetConfig = vf.ToolsetConfig()"
        task_decls += f"\n    tools = ({prefix}Toolset,)"
    if local_imports:
        imports += "\n\n" + "\n".join(local_imports)
    methods_block = ""
    has_task_config = add_tool
    task_config = (
        f"\n\nclass {prefix}TaskConfig(vf.TaskConfig):\n"
        '    """Knobs the task reads from ``self.config``; configure them under '
        f'``--taskset.task.*``."""{task_config_fields}\n'
        if has_task_config
        else ""
    )
    task_generic = (
        f"{prefix}Data, {state}, {prefix}TaskConfig"
        if has_task_config
        else f"{prefix}Data"
    )
    config_body = '    num_tasks: int = 5\n    """How many tasks to build."""\n' + (
        f"    task: {prefix}TaskConfig = {prefix}TaskConfig()\n"
        if has_task_config
        else ""
    )
    return f'''\
{imports}


class {prefix}Data(vf.TaskData):
    """One row's data. Add task-specific fields here, such as a reference answer."""
{task_config}

class {prefix}Task(vf.Task[{task_generic}]):
    """Rewards, hooks, and servers, with row data available on ``self.data``."""{task_decls}{methods_block}
    @vf.reward(weight=1.0)
    async def reward(self, trace: vf.Trace) -> float:
        raise NotImplementedError("Score the rollout and return a float (e.g. in [0, 1]).")


class {prefix}Config(vf.TasksetConfig):
{config_body}

class {prefix}Taskset(vf.Taskset[{prefix}Task, {prefix}Config]):
    def load(self) -> list[{prefix}Task]:
        raise NotImplementedError(
            "Return this taskset's tasks, e.g. "
            "[{prefix}Task({prefix}Data(idx=i, prompt=...), self.config.task) "
            "for i in range(self.config.num_tasks)]."
        )
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


def _harness_py(prefix: str) -> str:
    return f'''\
import verifiers.v1 as vf


class {prefix}HarnessConfig(vf.HarnessConfig):
    """Run knobs for this harness. Add fields here (e.g. a CLI version to install)."""


class {prefix}Harness(vf.Harness[{prefix}HarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True

    async def launch(
        self,
        ctx: vf.ModelContext,
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


def _readme(dash: str, pkg: str, *, add_tool: bool, add_harness: bool) -> str:
    layout = [
        f"- `{pkg}/taskset.py` — the task (`@reward` scoring + behavior) and the taskset: "
        "`load` (data + prompts)."
    ]
    if add_tool:
        layout.append(
            f"- `{pkg}/servers/tool.py` — a `vf.Toolset` tool server, declared on `Task.tools`."
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

1. Implement `load` and the `@reward` in `{pkg}/taskset.py` (see `environments/*_v1`).
2. Install + run:

```bash
uv pip install -e .        # install this package (or register it in your project)
uv run eval {dash} -n 3    # evaluate a few tasks with the default harness
```

## Layout

{layout_block}

Tune knobs from the CLI: `--taskset.num-tasks 10`, `--model <id>`, `-n`, and `-r`.
"""


def scaffold(config: InitConfig) -> Path:
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
        _readme(dash, pkg, add_tool=config.add_tool, add_harness=config.add_harness),
        config.force,
    )
    _write(
        pkg_dir / "__init__.py", _init_py(pkg, prefix, config.add_harness), config.force
    )
    _write(
        pkg_dir / "taskset.py",
        _taskset_py(pkg, prefix, add_tool=config.add_tool),
        config.force,
    )
    if config.add_harness:
        _write(pkg_dir / "harness.py", _harness_py(prefix), config.force)
    if config.add_tool:
        _write(pkg_dir / "servers" / "__init__.py", "", config.force)
        _write(pkg_dir / "servers" / "tool.py", _tool_py(stem, prefix), config.force)

    print(f"\ndone. next:\n  uv pip install -e {env_dir}\n  uv run eval {dash} -n 3")
    return env_dir


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if argv and not argv[0].startswith(("-", "@")):
        argv = ["--name", argv[0], *argv[1:]]

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        cli(InitConfig)
        return

    sys.argv = [sys.argv[0], *argv]
    config = cli(InitConfig)
    if not config.name:
        raise SystemExit(USAGE)
    if config.v0:
        if config.add_tool or config.add_harness:
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
