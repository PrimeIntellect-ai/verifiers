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

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig, cli

USAGE = (
    "usage: uv run init <name> [--path ./environments] [--add-tool] [--add-user] "
    "[--add-harness] [--v0]\n"
    "       scaffold a new v1 environment package (use --v0 for a legacy v0 environment)"
)


class InitConfig(BaseConfig):
    """What to scaffold. The `name` is the new environment id (a leading bare token, e.g.
    `init my-task-v1`); the package dir, ids, and class names are derived from it. The
    `--add-*` flags add optional pieces (tool server, user simulator, custom harness)."""

    name: str = ""
    """The new environment id, e.g. `my-task-v1` (positional: `init my-task-v1`)."""
    path: str = Field("./environments", validation_alias=AliasChoices("path", "p"))
    """Parent directory the package is created in (default `./environments`)."""
    add_tool: bool = False
    """Also scaffold a `vf.Toolset` tool server (`servers/tool.py`), wired into the taskset."""
    add_user: bool = False
    """Also scaffold a `vf.User` simulator (`servers/user.py`), wired into the taskset."""
    add_harness: bool = False
    """Also scaffold a custom `vf.Harness` (`harness.py`), selectable via `--harness.id <name>`."""
    v0: bool = False
    """Scaffold a legacy v0 environment (a `load_environment` package) instead of a v1 taskset."""
    force: bool = False
    """Overwrite files that already exist (default: keep them, scaffold only what's missing)."""


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
    """The taskset module, assembled so each enabled piece (tool/user) adds its import, config
    field, and wiring method. Runs out of the box with no flags."""
    imports = "import re\n\nimport verifiers.v1 as vf"
    local_imports: list[str] = []
    config_extra = ""
    methods: list[str] = []
    if add_tool:
        local_imports.append(f"from {pkg}.servers.tool import {prefix}Toolset")
        config_extra += "\n    tools: vf.ToolsetConfig = vf.ToolsetConfig()"
        methods.append(
            f"    def tools(self, task: {prefix}Task) -> list[vf.Toolset]:\n"
            f"        return [{prefix}Toolset(self.config.tools)]"
        )
    if add_user:
        local_imports.append(f"from {pkg}.servers.user import {prefix}User")
        config_extra += "\n    user: vf.UserConfig = vf.UserConfig()"
        methods.append(
            f"    def user(self, task: {prefix}Task) -> vf.User:\n"
            f"        return {prefix}User(self.config.user)"
        )
    if local_imports:
        imports += "\n\n" + "\n".join(local_imports)
    methods_block = "".join(f"\n{m}\n" for m in methods)
    return f'''\
"""{pkg.replace("_", "-")} — <one-line description of the task>.

A starter v1 taskset: replace `load_tasks` (your data + prompts) and the `@reward` (how a
rollout is scored). Runs out of the box: `uv run eval {pkg.replace("_", "-")} -n 3`.
"""

{imports}

WORDS = ["alpha", "bravo", "charlie", "delta", "echo"]
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class {prefix}Task(vf.Task):
    answer: str
    """The expected answer for this task (read by the reward)."""


class {prefix}Config(vf.TasksetConfig):
    num_tasks: int = 5
    """How many tasks to build."""{config_extra}


class {prefix}Taskset(vf.Taskset[{prefix}Task, {prefix}Config]):
    def load_tasks(self) -> list[{prefix}Task]:
        return [
            {prefix}Task(
                idx=i,
                instruction=f"Repeat the word {{word!r}} inside <answer></answer> tags.",
                answer=word,
            )
            for i, word in enumerate(WORDS[: self.config.num_tasks])
        ]
{methods_block}
    @vf.reward(weight=1.0)
    async def exact_match(self, task: {prefix}Task, trace: vf.Trace) -> float:
        # Reward 1.0 if any assistant turn answered with the expected word in <answer> tags.
        for message in trace.assistant_messages:
            match = _ANSWER.search(message.content or "")
            if match and match.group(1).strip() == task.answer:
                return 1.0
        return 0.0
'''


def _tool_py(stem: str, prefix: str) -> str:
    return f'''\
"""A tool server for {stem.replace("_", "-")} — a vf-native `Toolset` (no FastMCP boilerplate).

The framework launches it and surfaces each `@vf.tool` method to the model as `{stem}_<method>`.
Replace `echo` with your task's tools.
"""

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


def _user_py(stem: str, prefix: str) -> str:
    return f'''\
"""A user simulator for {stem.replace("_", "-")} — a vf-native `User` driving the conversation.

The framework calls `respond` after each model turn for the next user message(s) + a done flag.
If a task carries no prompt (`instruction=None`), `respond("")` is called first to open the
conversation. Replace the logic with your simulated user.
"""

import verifiers.v1 as vf


class {prefix}User(vf.User[vf.UserConfig]):
    async def setup_task(self, task) -> None:
        self.replied = False

    async def respond(self, message: str) -> tuple[vf.Messages, bool]:
        if self.replied:
            return [], True
        self.replied = True
        return [{{"role": "user", "content": "Thanks - anything else?"}}], False


if __name__ == "__main__":
    {prefix}User.run()
'''


def _harness_py(dash: str, prefix: str) -> str:
    return f'''\
"""A custom harness for {dash} — replace `launch` with how your agent runs.

A harness drives the rollout: it runs a program in `runtime` whose model calls hit the
interception server at `endpoint` (bearer token `secret`); `mcp_urls` are the task's tool
servers to wire in. See verifiers' built-in `default` harness for a chat-loop reference. This
package exports it alongside the taskset, so select it with `--harness.id {dash}`.
"""

import verifiers.v1 as vf


class {prefix}HarnessConfig(vf.HarnessConfig):
    id: str = "{dash}"


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

## Run

```bash
uv pip install -e .        # install this package (or register it in your project)
uv run eval {dash} -n 3    # evaluate a few tasks with the default harness
```

## Layout

{layout_block}

## Next steps

- Replace `load_tasks` and `exact_match` in `{pkg}/taskset.py` with your task and scoring.
- Tune knobs from the CLI: `--taskset.num-tasks 10`, `--model <id>`, `-n`/`-r`/`-t`/`-T`.
"""


def scaffold(config: InitConfig) -> Path:
    """Scaffold a v1 environment package from `config` and return its directory."""
    dash, pkg, stem, prefix = _names(config.name)
    env_dir = Path(config.path) / pkg
    pkg_dir = env_dir / pkg
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
        _write(pkg_dir / "harness.py", _harness_py(dash, prefix), config.force)
    if config.add_tool or config.add_user:
        _write(pkg_dir / "servers" / "__init__.py", "", config.force)
    if config.add_tool:
        _write(pkg_dir / "servers" / "tool.py", _tool_py(stem, prefix), config.force)
    if config.add_user:
        _write(pkg_dir / "servers" / "user.py", _user_py(stem, prefix), config.force)

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
