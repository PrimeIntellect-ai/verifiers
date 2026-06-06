import argparse
from pathlib import Path

import verifiers as vf

README_TEMPLATE = """\
# {env_id_dash}

> Replace the placeholders below, then remove this callout.

### Overview
- **Environment ID**: `{env_id_dash}`
- **Short description**: <one-sentence description>
- **Tags**: <comma-separated tags>

### Datasets
- **Primary dataset(s)**: <name(s) and brief description>
- **Source links**: <links>
- **Split sizes**: <train/eval counts>

### Task
- **Type**: <single-turn | multi-turn | tool use>
- **Output format expectations (optional)**: <e.g., plain text, XML tags, JSON schema>
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run {env_id_dash}
```

Configure model and sampling:

```bash
prime eval run {env_id_dash} \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- Put task-owned settings under `[env.taskset]` and harness-owned settings under `[env.harness]` in TOML configs.

### Taskset Config
Document any taskset config fields and their meaning. Example:

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Harness Config
Document any harness config fields and their meaning.

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |

"""

OPENENV_README_TEMPLATE = """\
# {env_id_dash}

### Overview
- **Environment ID**: `{env_id_dash}`
- **Short description**: OpenEnv-backed environment using `tasksets.OpenEnvTaskset`.
- **Tags**: openenv, tools, multi-turn

### Structure
This template enforces:
- `proj/` contains the OpenEnv project
- `proj/.build.json` contains the built image/runtime metadata

### Quickstart
Build and register the image:

```bash
uv run vf-build {env_id_dash}
```

Run eval:

```bash
prime eval run {env_id_dash}
```
"""

PYPROJECT_TEMPLATE = f"""\
[project]
name = "{{env_id}}"
description = "Your environment description here"
tags = ["placeholder-tag", "train", "eval"]
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "verifiers>={vf.__version__}",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["{{env_file}}.py", "pyproject.toml"] 

[tool.verifiers.eval]
num_examples = 5
rollouts_per_example = 3
"""

OPENENV_PYPROJECT_TEMPLATE = f"""\
[project]
name = "{{env_id}}"
description = "OpenEnv-backed environment"
tags = ["openenv", "tools", "multi-turn"]
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "verifiers>={vf.__version__}",
    "tasksets[openenv]>=0.1.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["{{env_file}}.py", "pyproject.toml", "README.md", "proj/**/*", "proj/.build.json"]

[tool.verifiers.eval]
num_examples = 5
rollouts_per_example = 3
"""

INIT_TEMPLATE = """\
from .{env_id} import {imports}

__all__ = {exports}
"""

V0_ENVIRONMENT_TEMPLATE = """\
import verifiers as vf


def load_environment(**kwargs) -> vf.Environment:
    \"\"\"
    Load this environment.

    v0 environments typically return vf.SingleTurnEnv, vf.ToolEnv, etc.
    For the v1 Taskset/Harness pattern: prime env init <name> --v1
    \"\"\"
    raise NotImplementedError("Implement load_environment here.")
"""

V1_TASKSET_TEMPLATE = """\
import verifiers.v1 as vf


class {taskset_config_name}(vf.TasksetConfig):
    \"\"\"User-facing task settings for {env_id_dash}.\"\"\"

    system_prompt: vf.SystemPrompt = "Answer exactly."


class {task_name}(vf.Task):
    answer: str


class {taskset_name}(vf.Taskset[{taskset_config_name}]):
    \"\"\"Taskset implementation for {env_id_dash}.

    Add task loading, task-owned toolsets, user behavior, lifecycle hooks,
    metrics, rewards, and advantages on this class.
    \"\"\"

    task_type = {task_name}

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        \"\"\"Return serializable task records as a list, generator, or Dataset.\"\"\"
        if split == "eval":
            return []
        return [
            {
                "prompt": [{"role": "user", "content": "Reverse abc."}],
                "answer": "cba",
                "max_turns": 1,
            }
        ]

    @vf.reward(weight=1.0)
    async def correct_answer(self, task: {task_name}, state: vf.State) -> float:
        \"\"\"Score the final assistant response for one rollout.\"\"\"
        messages = vf.get_messages(state.completion or [], role="assistant")
        if not messages:
            return 0.0
        response = str(messages[-1].content or "").strip()
        return float(response == task.answer)


def load_taskset(config: {taskset_config_name}) -> {taskset_name}:
    \"\"\"Typed taskset loader used by vf.load_taskset.\"\"\"
    return {taskset_name}(config=config)
"""


V1_HARNESS_TEMPLATE = """\

class {harness_config_name}(vf.HarnessConfig):
    \"\"\"Execution settings for {env_id_dash}.\"\"\"


class {harness_name}(vf.Harness[{harness_config_name}]):
    \"\"\"Reusable execution behavior for {env_id_dash}.

    Add harness-owned program, sandbox, endpoint, model, toolset, or lifecycle
    behavior here when this environment owns a custom execution mechanism.
    \"\"\"


def load_harness(config: {harness_config_name}) -> {harness_name}:
    \"\"\"Typed harness loader used by vf.load_harness.\"\"\"
    return {harness_name}(config=config)
"""


V1_ENV_LOADER_TEMPLATE = """\

def load_environment(config: vf.EnvConfig) -> vf.Env:
    \"\"\"Loader pattern for all Taskset/Harness environments.\"\"\"
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
        runtime=config.runtime,
    )
"""

V1_ENVIRONMENT_TEMPLATE = V1_TASKSET_TEMPLATE + V1_ENV_LOADER_TEMPLATE
V1_HARNESS_ENVIRONMENT_TEMPLATE = (
    V1_TASKSET_TEMPLATE + V1_HARNESS_TEMPLATE + V1_ENV_LOADER_TEMPLATE
)

OPENENV_ENVIRONMENT_TEMPLATE = """\
import verifiers.v1 as vf
from tasksets import OpenEnvTaskset, OpenEnvTasksetConfig


def load_taskset(config: OpenEnvTasksetConfig) -> OpenEnvTaskset:
    return OpenEnvTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    \"\"\"Loader pattern for all Taskset/Harness environments.\"\"\"
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
        runtime=config.runtime,
    )
"""

OPENENV_PROJ_README_TEMPLATE = """\
# OpenEnv Source

Place your full OpenEnv project in this folder.

Required files:
- `openenv.yaml`
- `pyproject.toml`
- `server/Dockerfile`
- `server/app.py`

Then build the sandbox image:

```bash
uv run vf-build {env_id_dash}
```
"""

OPENENV_PROJ_MANIFEST_TEMPLATE = """\
spec_version: 1
name: {env_id_underscore}
type: space
runtime: fastapi
app: server.app:app
port: 8000
"""

OPENENV_PROJ_PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{env_id_dash}-openenv"
version = "0.1.0"
description = "OpenEnv project bundled with a verifiers environment"
requires-python = ">=3.10"
dependencies = [
    "openenv-core>=0.3.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.24.0",
]
"""

OPENENV_PROJ_DOCKERFILE_TEMPLATE = """\
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app
COPY . /app/env
WORKDIR /app/env

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

FROM ${BASE_IMAGE}
WORKDIR /app
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
"""

OPENENV_PROJ_APP_TEMPLATE = """\
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Replace this stub with a real OpenEnv server implementation.
"""


def _write_if_missing(path: Path, content: str) -> None:
    if path.exists():
        print(f"{path.name} already exists at {path}, skipping...")
        return
    path.write_text(content)


def _init_openenv_proj(
    local_dir: Path, env_id_dash: str, env_id_underscore: str
) -> None:
    proj_dir = local_dir / "proj"
    server_dir = proj_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)

    _write_if_missing(
        proj_dir / "README.md",
        OPENENV_PROJ_README_TEMPLATE.format(env_id_dash=env_id_dash),
    )
    _write_if_missing(
        proj_dir / "openenv.yaml",
        OPENENV_PROJ_MANIFEST_TEMPLATE.format(env_id_underscore=env_id_underscore),
    )
    _write_if_missing(
        proj_dir / "pyproject.toml",
        OPENENV_PROJ_PYPROJECT_TEMPLATE.format(env_id_dash=env_id_dash),
    )
    _write_if_missing(
        server_dir / "Dockerfile",
        OPENENV_PROJ_DOCKERFILE_TEMPLATE,
    )
    _write_if_missing(
        server_dir / "app.py",
        OPENENV_PROJ_APP_TEMPLATE,
    )
    _write_if_missing(server_dir / "__init__.py", "")


def _class_name(env_id_underscore: str, suffix: str) -> str:
    prefix = "".join(
        part[:1].upper() + part[1:] for part in env_id_underscore.split("_") if part
    )
    if not prefix or not prefix[0].isalpha():
        prefix = f"Env{prefix}"
    return f"{prefix}{suffix}"


def init_environment(
    env: str,
    path: str = "./environments",
    rewrite_readme: bool = False,
    multi_file: bool = False,
    openenv: bool = False,
    v1: bool = False,
    with_harness: bool = False,
) -> Path:
    """
    Initialize a new verifiers environment.

    Args:
        env: The environment id to init
        path: Path to environments directory (default: ./environments)

    Returns:
        Path to the created environment directory
    """

    env_id_dash = env.replace("_", "-")
    env_id_underscore = env_id_dash.replace("-", "_")
    taskset_config_name = _class_name(env_id_underscore, "TasksetConfig")
    task_name = _class_name(env_id_underscore, "Task")
    taskset_name = _class_name(env_id_underscore, "Taskset")
    harness_config_name = _class_name(env_id_underscore, "HarnessConfig")
    harness_name = _class_name(env_id_underscore, "Harness")
    if openenv:
        v1 = True
    if with_harness and not v1:
        print("--with-harness only applies with --v1; ignoring.")
        with_harness = False

    # make environment parent directory if it doesn't exist
    local_dir = Path(path) / env_id_underscore
    local_dir.mkdir(parents=True, exist_ok=True)

    # create README.md if it doesn't exist (or rewrite if flag is set)
    readme_file = local_dir / "README.md"
    if rewrite_readme or not readme_file.exists():
        readme_template = OPENENV_README_TEMPLATE if openenv else README_TEMPLATE
        readme_file.write_text(
            readme_template.format(
                env_id_dash=env_id_dash, env_id_underscore=env_id_underscore
            )
        )
    else:
        print(f"README.md already exists at {readme_file}, skipping...")

    # create pyproject.toml if it doesn't exist
    pyproject_file = local_dir / "pyproject.toml"
    if not pyproject_file.exists():
        pyproject_template = (
            OPENENV_PYPROJECT_TEMPLATE if openenv else PYPROJECT_TEMPLATE
        )
        pyproject_file.write_text(
            pyproject_template.format(env_id=env_id_dash, env_file=env_id_underscore)
        )
    else:
        print(f"pyproject.toml already exists at {pyproject_file}, skipping...")

    # create environment directory if it doesn't exist
    environment_dir = local_dir / env_id_underscore if multi_file else local_dir
    environment_dir.mkdir(parents=True, exist_ok=True)

    # create init file if it doesn't exist
    if multi_file:
        init_file = environment_dir / "__init__.py"
        if not init_file.exists():
            exports = ["load_environment"]
            if v1:
                exports.append("load_taskset")
            if v1 and with_harness and not openenv:
                exports.append("load_harness")
            init_file.write_text(
                INIT_TEMPLATE.format(
                    env_id=env_id_underscore,
                    imports=", ".join(exports),
                    exports=repr(exports),
                )
            )
        else:
            print(f"__init__.py already exists at {init_file}, skipping...")

    # create environment file if it doesn't exist
    environment_file = environment_dir / f"{env_id_underscore}.py"
    if not environment_file.exists():
        if openenv:
            template = OPENENV_ENVIRONMENT_TEMPLATE
        elif v1 and with_harness:
            template = V1_HARNESS_ENVIRONMENT_TEMPLATE
        elif v1:
            template = V1_ENVIRONMENT_TEMPLATE
        else:
            template = V0_ENVIRONMENT_TEMPLATE
        environment_file.write_text(
            template.replace("{env_id_dash}", env_id_dash)
            .replace("{taskset_config_name}", taskset_config_name)
            .replace("{task_name}", task_name)
            .replace("{taskset_name}", taskset_name)
            .replace("{harness_config_name}", harness_config_name)
            .replace("{harness_name}", harness_name)
        )
    else:
        print(
            f"{env_id_underscore}.py already exists at {environment_file}, skipping..."
        )

    if openenv:
        _init_openenv_proj(local_dir, env_id_dash, env_id_underscore)

    return local_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env",
        type=str,
        help="The environment id to init",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="./environments",
        help="Path to environments directory (default: ./environments)",
    )
    parser.add_argument(
        "--rewrite-readme",
        action="store_true",
        default=False,
        help="Rewrite README.md even if it already exists",
    )
    parser.add_argument(
        "--multi-file",
        action="store_true",
        default=False,
        help="Create multi-file package structure instead of single file",
    )
    parser.add_argument(
        "--openenv",
        action="store_true",
        default=False,
        help="Initialize with the enforced OpenEnv layout (proj/ + vf-build workflow).",
    )
    parser.add_argument(
        "--v1",
        action="store_true",
        default=False,
        help="Initialize a v1 Taskset/Harness environment template.",
    )
    parser.add_argument(
        "--with-harness",
        action="store_true",
        default=False,
        help="Include an explicit v1 load_harness stub. Requires --v1.",
    )
    args = parser.parse_args()

    init_environment(
        args.env,
        args.path,
        rewrite_readme=args.rewrite_readme,
        multi_file=args.multi_file,
        openenv=args.openenv,
        v1=args.v1,
        with_harness=args.with_harness,
    )


if __name__ == "__main__":
    main()
