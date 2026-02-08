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
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{{"key": "value"}}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `foo` | str | `"bar"` | What this controls |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

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
- **Short description**: OpenEnv-backed environment using `vf.OpenEnvEnv`.
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
    "verifiers[openenv]>={vf.__version__}",
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
from .{env_id} import load_environment

__all__ = ["load_environment"]
"""

ENVIRONMENT_TEMPLATE = '''\
import verifiers as vf


def load_environment(**kwargs) -> vf.Environment:
    """
    Loads a custom environment.
    """
    raise NotImplementedError("Implement your custom environment here.")
'''

OPENENV_ENVIRONMENT_TEMPLATE = """\
from typing import Any

import verifiers as vf


def render_prompt(observation: Any) -> list[dict[str, Any]]:
    if isinstance(observation, dict):
        messages = observation.get("messages")
        if isinstance(messages, list) and messages:
            return messages
        prompt = observation.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return [{"role": "user", "content": prompt}]
    raise RuntimeError(
        "OpenEnv observation did not include a renderable prompt. "
        "Update render_prompt() for your project's observation schema."
    )


def load_environment(
    num_train_examples: int = 100,
    num_eval_examples: int = 50,
    seed: int = 0,
):
    return vf.OpenEnvEnv(
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        prompt_renderer=render_prompt,
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
    "openenv-core[core]==0.2.1",
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


def init_environment(
    env: str,
    path: str = "./environments",
    rewrite_readme: bool = False,
    multi_file: bool = False,
    openenv: bool = False,
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
            init_file.write_text(INIT_TEMPLATE.format(env_id=env_id_underscore))
        else:
            print(f"__init__.py already exists at {init_file}, skipping...")

    # create environment file if it doesn't exist
    environment_file = environment_dir / f"{env_id_underscore}.py"
    if not environment_file.exists():
        template = OPENENV_ENVIRONMENT_TEMPLATE if openenv else ENVIRONMENT_TEMPLATE
        environment_file.write_text(template)
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
    args = parser.parse_args()

    init_environment(
        args.env,
        args.path,
        rewrite_readme=args.rewrite_readme,
        multi_file=args.multi_file,
        openenv=args.openenv,
    )


if __name__ == "__main__":
    main()
