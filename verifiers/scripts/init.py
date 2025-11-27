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
- **Parser**: <e.g., ThinkParser, XMLParser, custom>
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval {env_id_dash}
```

Configure model and sampling:

```bash
uv run vf-eval {env_id_dash} \
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

INIT_TEMPLATE = """\
from .{env_id} import load_environment

__all__ = ["load_environment"]
"""

ENVIRONMENT_TEMPLATE = """\
import verifiers as vf


def load_environment(**kwargs) -> vf.Environment:
    '''
    Loads a custom environment.
    '''
    raise NotImplementedError("Implement your custom environment here.")
"""


def init_environment(
    env: str,
    path: str = "./environments",
    rewrite_readme: bool = False,
    multi_file: bool = False,
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
        readme_file.write_text(
            README_TEMPLATE.format(
                env_id_dash=env_id_dash, env_id_underscore=env_id_underscore
            )
        )
    else:
        print(f"README.md already exists at {readme_file}, skipping...")

    # create pyproject.toml if it doesn't exist
    pyproject_file = local_dir / "pyproject.toml"
    if not pyproject_file.exists():
        pyproject_file.write_text(
            PYPROJECT_TEMPLATE.format(env_id=env_id_dash, env_file=env_id_underscore)
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
        environment_file.write_text(ENVIRONMENT_TEMPLATE)
    else:
        print(
            f"{env_id_underscore}.py already exists at {environment_file}, skipping..."
        )

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
    args = parser.parse_args()

    init_environment(
        args.env,
        args.path,
        rewrite_readme=args.rewrite_readme,
        multi_file=args.multi_file,
    )


if __name__ == "__main__":
    main()
