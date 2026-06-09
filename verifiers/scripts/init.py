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
from .{env_id} import {imports}

__all__ = {exports}
"""

V0_ENVIRONMENT_TEMPLATE = """\
import verifiers as vf


def load_environment(**kwargs) -> vf.Environment:
    \"\"\"
    Load this environment.

    Environments typically return vf.SingleTurnEnv, vf.ToolEnv, etc.
    \"\"\"
    raise NotImplementedError("Implement load_environment here.")
"""














def _write_if_missing(path: Path, content: str) -> None:
    if path.exists():
        print(f"{path.name} already exists at {path}, skipping...")
        return
    path.write_text(content)




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
    taskset_name = _class_name(env_id_underscore, "Taskset")
    harness_config_name = _class_name(env_id_underscore, "HarnessConfig")
    harness_name = _class_name(env_id_underscore, "Harness")
    # make environment parent directory if it doesn't exist
    local_dir = Path(path) / env_id_underscore
    local_dir.mkdir(parents=True, exist_ok=True)

    # create README.md if it doesn't exist (or rewrite if flag is set)
    readme_file = local_dir / "README.md"
    if rewrite_readme or not readme_file.exists():
        readme_template = README_TEMPLATE
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
        pyproject_template = PYPROJECT_TEMPLATE
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
        template = V0_ENVIRONMENT_TEMPLATE
        environment_file.write_text(
            template.replace("{env_id_dash}", env_id_dash)
            .replace("{taskset_config_name}", taskset_config_name)
            .replace("{taskset_name}", taskset_name)
            .replace("{harness_config_name}", harness_config_name)
            .replace("{harness_name}", harness_name)
        )
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
