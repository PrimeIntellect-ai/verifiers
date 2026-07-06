"""V0 environment scaffolding used by ``init --v0``."""

from pathlib import Path

import verifiers as vf

README_TEMPLATE = """\
# {env_id_dash}

Legacy V0 Verifiers environment.

## Develop

Implement `load_environment()` in `{env_id_underscore}.py`, install the package, and run it
through the V0 bridge:

```console
uv pip install -e .
prime eval {env_id_dash} --model openai/gpt-4.1-mini --num-tasks 20
```

V1 tasksets are the default for new work; run `prime env init <name>` without `--v0` to
scaffold one.
"""

PYPROJECT_TEMPLATE = f"""\
[project]
name = "{{env_id}}"
description = "Your environment description here"
tags = ["placeholder-tag", "train", "eval"]
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "verifiers>={vf.__version__}",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["{{env_file}}.py", "proj/**", "pyproject.toml"]
"""

V0_ENVIRONMENT_TEMPLATE = """\
import verifiers as vf


def load_environment(**kwargs) -> vf.Environment:
    \"\"\"Build and return this V0 environment.\"\"\"
    raise NotImplementedError("Implement load_environment here.")
"""


def init_environment(
    env: str,
    path: str = "./environments",
    rewrite_readme: bool = False,
) -> Path:
    """Create a single-module V0 package and return its directory."""
    env_id_dash = env.replace("_", "-")
    env_id_underscore = env_id_dash.replace("-", "_")
    local_dir = Path(path) / env_id_underscore
    local_dir.mkdir(parents=True, exist_ok=True)

    readme = local_dir / "README.md"
    if rewrite_readme or not readme.exists():
        readme.write_text(
            README_TEMPLATE.format(
                env_id_dash=env_id_dash,
                env_id_underscore=env_id_underscore,
            )
        )
    else:
        print(f"README.md already exists at {readme}, skipping...")

    pyproject = local_dir / "pyproject.toml"
    if not pyproject.exists():
        pyproject.write_text(
            PYPROJECT_TEMPLATE.format(
                env_id=env_id_dash,
                env_file=env_id_underscore,
            )
        )
    else:
        print(f"pyproject.toml already exists at {pyproject}, skipping...")

    module = local_dir / f"{env_id_underscore}.py"
    if not module.exists():
        module.write_text(V0_ENVIRONMENT_TEMPLATE)
    else:
        print(f"{module.name} already exists at {module}, skipping...")
    return local_dir
