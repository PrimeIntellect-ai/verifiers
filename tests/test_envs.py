import os
import subprocess
from pathlib import Path

import pytest
import tomllib

SKIPPED_ENVS = [
    # Requires fix for resource setup
    # uv run pytest tests/test_envs.py -vv -k mcp_env
    #
    # npm ERR! Cannot read property 'split' of undefined
    # npm ERR! A complete log of this run can be found in:
    # npm ERR!     /home/ubuntu/.npm/_logs/2025-12-12T11_25_18_731Z-debug-0.log
    "mcp_env",
    # Requires fix for completion dataset setup
    # uv run pytest tests/test_envs.py -vv -k continuation_quality
    #
    #     example_id = input_item["example_id"]
    #                 ~~~~~~~~~~^^^^^^^^^^^^^^
    # KeyError: 'example_id'
    "continuation_quality",
]


def get_environments() -> list[Path]:
    """Get all subdirectories of `environments/`, or only changed environments if CHANGED_ENVS is set."""
    all_envs = list(x for x in Path("environments").iterdir() if x.is_dir())

    # Filter environments if CHANGED_ENVS is set (for PRs)
    changed_envs = os.getenv("CHANGED_ENVS")
    if changed_envs == "none":
        return []
    if changed_envs:
        changed_list = [e.strip() for e in changed_envs.split(",") if e.strip()]
        if changed_list:
            all_envs = [env for env in all_envs if env.name in changed_list]

    return all_envs


@pytest.mark.parametrize("env_dir", get_environments(), ids=lambda x: x.name)
def test_pyproject_exists(env_dir: Path):
    """Test that the pyproject.toml file exists for the given environment directory."""
    assert (env_dir / "pyproject.toml").exists(), "pyproject.toml does not exist"


@pytest.mark.parametrize("env_dir", get_environments(), ids=lambda x: x.name)
def test_pyproject_has_metadata(env_dir: Path):
    """Test that the pyproject.toml file has the required metadata."""
    with open(env_dir / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    assert "name" in pyproject["project"], "pyproject.toml does not have a name"
    assert "version" in pyproject["project"], "pyproject.toml does not have a version"
    assert "description" in pyproject["project"], (
        "pyproject.toml does not have a description"
    )
    assert pyproject["project"]["description"] != "Your environment description here", (
        "Still uses placeholder description"
    )
    assert "tags" in pyproject["project"], "pyproject.toml does not have tags"
    assert pyproject["project"]["tags"] != ["placeholder-tag", "train", "eval"], (
        "Still uses placeholder tags"
    )


@pytest.mark.parametrize("env_dir", get_environments(), ids=lambda x: x.name)
def test_readme_exists(env_dir: Path):
    """Test that the README.md file exists for the given environment directory."""
    assert (env_dir / "README.md").exists(), "README.md does not exist"


@pytest.mark.parametrize("env_dir", get_environments(), ids=lambda x: x.name)
def test_env(env_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    """Test environment in a fresh venv with local verifiers installed first."""
    if env_dir.name in SKIPPED_ENVS:
        pytest.skip(f"Skipping {env_dir.name}")
    tmp_venv_dir = tmp_path_factory.mktemp(f"venv_{env_dir.name}")
    repo_root = Path(__file__).parent.parent
    cmd = (
        f"cd {tmp_venv_dir} && uv venv --clear && source .venv/bin/activate && "
        f"uv pip install {repo_root.as_posix()} && "
        f"uv pip install {env_dir.absolute().as_posix()}"
    )
    process = subprocess.run(
        cmd, shell=True, executable="/bin/bash", capture_output=True, text=True
    )
    assert process.returncode == 0, (
        f"Failed to create virtual environment: {process.stderr}"
    )

    help_test_can_import_env(tmp_venv_dir, env_dir)
    help_test_can_load_env(tmp_venv_dir, env_dir)
    help_test_can_eval_env(tmp_venv_dir, env_dir)


def help_test_can_import_env(tmp_venv_dir: Path, env_dir: Path):
    """Test that the environment can be imported as a package."""
    import_cmd = f"cd {tmp_venv_dir} && source .venv/bin/activate && uv run python -c 'import {env_dir.name}'"
    process = subprocess.run(
        import_cmd, shell=True, executable="/bin/bash", capture_output=True, text=True
    )
    assert process.returncode == 0, "Failed to import environment"


def help_test_can_load_env(tmp_venv_dir: Path, env_dir: Path):
    """Test that the environment can be loaded."""
    load_cmd = f"""cd {tmp_venv_dir} && source .venv/bin/activate && uv run python -c 'import verifiers as vf; vf.load_environment("{env_dir.name}")'"""
    process = subprocess.run(
        load_cmd, shell=True, executable="/bin/bash", capture_output=True, text=True
    )
    assert process.returncode == 0, "Failed to load environment"


def help_test_can_eval_env(tmp_venv_dir: Path, env_dir: Path):
    """Test that the environment can be run via vf-eval."""
    eval_cmd = f"cd {tmp_venv_dir} && source .venv/bin/activate && uv run vf-eval {env_dir.name} -n 1 -r 1 -t 512"
    process = subprocess.run(
        eval_cmd, shell=True, executable="/bin/bash", capture_output=True, text=True
    )
    assert process.returncode == 0, "Failed to evaluate environment"
