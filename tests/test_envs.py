import os
import subprocess
from pathlib import Path

import pytest
import tomllib

# Timeout in seconds for each subprocess step
INSTALL_TIMEOUT = 600  # 10 minutes for venv creation + package install
IMPORT_TIMEOUT = 120  # 2 minutes for importing a package
LOAD_TIMEOUT = 300  # 5 minutes for loading an environment (may download datasets)
EVAL_TIMEOUT = 600  # 10 minutes for running a capped eval (-n 1 -r 1)

SKIPPED_ENVS = [
    # Requires fix for completion dataset setup
    # uv run pytest tests/test_envs.py -vv -k continuation_quality
    #
    #     example_id = input_item["example_id"]
    #                 ~~~~~~~~~~^^^^^^^^^^^^^^
    # KeyError: 'example_id'
    "continuation_quality",
    # Different project structure (uses src/ layout, no pyproject.toml at root)
    "mcp_env",
    # Requires BROWSERBASE_API_KEY, BROWSERBASE_PROJECT_ID, MODEL_API_KEY
    "browser_dom_example",
    # Requires BROWSERBASE_API_KEY, BROWSERBASE_PROJECT_ID, and running CUA server
    "browser_cua_example",
    # Uses prime-tunnel which is still experimental and has low usage limits
    "terminus_harbor",
    "opencode_harbor",
    # v1 SWE / container tasksets: need a docker/prime runtime + image-backed
    # sandboxes, so they can't run in plain CI — covered by dedicated v1 e2e tests.
    "r2e_gym_v1",
    "scaleswe_v1",
    "swelego_v1",
    "terminal_bench_2_v1",
]

SKIPPED_ENV_LOADING_ENVS = [
    # OpenEnv datasets are built by resetting seeds in sandbox-backed env servers.
    # Skip generic load checks here and cover via dedicated OpenEnv tests.
    "openenv_echo",
    "openenv_textarena",
    # R2E-Gym pulls a full image-backed SWE taskset; cover it with dedicated v1 tests.
    "rlm_swe_v1",
]

# v1 plugins are resolved by id (a `_v1` taskset, or the `compact` harness) instead of
# `verifiers.load_environment`, so they don't follow the v0 hub-env conventions (tags +
# README) and are evaluated through the unified `eval` CLI, not `vf-eval`.
V1_HARNESSES = {"compact"}

# Envs that install/import/load fine but can't run the capped smoke eval here:
#   self_reward: scores only with @group_reward (no individual reward), but the v0 legacy
#                `--id` bridge scores per-rollout (requires an individual-level reward).
SKIPPED_EVAL_ENVS = {"self_reward"}


def is_v1(env_dir: Path) -> bool:
    return env_dir.name.endswith("_v1") or env_dir.name in V1_HARNESSES


def get_environments() -> list[Path]:
    """Get all subdirectories of `environments/`, or only changed environments if CHANGED_ENVS is set."""
    all_envs = list(x for x in Path("environments").iterdir() if x.is_dir())

    # Filter out skipped environments
    all_envs = [env for env in all_envs if env.name not in SKIPPED_ENVS]

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
    """Test that the pyproject.toml file has the required metadata. `tags` are a v0 hub-env
    convention, so they're only required of v0 envs (v1 plugins are resolved by id)."""
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
    if not is_v1(env_dir):
        assert "tags" in pyproject["project"], "pyproject.toml does not have tags"
        assert pyproject["project"]["tags"] != ["placeholder-tag", "train", "eval"], (
            "Still uses placeholder tags"
        )


@pytest.mark.parametrize("env_dir", get_environments(), ids=lambda x: x.name)
def test_readme_exists(env_dir: Path):
    """Test that the README.md file exists for the given environment directory (v0 hub
    convention; v1 plugins are documented in their module docstring instead)."""
    if is_v1(env_dir):
        pytest.skip(f"{env_dir.name} is a v1 plugin (no hub README requirement)")
    assert (env_dir / "README.md").exists(), "README.md does not exist"


@pytest.mark.slow
@pytest.mark.parametrize("env_dir", get_environments(), ids=lambda x: x.name)
def test_env(env_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    """Test environment in a fresh venv with local verifiers installed first."""
    if env_dir.name in SKIPPED_ENVS:
        pytest.skip(f"Skipping {env_dir.name}")
    if env_dir.name in SKIPPED_ENV_LOADING_ENVS:
        pytest.skip(f"Skipping dedicated-runtime smoke test for {env_dir.name}")
    tmp_venv_dir = tmp_path_factory.mktemp(f"venv_{env_dir.name}")
    repo_root = Path(__file__).parent.parent
    cmd = (
        f"cd {tmp_venv_dir} && uv venv --clear && source .venv/bin/activate && "
        "uv pip install "
        f"{repo_root.as_posix()} && "
        "uv pip install "
        f"{(repo_root / 'packages' / 'tasksets').as_posix()} "
        f"{(repo_root / 'packages' / 'harnesses').as_posix()} && "
        "uv pip install "
        f"{env_dir.absolute().as_posix()}"
    )
    try:
        process = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {INSTALL_TIMEOUT}s installing {env_dir.name}")
    assert process.returncode == 0, (
        f"Failed to create virtual environment: {process.stderr}"
    )

    help_test_can_import_env(tmp_venv_dir, env_dir)
    help_test_can_load_env(tmp_venv_dir, env_dir)
    help_test_can_eval_env(tmp_venv_dir, env_dir)


def _run_in_venv(tmp_venv_dir: Path, inner: str, timeout: int, what: str, env_name: str):
    cmd = f"cd {tmp_venv_dir} && source .venv/bin/activate && {inner}"
    try:
        process = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {timeout}s {what} {env_name}")
    assert process.returncode == 0, (
        f"Failed to {what} {env_name}: {(process.stderr or process.stdout)[-2000:]}"
    )


def help_test_can_import_env(tmp_venv_dir: Path, env_dir: Path):
    """Test that the environment can be imported as a package."""
    _run_in_venv(
        tmp_venv_dir,
        f"uv run python -c 'import {env_dir.name}'",
        IMPORT_TIMEOUT,
        "importing",
        env_dir.name,
    )


def help_test_can_load_env(tmp_venv_dir: Path, env_dir: Path):
    """Test that the environment can be loaded — a v0 env via `verifiers.load_environment`,
    a v1 plugin via its id-based loader (taskset, or harness for `compact`)."""
    if env_dir.name in V1_HARNESSES:
        inner = (
            f"uv run python -c 'from verifiers.v1.loaders import harness_class; "
            f'harness_class("{env_dir.name}")\''
        )
    elif is_v1(env_dir):
        inner = (
            f"uv run python -c 'from verifiers.v1.loaders import taskset_class; "
            f'taskset_class("{env_dir.name}")\''
        )
    else:
        inner = (
            f'uv run python -c \'import verifiers as vf; vf.load_environment("{env_dir.name}")\''
        )
    _run_in_venv(tmp_venv_dir, inner, LOAD_TIMEOUT, "loading", env_dir.name)


def help_test_can_eval_env(tmp_venv_dir: Path, env_dir: Path):
    """Smoke-eval the environment through the unified `eval` CLI: a v1 taskset via
    `--taskset.id`/`--harness.id`, a v0 env via the legacy `--id` bridge. Capped to one
    short rollout so CI stays quick."""
    if env_dir.name in V1_HARNESSES:
        pytest.skip(f"{env_dir.name} is a harness, not an evaluatable taskset")
    if env_dir.name in SKIPPED_EVAL_ENVS:
        pytest.skip(f"{env_dir.name}: not runnable via the capped `eval` smoke test")
    if os.getenv("PRIME_API_KEY"):
        model_flags = (
            "-m openai/gpt-4.1-mini "
            "--client.base-url https://api.pinference.ai/api/v1 "
            "--client.api-key-var PRIME_API_KEY"
        )
    elif os.getenv("OPENAI_API_KEY"):
        model_flags = (
            "-m gpt-4.1-mini "
            "--client.base-url https://api.openai.com/v1 "
            "--client.api-key-var OPENAI_API_KEY"
        )
    else:
        pytest.skip("Skipping eval smoke test because no API key is configured")

    # `-r 2`: a taskset with @group_reward(s) needs >=2 rollouts to compare.
    caps = "-n 1 -r 2 --max-turns 4 --sampling.max-tokens 512 --rich false"
    if is_v1(env_dir):
        selector = f"--taskset.id {env_dir.name} --harness.id default"
    else:
        selector = f"--id {env_dir.name}"
    _run_in_venv(
        tmp_venv_dir,
        f"uv run eval {selector} {model_flags} {caps}",
        EVAL_TIMEOUT,
        "evaluating",
        env_dir.name,
    )
