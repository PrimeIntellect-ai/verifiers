"""mini-SWE-agent harness configuration.

This injects a small OpenAI SDK model adapter so the harness can avoid mini's
default LiteLLM-backed model class.
"""

from __future__ import annotations

import shlex
from pathlib import Path

DEFAULT_INSTALL_DIR = "/opt/mini-swe-agent"
DEFAULT_VENV_DIR = f"{DEFAULT_INSTALL_DIR}/venv"
DEFAULT_MINI_BINARY = f"{DEFAULT_VENV_DIR}/bin/mini"
MINI_SWE_AGENT_CLI_PACKAGE = "mini-swe-agent"
MINI_SWE_AGENT_CLI_VERSION = "2.2.8"
MINI_SWE_AGENT_CLI_SHA256 = (
    "694df4de1337e665e3cd82e99f93374f573bf52b8e7c362ac5d8045ad9f7c37c"
)
DEFAULT_PACKAGE_VERSION = MINI_SWE_AGENT_CLI_VERSION
DEFAULT_PACKAGE_SHA256 = MINI_SWE_AGENT_CLI_SHA256
DEFAULT_INSTRUCTION_PATH = "/mini-swe-agent/prompt.txt"
DEFAULT_SYSTEM_PROMPT_PATH = "/mini-swe-agent/system.txt"
DEFAULT_LOG_DIR = "/logs/agent"
DEFAULT_LOG_PATH = f"{DEFAULT_LOG_DIR}/mini-swe-agent.log"
DEFAULT_TRAJECTORY_PATH = f"{DEFAULT_LOG_DIR}/mini-swe-agent.traj.json"
DEFAULT_AGENT_WORKDIR = "${AGENT_WORKDIR:-/app}"
DEFAULT_CONFIG_SPEC = "mini_textbased"
DEFAULT_OPENAI_MODEL_CLASS = "openai_text_model.OpenAITextModel"
DEFAULT_ENVIRONMENT_TIMEOUT = 120
OPENAI_TEXT_MODEL_SOURCE = (Path(__file__).parent / "openai_text_model.py").read_text()


def build_mini_swe_agent_install_script(
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
    venv_dir: str = DEFAULT_VENV_DIR,
    install_python: bool = True,
) -> str:
    """Build the shell script that installs mini-SWE-agent in an isolated venv.

    The script also writes the bundled OpenAI model adapter into the sandbox
    install dir so the mini CLI can import it through PYTHONPATH.
    """
    # The adapter module is installed next to the venv so mini can import it via
    # PYTHONPATH without modifying the upstream mini-SWE-agent package.
    install_tools = ""
    if install_python:
        install_tools = """\
if ! command -v python3 >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq python3 ca-certificates
fi
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq curl ca-certificates
fi
"""

    quoted_venv_dir = shlex.quote(venv_dir)
    quoted_install_dir = shlex.quote(DEFAULT_INSTALL_DIR)
    quoted_uv_bin_dir = shlex.quote(f"{DEFAULT_INSTALL_DIR}/uv-bin")
    quoted_uv_python_dir = shlex.quote(f"{DEFAULT_INSTALL_DIR}/uv-python")
    quoted_uv_cache_dir = shlex.quote(f"{DEFAULT_INSTALL_DIR}/uv-cache")
    return f"""\
set -e
{install_tools}
mkdir -p {quoted_install_dir} {quoted_uv_bin_dir} {quoted_uv_python_dir} {quoted_uv_cache_dir} {shlex.quote(DEFAULT_LOG_DIR)} /mini-swe-agent
UV_INSTALLER="$(mktemp)"
python3 -c 'import urllib.request; urllib.request.urlretrieve("https://astral.sh/uv/install.sh", "'"$UV_INSTALLER"'")'
UV_UNMANAGED_INSTALL={quoted_uv_bin_dir} sh "$UV_INSTALLER" --quiet
export PATH={quoted_uv_bin_dir}:"$PATH"
export UV_PYTHON_INSTALL_DIR={quoted_uv_python_dir}
export UV_CACHE_DIR={quoted_uv_cache_dir}
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL
uv --quiet venv --python 3.11 --seed {quoted_venv_dir}
{quoted_venv_dir}/bin/python -m pip install --quiet --upgrade pip
MINI_SWE_AGENT_WHEEL_DIR="$(mktemp -d)"
trap 'rm -rf "$MINI_SWE_AGENT_WHEEL_DIR"' EXIT
{quoted_venv_dir}/bin/python -m pip download --quiet --only-binary=:all: --no-deps --dest "$MINI_SWE_AGENT_WHEEL_DIR" {shlex.quote(f"{MINI_SWE_AGENT_CLI_PACKAGE}=={package_version}")}
MINI_SWE_AGENT_WHEEL="$(find "$MINI_SWE_AGENT_WHEEL_DIR" -maxdepth 1 -type f -name 'mini_swe_agent-*.whl' -print -quit)"
if [ -z "$MINI_SWE_AGENT_WHEEL" ]; then
  echo "pip download did not create a mini-SWE-agent wheel" >&2
  exit 1
fi
echo "{package_sha256}  $MINI_SWE_AGENT_WHEEL" | sha256sum -c -
{quoted_venv_dir}/bin/python -m pip install --quiet "$MINI_SWE_AGENT_WHEEL"
cat > {quoted_install_dir}/openai_text_model.py <<'MINI_SWE_AGENT_OPENAI_MODEL'
{OPENAI_TEXT_MODEL_SOURCE}
MINI_SWE_AGENT_OPENAI_MODEL
"""


def build_mini_swe_agent_run_command(
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    trajectory_path: str = DEFAULT_TRAJECTORY_PATH,
    mini_binary: str = DEFAULT_MINI_BINARY,
    config_spec: str = DEFAULT_CONFIG_SPEC,
    model_class: str = DEFAULT_OPENAI_MODEL_CLASS,
    environment_timeout: int = DEFAULT_ENVIRONMENT_TIMEOUT,
    extra_config_specs: list[str] | None = None,
) -> str:
    """Build the shell command that configures and runs mini-SWE-agent.

    Config specs layer the cwd, timeout, OpenAI adapter class, optional system
    prompt template, and any caller-provided overrides before writing the
    trajectory and teeing logs.
    """
    # Keep the default workdir shell-expanded for env-level overrides, mirroring
    # the other harnesses.
    if agent_workdir == DEFAULT_AGENT_WORKDIR:
        workdir_assignment = f"MINI_SWE_AGENT_WORKDIR={DEFAULT_AGENT_WORKDIR}"
    else:
        workdir_assignment = f"MINI_SWE_AGENT_WORKDIR={shlex.quote(agent_workdir)}"

    config_args = [
        "-c",
        shlex.quote(config_spec),
        "-c",
        "agent.cost_limit=0",
        "-c",
        f"environment.timeout={environment_timeout}",
        "-c",
        f"model.model_class={shlex.quote(model_class)}",
    ]
    # Config specs are the mini CLI's native override format; use them for cwd,
    # timeout, model class, and optional system prompt wiring.
    for spec in extra_config_specs or []:
        config_args.extend(["-c", shlex.quote(spec)])

    log_dir = log_path.rsplit("/", 1)[0]
    trajectory_dir = trajectory_path.rsplit("/", 1)[0]
    script = f"""\
set -eo pipefail
export PATH={shlex.quote(DEFAULT_VENV_DIR)}/bin:"$PATH"
export MSWEA_CONFIGURED=true
export MSWEA_SILENT_STARTUP=true
export MSWEA_GLOBAL_CONFIG_DIR=/tmp/mini-swe-agent-config
export OPENAI_API_KEY="${{OPENAI_API_KEY:-intercepted}}"
export PYTHONPATH={shlex.quote(DEFAULT_INSTALL_DIR)}:"${{PYTHONPATH:-}}"

{workdir_assignment}
mkdir -p {shlex.quote(log_dir)} {shlex.quote(trajectory_dir)} "$MINI_SWE_AGENT_WORKDIR" "$MSWEA_GLOBAL_CONFIG_DIR"

MINI_SWE_AGENT_TASK="$(cat {shlex.quote(instruction_path)})"
CONFIG_ARGS=({" ".join(config_args)})
CONFIG_ARGS+=(-c "environment.cwd=$MINI_SWE_AGENT_WORKDIR")
if [ -s {shlex.quote(system_prompt_path)} ]; then
  CONFIG_ARGS+=(-c "agent.system_template=$(cat {shlex.quote(system_prompt_path)})")
fi

cd "$MINI_SWE_AGENT_WORKDIR"
{shlex.quote(mini_binary)} \\
  --model "$OPENAI_MODEL" \\
  --model-class {shlex.quote(model_class)} \\
  --task "$MINI_SWE_AGENT_TASK" \\
  --output {shlex.quote(trajectory_path)} \\
  --exit-immediately \\
  --yolo \\
  "${{CONFIG_ARGS[@]}}" 2>&1 | tee -a {shlex.quote(log_path)}
"""
    return f"bash -lc {shlex.quote(script)}"


MINI_SWE_AGENT_INSTALL_SCRIPT = build_mini_swe_agent_install_script()
MINI_SWE_AGENT_CONFIG = {
    "install_script": MINI_SWE_AGENT_INSTALL_SCRIPT,
    "cli_package": MINI_SWE_AGENT_CLI_PACKAGE,
    "cli_version": MINI_SWE_AGENT_CLI_VERSION,
    "cli_sha256": MINI_SWE_AGENT_CLI_SHA256,
}


def mini_swe_agent_harness(
    system_prompt: str | None = None,
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    trajectory_path: str = DEFAULT_TRAJECTORY_PATH,
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
    config_spec: str = DEFAULT_CONFIG_SPEC,
    model_class: str = DEFAULT_OPENAI_MODEL_CLASS,
    environment_timeout: int = DEFAULT_ENVIRONMENT_TIMEOUT,
    extra_config_specs: list[str] | None = None,
):
    """Create a Harness configured for mini-SWE-agent."""
    from harnesses.base import Harness

    # The system prompt is passed through ComposableEnv as a file and injected
    # into mini's agent.system_template at runtime.
    return Harness(
        install_script=build_mini_swe_agent_install_script(
            package_version=package_version,
            package_sha256=package_sha256,
        ),
        run_command=build_mini_swe_agent_run_command(
            agent_workdir=agent_workdir,
            instruction_path=instruction_path,
            system_prompt_path=system_prompt_path,
            log_path=log_path,
            trajectory_path=trajectory_path,
            config_spec=config_spec,
            model_class=model_class,
            environment_timeout=environment_timeout,
            extra_config_specs=extra_config_specs,
        ),
        system_prompt=system_prompt,
        instruction_path=instruction_path,
        system_prompt_path=system_prompt_path,
        log_path=log_path,
    )
