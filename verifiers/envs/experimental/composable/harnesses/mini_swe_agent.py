"""mini-SWE-agent harness configuration."""

from __future__ import annotations

from pathlib import PurePosixPath
import shlex

DEFAULT_INSTALL_DIR = "/opt/mini-swe-agent"
DEFAULT_PREFIX_DIR = f"{DEFAULT_INSTALL_DIR}/prefix"
DEFAULT_SITE_PACKAGES_DIR = f"{DEFAULT_PREFIX_DIR}/site-packages"
DEFAULT_UV_SITE_PACKAGES_DIR = f"{DEFAULT_INSTALL_DIR}/uv-site-packages"
DEFAULT_MINI_BINARY = f"{DEFAULT_PREFIX_DIR}/bin/mini"
MINI_SWE_AGENT_CLI_PACKAGE = "mini-swe-agent"
MINI_SWE_AGENT_CLI_VERSION = "2.2.8"
MINI_SWE_AGENT_CLI_SHA256 = (
    "694df4de1337e665e3cd82e99f93374f573bf52b8e7c362ac5d8045ad9f7c37c"
)
MINI_SWE_AGENT_PYTHON_VERSION = "3.11"
UV_PACKAGE_VERSION = "0.11.7"
DEFAULT_PACKAGE_VERSION = MINI_SWE_AGENT_CLI_VERSION
DEFAULT_PACKAGE_SHA256 = MINI_SWE_AGENT_CLI_SHA256
DEFAULT_INSTRUCTION_PATH = "/mini-swe-agent/prompt.txt"
DEFAULT_SYSTEM_PROMPT_PATH = "/mini-swe-agent/system.txt"
DEFAULT_LOG_DIR = "/logs/agent"
DEFAULT_LOG_PATH = f"{DEFAULT_LOG_DIR}/mini-swe-agent.log"
DEFAULT_TRAJECTORY_PATH = f"{DEFAULT_LOG_DIR}/mini-swe-agent.traj.json"
DEFAULT_AGENT_WORKDIR = "${AGENT_WORKDIR:-/app}"
DEFAULT_CONFIG_SPEC = "mini_textbased"
DEFAULT_MODEL_CLASS = "litellm_textbased"
DEFAULT_ENVIRONMENT_TIMEOUT = 120


def build_mini_swe_agent_install_script(
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
    prefix_dir: str = DEFAULT_PREFIX_DIR,
    install_python: bool = True,
) -> str:
    """Build the shell script that installs mini-SWE-agent."""
    install_tools = ""
    if install_python:
        install_tools = """\
export DEBIAN_FRONTEND=noninteractive
if ! command -v python3 >/dev/null 2>&1 || ! python3 -m pip --version >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq python3 python3-pip ca-certificates
fi
"""

    quoted_prefix_dir = shlex.quote(prefix_dir)
    site_packages_dir = f"{prefix_dir}/site-packages"
    wheel_filename = f"mini_swe_agent-{package_version}-py3-none-any.whl"
    wheel_url = (
        f"https://files.pythonhosted.org/packages/py3/m/mini-swe-agent/{wheel_filename}"
    )
    quoted_site_packages_dir = shlex.quote(site_packages_dir)
    quoted_install_dir = shlex.quote(DEFAULT_INSTALL_DIR)
    quoted_uv_site_packages_dir = shlex.quote(DEFAULT_UV_SITE_PACKAGES_DIR)
    return f"""\
set -e
{install_tools}
rm -rf {quoted_prefix_dir}
mkdir -p {quoted_install_dir} {quoted_prefix_dir}/bin {quoted_site_packages_dir} {quoted_uv_site_packages_dir} {shlex.quote(DEFAULT_LOG_DIR)} /mini-swe-agent
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_BREAK_SYSTEM_PACKAGES=1
unset PIP_EXTRA_INDEX_URL
PYTHON_BIN="$(command -v python3)"
MINI_SWE_AGENT_PYTHON="$PYTHON_BIN"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  "$PYTHON_BIN" -m pip install --quiet --target {quoted_uv_site_packages_dir} uv=={UV_PACKAGE_VERSION}
  env PYTHONPATH={quoted_uv_site_packages_dir} "$PYTHON_BIN" -m uv python install {MINI_SWE_AGENT_PYTHON_VERSION}
  MINI_SWE_AGENT_PYTHON="$(env PYTHONPATH={quoted_uv_site_packages_dir} "$PYTHON_BIN" -m uv python find {MINI_SWE_AGENT_PYTHON_VERSION})"
fi
MINI_SWE_AGENT_WHEEL_DIR="$(mktemp -d)"
trap 'rm -rf "$MINI_SWE_AGENT_WHEEL_DIR"' EXIT
MINI_SWE_AGENT_WHEEL="$MINI_SWE_AGENT_WHEEL_DIR/{wheel_filename}"
MINI_SWE_AGENT_WHEEL_URL={shlex.quote(wheel_url)}
export MINI_SWE_AGENT_WHEEL MINI_SWE_AGENT_WHEEL_URL
"$PYTHON_BIN" -c 'import os, urllib.request; urllib.request.urlretrieve(os.environ["MINI_SWE_AGENT_WHEEL_URL"], os.environ["MINI_SWE_AGENT_WHEEL"])'
echo "{package_sha256}  $MINI_SWE_AGENT_WHEEL" | sha256sum -c -
if [ "$MINI_SWE_AGENT_PYTHON" = "$PYTHON_BIN" ]; then
  "$PYTHON_BIN" -m pip install --quiet --target {quoted_site_packages_dir} "$MINI_SWE_AGENT_WHEEL"
else
  env PYTHONPATH={quoted_uv_site_packages_dir} "$PYTHON_BIN" -m uv pip install --python "$MINI_SWE_AGENT_PYTHON" --target {quoted_site_packages_dir} "$MINI_SWE_AGENT_WHEEL"
fi
echo "$MINI_SWE_AGENT_PYTHON" > {quoted_prefix_dir}/python
cat > {quoted_prefix_dir}/bin/mini <<'EOF'
#!/usr/bin/env sh
export PYTHONPATH={shlex.quote(site_packages_dir)}:${{PYTHONPATH:-}}
exec "$(cat {quoted_prefix_dir}/python)" -m minisweagent.run.mini "$@"
EOF
chmod +x {quoted_prefix_dir}/bin/mini
test -x {quoted_prefix_dir}/bin/mini
"""


def build_mini_swe_agent_run_command(
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    trajectory_path: str = DEFAULT_TRAJECTORY_PATH,
    mini_binary: str = DEFAULT_MINI_BINARY,
    config_spec: str = DEFAULT_CONFIG_SPEC,
    model_class: str = DEFAULT_MODEL_CLASS,
    environment_timeout: int = DEFAULT_ENVIRONMENT_TIMEOUT,
    extra_config_specs: list[str] | None = None,
) -> str:
    """Build the shell command that configures and runs mini-SWE-agent.

    Config specs layer the cwd, timeout, LiteLLM model class, optional system
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
        "-c",
        "model.cost_tracking=ignore_errors",
        "-c",
        "model.model_kwargs.custom_llm_provider=openai",
    ]
    # Config specs are the mini CLI's native override format; use them for cwd,
    # timeout, model class, and optional system prompt wiring.
    for spec in extra_config_specs or []:
        config_args.extend(["-c", shlex.quote(spec)])

    log_dir = str(PurePosixPath(log_path).parent)
    trajectory_dir = str(PurePosixPath(trajectory_path).parent)
    script = f"""\
set -eo pipefail
export PATH={shlex.quote(DEFAULT_PREFIX_DIR)}/bin:"$PATH"
export PYTHONPATH={shlex.quote(DEFAULT_SITE_PACKAGES_DIR)}:"${{PYTHONPATH:-}}"
export MSWEA_CONFIGURED=true
export MSWEA_SILENT_STARTUP=true
export MSWEA_GLOBAL_CONFIG_DIR=/tmp/mini-swe-agent-config
export OPENAI_API_KEY="${{OPENAI_API_KEY:-intercepted}}"

{workdir_assignment}
mkdir -p {shlex.quote(log_dir)} {shlex.quote(trajectory_dir)} "$MINI_SWE_AGENT_WORKDIR" "$MSWEA_GLOBAL_CONFIG_DIR"

MINI_SWE_AGENT_TASK="$(cat {shlex.quote(instruction_path)})"
CONFIG_ARGS=({" ".join(config_args)})
CONFIG_ARGS+=(-c "environment.cwd=$MINI_SWE_AGENT_WORKDIR")
if [ -s {shlex.quote(system_prompt_path)} ]; then
  CONFIG_ARGS+=(-c "agent.system_template=$(cat {shlex.quote(system_prompt_path)})")
fi

cd "$MINI_SWE_AGENT_WORKDIR"
timeout --kill-after=30s "${{AGENT_TIMEOUT_SECONDS:-3600}}" {shlex.quote(mini_binary)} \\
  --model "$OPENAI_MODEL" \\
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
    task_system_prompt: str | None = None,
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    trajectory_path: str = DEFAULT_TRAJECTORY_PATH,
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
    config_spec: str = DEFAULT_CONFIG_SPEC,
    model_class: str = DEFAULT_MODEL_CLASS,
    environment_timeout: int = DEFAULT_ENVIRONMENT_TIMEOUT,
    extra_config_specs: list[str] | None = None,
):
    """Create a Harness configured for mini-SWE-agent."""
    from verifiers.envs.experimental.composable import Harness

    if task_system_prompt:
        if system_prompt:
            system_prompt = system_prompt + "\n" + task_system_prompt
        else:
            system_prompt = task_system_prompt

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
