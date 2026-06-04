"""mini-SWE-agent harness configuration."""

from pathlib import PurePosixPath
import shlex

from harnesses.mini_swe_agent import (
    MINI_SWE_AGENT_DEFAULT_VERSION,
    build_mini_swe_agent_install_script as build_packaged_install_script,
)

DEFAULT_INSTALL_DIR = "/opt/mini-swe-agent"
DEFAULT_PREFIX_DIR = f"{DEFAULT_INSTALL_DIR}/prefix"
DEFAULT_SITE_PACKAGES_DIR = f"{DEFAULT_PREFIX_DIR}/site-packages"
DEFAULT_MINI_BINARY = f"{DEFAULT_PREFIX_DIR}/bin/mini"
MINI_SWE_AGENT_CLI_PACKAGE = "mini-swe-agent"
MINI_SWE_AGENT_CLI_VERSION = "2.2.8"
DEFAULT_VERSION = MINI_SWE_AGENT_DEFAULT_VERSION
DEFAULT_INSTRUCTION_PATH = "/mini-swe-agent/prompt.txt"
DEFAULT_SYSTEM_PROMPT_PATH = "/mini-swe-agent/system.txt"
DEFAULT_LOG_DIR = "/logs/agent"
DEFAULT_LOG_PATH = f"{DEFAULT_LOG_DIR}/mini-swe-agent.log"
DEFAULT_TRAJECTORY_PATH = f"{DEFAULT_LOG_DIR}/mini-swe-agent.traj.json"
DEFAULT_AGENT_WORKDIR = "${AGENT_WORKDIR:-/app}"
DEFAULT_CONFIG_SPEC = "mini"
DEFAULT_MODEL_CLASS = "litellm"
DEFAULT_ENVIRONMENT_TIMEOUT = 120


def build_mini_swe_agent_install_script(
    version: str = DEFAULT_VERSION,
    prefix_dir: str = DEFAULT_PREFIX_DIR,
) -> str:
    """Build the shell script that installs mini-SWE-agent."""
    return build_packaged_install_script(
        version=version,
        prefix_dir=prefix_dir,
    )


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
    parallel_tool_calls: bool = True,
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
        workdir_line = f"MINI_SWE_AGENT_WORKDIR={DEFAULT_AGENT_WORKDIR}"
    else:
        workdir_line = f"MINI_SWE_AGENT_WORKDIR={shlex.quote(agent_workdir)}"

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
        "-c",
        f"model.model_kwargs.parallel_tool_calls={str(parallel_tool_calls).lower()}",
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

{workdir_line}
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
}


def mini_swe_agent_harness(
    system_prompt: str | None = None,
    task_system_prompt: str | None = None,
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    trajectory_path: str = DEFAULT_TRAJECTORY_PATH,
    version: str = DEFAULT_VERSION,
    config_spec: str = DEFAULT_CONFIG_SPEC,
    model_class: str = DEFAULT_MODEL_CLASS,
    environment_timeout: int = DEFAULT_ENVIRONMENT_TIMEOUT,
    parallel_tool_calls: bool = True,
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
            version=version,
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
            parallel_tool_calls=parallel_tool_calls,
            extra_config_specs=extra_config_specs,
        ),
        system_prompt=system_prompt,
        instruction_path=instruction_path,
        system_prompt_path=system_prompt_path,
        log_path=log_path,
    )
