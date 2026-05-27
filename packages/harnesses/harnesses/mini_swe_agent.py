import shlex
from pathlib import PurePosixPath

from verifiers.v1.config import HarnessConfig, PromptInput, SandboxConfig
from verifiers.v1.harness import Harness
from verifiers.v1.program import Program
from verifiers.v1.types import (
    ConfigData,
    ConfigMap,
    ProgramCommand,
    ProgramOptionMap,
    ProgramSetup,
    ProgramValue,
)
from verifiers.v1.utils.sandbox_python_utils import python_runtime_setup_command

DEFAULT_INSTALL_DIR = "/opt/mini-swe-agent"
DEFAULT_PREFIX_DIR = f"{DEFAULT_INSTALL_DIR}/prefix"
DEFAULT_SITE_PACKAGES_DIR = f"{DEFAULT_PREFIX_DIR}/site-packages"
DEFAULT_MINI_BINARY = f"{DEFAULT_PREFIX_DIR}/bin/mini"
DEFAULT_LOG_DIR = "/logs/agent"
MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR = "${AGENT_WORKDIR:-/app}"
MINI_SWE_AGENT_DEFAULT_INSTRUCTION_PATH = "/mini-swe-agent/prompt.txt"
MINI_SWE_AGENT_DEFAULT_SYSTEM_PROMPT_PATH = "/mini-swe-agent/system.txt"
MINI_SWE_AGENT_DEFAULT_LOG_PATH = "/logs/agent/mini-swe-agent.log"
MINI_SWE_AGENT_DEFAULT_TRAJECTORY_PATH = "/logs/agent/mini-swe-agent.traj.json"
MINI_SWE_AGENT_DEFAULT_PACKAGE_VERSION = "2.2.8"
MINI_SWE_AGENT_DEFAULT_PACKAGE_SHA256 = (
    "694df4de1337e665e3cd82e99f93374f573bf52b8e7c362ac5d8045ad9f7c37c"
)
MINI_SWE_AGENT_DEFAULT_CONFIG_SPEC = "mini_textbased"
MINI_SWE_AGENT_DEFAULT_MODEL_CLASS = "litellm_textbased"
MINI_SWE_AGENT_DEFAULT_ENVIRONMENT_TIMEOUT = 120


class MiniSWEAgentConfig(HarnessConfig):
    agent_workdir: str = MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR
    instruction_path: str = MINI_SWE_AGENT_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = MINI_SWE_AGENT_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = MINI_SWE_AGENT_DEFAULT_LOG_PATH
    trajectory_path: str = MINI_SWE_AGENT_DEFAULT_TRAJECTORY_PATH
    package_version: str = MINI_SWE_AGENT_DEFAULT_PACKAGE_VERSION
    package_sha256: str = MINI_SWE_AGENT_DEFAULT_PACKAGE_SHA256
    config_spec: str = MINI_SWE_AGENT_DEFAULT_CONFIG_SPEC
    model_class: str = MINI_SWE_AGENT_DEFAULT_MODEL_CLASS
    environment_timeout: int = MINI_SWE_AGENT_DEFAULT_ENVIRONMENT_TIMEOUT
    extra_config_specs: list[str] | None = None
    system_prompt: PromptInput | None = None
    sandbox: SandboxConfig | None = SandboxConfig()
    max_turns: int = 4


class MiniSWEAgent(Harness[MiniSWEAgentConfig]):
    config: MiniSWEAgentConfig

    def load_program(self) -> Program:
        program, _ = mini_swe_agent_program_config(self.config)
        return program

    def load_sandbox(self) -> ConfigMap | None:
        _, sandbox = mini_swe_agent_program_config(self.config)
        return sandbox


def load_harness(config: MiniSWEAgentConfig) -> MiniSWEAgent:
    return MiniSWEAgent(config=config)


def mini_swe_agent_program_config(
    config: MiniSWEAgentConfig,
) -> tuple[Program, ConfigData | None]:
    return Harness.command_program_config(
        config,
        command=mini_swe_agent_command(config),
        files=mini_swe_agent_files(config),
        setup=mini_swe_agent_setup(config),
        env=mini_swe_agent_env(config),
        artifacts=mini_swe_agent_artifacts(config),
    )


def mini_swe_agent_command(config: MiniSWEAgentConfig) -> ProgramCommand:
    return [
        "bash",
        "-lc",
        build_mini_swe_agent_run_script(
            agent_workdir=config.agent_workdir,
            instruction_path=config.instruction_path,
            system_prompt_path=config.system_prompt_path
            if config.system_prompt is not None
            else None,
            log_path=config.log_path,
            trajectory_path=config.trajectory_path,
            config_spec=config.config_spec,
            model_class=config.model_class,
            environment_timeout=config.environment_timeout,
            extra_config_specs=config.extra_config_specs,
        ),
    ]


def mini_swe_agent_setup(config: MiniSWEAgentConfig) -> ProgramSetup:
    return build_mini_swe_agent_install_script(
        package_version=config.package_version,
        package_sha256=config.package_sha256,
    )


def mini_swe_agent_files(config: MiniSWEAgentConfig) -> ProgramOptionMap:
    files: dict[str, ProgramValue] = {
        config.instruction_path: {"fn": "verifiers.v1.utils.prompt_utils:task_text"},
    }
    if config.system_prompt is not None:
        files[config.system_prompt_path] = {
            "fn": "verifiers.v1.utils.prompt_utils:state_system_prompt_text"
        }
    return files


def mini_swe_agent_env(config: MiniSWEAgentConfig) -> ProgramOptionMap:
    return {"OPENAI_MODEL": "runtime.model"}


def mini_swe_agent_artifacts(config: MiniSWEAgentConfig) -> ProgramOptionMap:
    return {
        "mini_swe_agent_log": {
            "path": config.log_path,
            "format": "text",
            "optional": True,
        },
        "mini_swe_agent_trajectory": {
            "path": config.trajectory_path,
            "format": "json",
            "optional": True,
        },
    }


def build_mini_swe_agent_install_script(
    package_version: str,
    package_sha256: str,
    prefix_dir: str = DEFAULT_PREFIX_DIR,
) -> str:
    quoted_prefix_dir = shlex.quote(prefix_dir)
    site_packages_dir = f"{prefix_dir}/site-packages"
    wheel_filename = f"mini_swe_agent-{package_version}-py3-none-any.whl"
    wheel_url = (
        f"https://files.pythonhosted.org/packages/py3/m/mini-swe-agent/{wheel_filename}"
    )
    quoted_site_packages_dir = shlex.quote(site_packages_dir)
    quoted_install_dir = shlex.quote(DEFAULT_INSTALL_DIR)
    return f"""\
set -e
{python_runtime_setup_command()}
rm -rf {quoted_prefix_dir}
mkdir -p {quoted_install_dir} {quoted_prefix_dir}/bin {quoted_site_packages_dir} {shlex.quote(DEFAULT_LOG_DIR)} /mini-swe-agent
MINI_SWE_AGENT_WHEEL_DIR="$(mktemp -d)"
trap 'rm -rf "$MINI_SWE_AGENT_WHEEL_DIR"' EXIT
MINI_SWE_AGENT_WHEEL="$MINI_SWE_AGENT_WHEEL_DIR/{wheel_filename}"
MINI_SWE_AGENT_WHEEL_URL={shlex.quote(wheel_url)}
export MINI_SWE_AGENT_WHEEL MINI_SWE_AGENT_WHEEL_URL
"$VF_PYTHON" -c 'import os, urllib.request; urllib.request.urlretrieve(os.environ["MINI_SWE_AGENT_WHEEL_URL"], os.environ["MINI_SWE_AGENT_WHEEL"])'
echo "{package_sha256}  $MINI_SWE_AGENT_WHEEL" | sha256sum -c -
vf_python_install --target {quoted_site_packages_dir} "$MINI_SWE_AGENT_WHEEL"
echo "$VF_PYTHON" > {quoted_prefix_dir}/python
cat > {quoted_prefix_dir}/bin/mini <<'EOF'
#!/usr/bin/env sh
export PYTHONPATH={shlex.quote(site_packages_dir)}:${{PYTHONPATH:-}}
exec "$(cat {quoted_prefix_dir}/python)" -m minisweagent.run.mini "$@"
EOF
chmod +x {quoted_prefix_dir}/bin/mini
test -x {quoted_prefix_dir}/bin/mini
"""


def build_mini_swe_agent_run_script(
    agent_workdir: str,
    instruction_path: str,
    system_prompt_path: str | None,
    log_path: str,
    trajectory_path: str,
    config_spec: str,
    model_class: str,
    environment_timeout: int,
    mini_binary: str = DEFAULT_MINI_BINARY,
    extra_config_specs: list[str] | None = None,
) -> str:
    if agent_workdir == MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR:
        workdir_assignment = (
            f"MINI_SWE_AGENT_WORKDIR={MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR}"
        )
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
    for spec in extra_config_specs or []:
        config_args.extend(["-c", shlex.quote(spec)])

    log_dir = str(PurePosixPath(log_path).parent)
    trajectory_dir = str(PurePosixPath(trajectory_path).parent)
    system_prompt_block = ""
    if system_prompt_path is not None:
        system_prompt_block = f"""\
if [ -s {shlex.quote(system_prompt_path)} ]; then
  CONFIG_ARGS+=(-c "agent.system_template=$(cat {shlex.quote(system_prompt_path)})")
fi
"""
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
{system_prompt_block}
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
