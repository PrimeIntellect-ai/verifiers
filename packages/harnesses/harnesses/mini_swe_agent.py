import shlex
from pathlib import PurePosixPath

import verifiers as vf
from pydantic import model_validator
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


def build_mini_swe_agent_install_script(
    package_version: str = MINI_SWE_AGENT_DEFAULT_PACKAGE_VERSION,
    package_sha256: str = MINI_SWE_AGENT_DEFAULT_PACKAGE_SHA256,
    prefix_dir: str = DEFAULT_PREFIX_DIR,
) -> str:
    install_dir = str(PurePosixPath(prefix_dir).parent)
    site_packages_dir = f"{prefix_dir}/site-packages"
    setup_prefix_dir = shlex.quote(prefix_dir)
    setup_site_packages_dir = f"{prefix_dir}/site-packages"
    wheel_filename = f"mini_swe_agent-{package_version}-py3-none-any.whl"
    wheel_url = (
        f"https://files.pythonhosted.org/packages/py3/m/mini-swe-agent/{wheel_filename}"
    )
    return f"""\
set -e
{python_runtime_setup_command()}
rm -rf {setup_prefix_dir}
mkdir -p {shlex.quote(install_dir)} {setup_prefix_dir}/bin {shlex.quote(site_packages_dir)} {shlex.quote(DEFAULT_LOG_DIR)} /mini-swe-agent
MINI_SWE_AGENT_WHEEL_DIR="$(mktemp -d)"
trap 'rm -rf "$MINI_SWE_AGENT_WHEEL_DIR"' EXIT
MINI_SWE_AGENT_WHEEL="$MINI_SWE_AGENT_WHEEL_DIR/{wheel_filename}"
MINI_SWE_AGENT_WHEEL_URL={shlex.quote(wheel_url)}
export MINI_SWE_AGENT_WHEEL MINI_SWE_AGENT_WHEEL_URL
"$VF_PYTHON" -c 'import os, urllib.request; urllib.request.urlretrieve(os.environ["MINI_SWE_AGENT_WHEEL_URL"], os.environ["MINI_SWE_AGENT_WHEEL"])'
echo "{package_sha256}  $MINI_SWE_AGENT_WHEEL" | sha256sum -c -
vf_python_install --target {shlex.quote(setup_site_packages_dir)} "$MINI_SWE_AGENT_WHEEL"
echo "$VF_PYTHON" > {setup_prefix_dir}/python
cat > {setup_prefix_dir}/bin/mini <<'EOF'
#!/usr/bin/env sh
export PYTHONPATH={shlex.quote(setup_site_packages_dir)}:${{PYTHONPATH:-}}
exec "$(cat {setup_prefix_dir}/python)" -m minisweagent.run.mini "$@"
EOF
chmod +x {setup_prefix_dir}/bin/mini
test -x {setup_prefix_dir}/bin/mini
"""


class MiniSWEAgentConfig(vf.HarnessConfig):
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
    system_prompt: vf.PromptInput | None = None
    sandbox: vf.SandboxConfig | None = vf.SandboxConfig()
    max_turns: int = 4

    @model_validator(mode="after")
    def configure_program(self) -> "MiniSWEAgentConfig":
        if self.program.command is not None and "program" not in self.model_fields_set:
            return self
        config = self
        files: dict[str, vf.ProgramValue] = {
            config.instruction_path: {
                "fn": "verifiers.v1.utils.prompt_utils:task_text"
            },
        }
        if config.system_prompt is not None:
            files[config.system_prompt_path] = {
                "fn": "verifiers.v1.utils.prompt_utils:state_system_prompt_text"
            }
        artifacts: dict[str, vf.ProgramValue] = {
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
        if config.agent_workdir == MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR:
            workdir_assignment = (
                f"MINI_SWE_AGENT_WORKDIR={MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR}"
            )
        else:
            workdir_assignment = (
                f"MINI_SWE_AGENT_WORKDIR={shlex.quote(config.agent_workdir)}"
            )

        config_args = [
            "-c",
            shlex.quote(config.config_spec),
            "-c",
            "agent.cost_limit=0",
            "-c",
            f"environment.timeout={config.environment_timeout}",
            "-c",
            f"model.model_class={shlex.quote(config.model_class)}",
            "-c",
            "model.cost_tracking=ignore_errors",
            "-c",
            "model.model_kwargs.custom_llm_provider=openai",
        ]
        for spec in config.extra_config_specs or []:
            config_args.extend(["-c", shlex.quote(spec)])

        setup = build_mini_swe_agent_install_script(
            package_version=config.package_version,
            package_sha256=config.package_sha256,
            prefix_dir=DEFAULT_PREFIX_DIR,
        )
        log_dir = str(PurePosixPath(config.log_path).parent)
        trajectory_dir = str(PurePosixPath(config.trajectory_path).parent)
        system_prompt_path = (
            config.system_prompt_path if config.system_prompt is not None else None
        )
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

MINI_SWE_AGENT_TASK="$(cat {shlex.quote(config.instruction_path)})"
CONFIG_ARGS=({" ".join(config_args)})
CONFIG_ARGS+=(-c "environment.cwd=$MINI_SWE_AGENT_WORKDIR")
{system_prompt_block}
cd "$MINI_SWE_AGENT_WORKDIR"
timeout --kill-after=30s "${{AGENT_TIMEOUT_SECONDS:-3600}}" {shlex.quote(DEFAULT_MINI_BINARY)} \\
  --model "$OPENAI_MODEL" \\
  --task "$MINI_SWE_AGENT_TASK" \\
  --output {shlex.quote(config.trajectory_path)} \\
  --exit-immediately \\
  --yolo \\
  "${{CONFIG_ARGS[@]}}" 2>&1 | tee -a {shlex.quote(config.log_path)}
"""
        self.program = vf.ProgramConfig.from_command(
            command=["bash", "-lc", script],
            program=config.program,
            default_sandbox=config.sandbox,
            files=files,
            setup=setup,
            env={"OPENAI_MODEL": "runtime.model"},
            artifacts=artifacts,
        )
        self.model_fields_set.discard("program")
        return self


class MiniSWEAgent(vf.Harness[MiniSWEAgentConfig]):
    config: MiniSWEAgentConfig


def load_harness(config: MiniSWEAgentConfig) -> MiniSWEAgent:
    return MiniSWEAgent(config=config)
