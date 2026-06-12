import shlex
from pathlib import PurePosixPath

import verifiers as vf
from verifiers.v1.utils.sandbox_python_utils import python_runtime_setup_command

from .utils import split_versioned_agent_spec

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
MINI_SWE_AGENT_DEFAULT_VERSION = "mini-swe-agent@2.2.8"
MINI_SWE_AGENT_DEFAULT_PACKAGE = MINI_SWE_AGENT_DEFAULT_VERSION
MINI_SWE_AGENT_DEFAULT_CONFIG_SPEC = "mini"
MINI_SWE_AGENT_DEFAULT_MODEL_CLASS = "litellm"
MINI_SWE_AGENT_DEFAULT_ENVIRONMENT_TIMEOUT = 120


def build_mini_swe_agent_install_script(
    version: str = MINI_SWE_AGENT_DEFAULT_VERSION,
    prefix_dir: str = DEFAULT_PREFIX_DIR,
    package: str | None = None,
) -> str:
    root = shlex.quote(str(PurePosixPath(prefix_dir).parent))
    prefix = shlex.quote(prefix_dir)
    site_packages = shlex.quote(f"{prefix_dir.rstrip('/')}/site-packages")
    if package is not None:
        version = package
    name, parsed_version = split_versioned_agent_spec(version)
    requirement = name
    if parsed_version and parsed_version != "latest":
        requirement = f"{name}=={parsed_version}"
    return f"""\
set -e
{python_runtime_setup_command()}
rm -rf {prefix}
mkdir -p {root} {prefix}/bin {site_packages} {shlex.quote(DEFAULT_LOG_DIR)} /mini-swe-agent
vf_python_install --target {site_packages} {shlex.quote(requirement)}
echo "$VF_PYTHON" > {prefix}/python
cat > {prefix}/bin/mini <<'EOF'
#!/usr/bin/env sh
export PYTHONPATH={site_packages}:${{PYTHONPATH:-}}
exec "$(cat {prefix}/python)" -m minisweagent.run.mini "$@"
EOF
chmod +x {prefix}/bin/mini
test -x {prefix}/bin/mini
"""


class MiniSWEAgentProgramConfig(vf.ProgramConfig):
    agent_workdir: str = MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR
    instruction_path: str = MINI_SWE_AGENT_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = MINI_SWE_AGENT_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = MINI_SWE_AGENT_DEFAULT_LOG_PATH
    trajectory_path: str = MINI_SWE_AGENT_DEFAULT_TRAJECTORY_PATH
    config_spec: str = MINI_SWE_AGENT_DEFAULT_CONFIG_SPEC
    model_class: str = MINI_SWE_AGENT_DEFAULT_MODEL_CLASS
    environment_timeout: int = MINI_SWE_AGENT_DEFAULT_ENVIRONMENT_TIMEOUT
    parallel_tool_calls: bool = True
    extra_config_specs: list[str] | None = None
    sandbox: vf.SandboxConfig | None = vf.SandboxConfig()

    def resolve(
        self, version: str = MINI_SWE_AGENT_DEFAULT_VERSION
    ) -> vf.ProgramConfig:
        files: dict[str, vf.ProgramValue] = {
            self.instruction_path: {"fn": "verifiers.v1.utils.prompt_utils:task_text"},
            self.system_prompt_path: {
                "fn": "verifiers.v1.utils.prompt_utils:state_system_prompt_text"
            },
        }
        artifacts = vf.ArtifactsConfig.model_validate(
            {
                "mini_swe_agent_log": {
                    "path": self.log_path,
                    "format": "text",
                    "optional": True,
                },
                "mini_swe_agent_trajectory": {
                    "path": self.trajectory_path,
                    "format": "json",
                    "optional": True,
                },
            }
        )
        if self.agent_workdir == MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR:
            workdir_line = (
                f"MINI_SWE_AGENT_WORKDIR={MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR}"
            )
        else:
            workdir_line = f"MINI_SWE_AGENT_WORKDIR={shlex.quote(self.agent_workdir)}"

        config_args = [
            "-c",
            shlex.quote(self.config_spec),
            "-c",
            "agent.cost_limit=0",
            "-c",
            f"environment.timeout={self.environment_timeout}",
            "-c",
            f"model.model_class={shlex.quote(self.model_class)}",
            "-c",
            "model.cost_tracking=ignore_errors",
            "-c",
            "model.model_kwargs.custom_llm_provider=openai",
            "-c",
            f"model.model_kwargs.parallel_tool_calls={str(self.parallel_tool_calls).lower()}",
        ]
        for spec in self.extra_config_specs or []:
            config_args.extend(["-c", shlex.quote(spec)])

        setup = build_mini_swe_agent_install_script(
            version=version,
        )
        log_dir = str(PurePosixPath(self.log_path).parent)
        trajectory_dir = str(PurePosixPath(self.trajectory_path).parent)
        system_prompt_file = shlex.quote(self.system_prompt_path)
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

MINI_SWE_AGENT_TASK="$(cat {shlex.quote(self.instruction_path)})"
CONFIG_ARGS=({" ".join(config_args)})
CONFIG_ARGS+=(-c "environment.cwd=$MINI_SWE_AGENT_WORKDIR")
if [ -s {system_prompt_file} ]; then
  CONFIG_ARGS+=(-c "agent.system_template=$(cat {system_prompt_file})")
fi
cd "$MINI_SWE_AGENT_WORKDIR"
{shlex.quote(DEFAULT_MINI_BINARY)} \\
  --model "$OPENAI_MODEL" \\
  --task "$MINI_SWE_AGENT_TASK" \\
  --output {shlex.quote(self.trajectory_path)} \\
  --exit-immediately \\
  --yolo \\
  "${{CONFIG_ARGS[@]}}" 2>&1 | tee -a {shlex.quote(self.log_path)}
"""
        return self.resolve_command(
            command=["bash", "-lc", script],
            default_sandbox=self.sandbox,
            files=files,
            setup=setup,
            env={"OPENAI_MODEL": "runtime.model"},
            artifacts=artifacts,
        )


class MiniSWEAgentConfig(vf.HarnessConfig):
    version: str = MINI_SWE_AGENT_DEFAULT_VERSION
    program: MiniSWEAgentProgramConfig = MiniSWEAgentProgramConfig()
    max_turns: int = 4


class MiniSWEAgent(vf.Harness[MiniSWEAgentConfig]):
    config: MiniSWEAgentConfig

    def load_program_config(self, config: MiniSWEAgentConfig) -> vf.ProgramConfig:
        return config.program.resolve(version=config.version)


def load_harness(config: MiniSWEAgentConfig) -> MiniSWEAgent:
    return MiniSWEAgent(config=config)
