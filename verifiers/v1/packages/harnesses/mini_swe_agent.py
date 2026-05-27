import shlex
from pathlib import PurePosixPath

from .command import configure_command_harness
from .configs import (
    MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR,
    MiniSWEAgentConfig,
)
from ...harness import Harness
from ...types import ProgramCommand, ProgramOptionMap, ProgramSetup
from ...utils.sandbox_python_utils import python_runtime_setup_command

DEFAULT_INSTALL_DIR = "/opt/mini-swe-agent"
DEFAULT_PREFIX_DIR = f"{DEFAULT_INSTALL_DIR}/prefix"
DEFAULT_SITE_PACKAGES_DIR = f"{DEFAULT_PREFIX_DIR}/site-packages"
DEFAULT_MINI_BINARY = f"{DEFAULT_PREFIX_DIR}/bin/mini"
DEFAULT_LOG_DIR = "/logs/agent"


class MiniSWEAgent(Harness):
    def __init__(self, config: MiniSWEAgentConfig | None = None):
        config = MiniSWEAgentConfig() if config is None else config
        assert isinstance(config, MiniSWEAgentConfig)
        super().__init__(config=config.model_copy(update={"program": None}))
        self.config = config
        configure_command_harness(
            self,
            config,
            command=self.command(config),
            setup=self.setup(config),
            env=self.env(config),
            artifacts=self.artifacts(config),
        )

    def command(self, config: MiniSWEAgentConfig) -> ProgramCommand:
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

    def setup(self, config: MiniSWEAgentConfig) -> ProgramSetup:
        return build_mini_swe_agent_install_script(
            package_version=config.package_version,
            package_sha256=config.package_sha256,
        )

    def env(self, config: MiniSWEAgentConfig) -> ProgramOptionMap:
        return {"OPENAI_MODEL": "runtime.model"}

    def artifacts(self, config: MiniSWEAgentConfig) -> ProgramOptionMap:
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


def load_harness(config: MiniSWEAgentConfig) -> MiniSWEAgent:
    return MiniSWEAgent(config=config)


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
