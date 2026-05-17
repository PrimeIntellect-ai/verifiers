import shlex
from pathlib import PurePosixPath

from verifiers.types import SamplingArgs

from .command import command_program, command_sandbox
from ...config import SandboxConfig
from ...harness import Harness
from ...utils.prompt_utils import (
    state_system_prompt_text,
    task_text as task_instruction_text,
)
from ...toolset import ToolsetCollection
from ...types import (
    ConfigMap,
    Handler,
    ModelClient,
    ProgramMap,
    ProgramOptionMap,
    ProgramValue,
    PromptInput,
)

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


class MiniSWEAgent(Harness):
    def __init__(
        self,
        *,
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
        install_python: bool = True,
        system_prompt: PromptInput | None = None,
        sandbox: bool | ConfigMap | SandboxConfig = True,
        program: ProgramMap | None = None,
        max_turns: int | None = 4,
        user: Handler | str | ConfigMap | None = None,
        client: ModelClient | None = None,
        model: str | None = None,
        sampling_args: SamplingArgs | None = None,
        toolsets: ToolsetCollection | None = None,
        stops: list[Handler] | None = None,
        setups: list[Handler] | None = None,
        updates: list[Handler] | None = None,
        metrics: list[Handler] | None = None,
        rewards: list[Handler] | None = None,
        advantages: list[Handler] | None = None,
        cleanups: list[Handler] | None = None,
    ):
        files: dict[str, ProgramValue] = {
            instruction_path: task_instruction_text,
        }
        if system_prompt is not None:
            files[system_prompt_path] = state_system_prompt_text
        artifacts: ProgramOptionMap = {
            "mini_swe_agent_log": {
                "path": log_path,
                "format": "text",
                "optional": True,
            },
            "mini_swe_agent_trajectory": {
                "path": trajectory_path,
                "format": "json",
                "optional": True,
            },
        }
        command = [
            "bash",
            "-lc",
            build_mini_swe_agent_run_script(
                agent_workdir=agent_workdir,
                instruction_path=instruction_path,
                system_prompt_path=system_prompt_path
                if system_prompt is not None
                else None,
                log_path=log_path,
                trajectory_path=trajectory_path,
                config_spec=config_spec,
                model_class=model_class,
                environment_timeout=environment_timeout,
                extra_config_specs=extra_config_specs,
            ),
        ]
        super().__init__(
            program=command_program(
                command=command,
                sandbox=sandbox,
                files=files,
                setup=build_mini_swe_agent_install_script(
                    package_version=package_version,
                    package_sha256=package_sha256,
                    install_python=install_python,
                ),
                env={
                    "OPENAI_MODEL": "runtime.model",
                },
                artifacts=artifacts,
                program=program,
            ),
            sandbox=command_sandbox(sandbox),
            system_prompt=system_prompt,
            max_turns=max_turns,
            user=user,
            client=client,
            model=model,
            sampling_args=sampling_args,
            toolsets=toolsets,
            stops=stops,
            setups=setups,
            updates=updates,
            metrics=metrics,
            rewards=rewards,
            advantages=advantages,
            cleanups=cleanups,
        )


def build_mini_swe_agent_install_script(
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
    prefix_dir: str = DEFAULT_PREFIX_DIR,
    install_python: bool = True,
) -> str:
    install_tools = ""
    if install_python:
        install_tools = """\
export DEBIAN_FRONTEND=noninteractive
if ! command -v python3 >/dev/null 2>&1 || ! python3 -m pip --version >/dev/null 2>&1; then
  apt-get -o Acquire::Retries=3 update -qq
  apt-get -o Acquire::Retries=3 install -y -qq python3 python3-pip ca-certificates
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


def build_mini_swe_agent_run_script(
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str | None = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    trajectory_path: str = DEFAULT_TRAJECTORY_PATH,
    mini_binary: str = DEFAULT_MINI_BINARY,
    config_spec: str = DEFAULT_CONFIG_SPEC,
    model_class: str = DEFAULT_MODEL_CLASS,
    environment_timeout: int = DEFAULT_ENVIRONMENT_TIMEOUT,
    extra_config_specs: list[str] | None = None,
) -> str:
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
