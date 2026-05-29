import shlex
from collections.abc import Mapping
from typing import cast

import verifiers as vf
from pydantic import model_validator
from verifiers.v1.sandbox import sandbox_config_mapping
from verifiers.v1.utils.program_utils import int_config

from .utils.rlm_utils import (
    DEFAULT_RLM_CHECKOUT_PATH,
    DEFAULT_RLM_SKILLS_PATH,
    DEFAULT_RLM_TOOL_SKILL_MARKER,
    DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH,
    DEFAULT_RLM_TOOL_SKILLS_MANIFEST_NAME,
)

RLM_DEFAULT_REPO_URL = "github.com/PrimeIntellect-ai/rlm-harness.git"
RLM_DEFAULT_REPO_REF = "main"
RLM_DEFAULT_MAX_TURNS = 100
RLM_DEFAULT_EXEC_TIMEOUT = 300
RLM_DEFAULT_MAX_DEPTH = 0
RLM_DEFAULT_INSTRUCTION_PATH = "/rlm/instruction.txt"
RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/rlm/append_to_system_prompt.txt"
RLM_DEFAULT_WORKDIR = "/workspace"
RLM_DEFAULT_TOOLS = ["ipython"]


class RLMConfig(vf.HarnessConfig):
    workdir: str = RLM_DEFAULT_WORKDIR
    instruction_path: str = RLM_DEFAULT_INSTRUCTION_PATH
    rlm_repo_url: str = RLM_DEFAULT_REPO_URL
    rlm_repo_ref: str = RLM_DEFAULT_REPO_REF
    rlm_max_turns: int = RLM_DEFAULT_MAX_TURNS
    rlm_exec_timeout: int = RLM_DEFAULT_EXEC_TIMEOUT
    rlm_max_depth: int = RLM_DEFAULT_MAX_DEPTH
    summarize_at_tokens: int | None = None
    append_to_system_prompt: str = ""
    local_checkout: str | None = None
    gh_token_var: str | None = "GH_TOKEN"
    rlm_tools: list[str] = RLM_DEFAULT_TOOLS
    env_vars: dict[str, str] = {}
    skills: str | None = None

    @model_validator(mode="after")
    def configure_program(self) -> "RLMConfig":
        if self.program.command is not None and "program" not in self.model_fields_set:
            return self
        config = self
        files: dict[str, vf.ProgramValue] = {
            config.instruction_path: {
                "fn": "verifiers.v1.utils.prompt_utils:task_text",
                "keys": ["instruction", "question"],
            },
            RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH: config.append_to_system_prompt,
            DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH: {
                "fn": "harnesses.utils.rlm_utils:rlm_tool_skills_archive"
            },
        }
        dirs: dict[str, vf.ProgramValue] = {
            DEFAULT_RLM_CHECKOUT_PATH: {
                "fn": "harnesses.utils.rlm_utils:rlm_checkout_path",
                **(
                    {"local_checkout": config.local_checkout}
                    if config.local_checkout
                    else {}
                ),
                "rlm_repo_url": config.rlm_repo_url,
                "rlm_repo_ref": config.rlm_repo_ref,
                **(
                    {"gh_token_var": config.gh_token_var} if config.gh_token_var else {}
                ),
            }
        }
        if config.skills is not None:
            dirs[DEFAULT_RLM_SKILLS_PATH] = config.skills
        else:
            dirs[DEFAULT_RLM_SKILLS_PATH] = {
                "fn": "harnesses.utils.rlm_utils:rlm_skills_dir"
            }

        env: dict[str, vf.ProgramValue] = {
            "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "OPENAI_MODEL": "runtime.model",
            "RLM_MODEL": "runtime.model",
            "RLM_TOOLS": ",".join(config.rlm_tools),
            "RLM_MAX_TURNS": str(config.rlm_max_turns),
            "RLM_EXEC_TIMEOUT": str(config.rlm_exec_timeout),
            "RLM_MAX_DEPTH": str(config.rlm_max_depth),
            **config.env_vars,
        }
        if config.summarize_at_tokens is not None:
            assert config.summarize_at_tokens > 0
            env["RLM_SUMMARIZE_AT_TOKENS"] = str(config.summarize_at_tokens)

        artifacts: dict[str, vf.ProgramValue] = {
            "rlm_metrics": {
                "path": f"{config.workdir}/.rlm/sessions/*/meta.json",
                "format": "json",
                "key": "metrics",
                "optional": True,
            }
        }
        setup_timeout = max(config.rlm_exec_timeout + 120, 600)
        if config.sandbox is not None:
            explicit_sandbox_options = (
                sandbox_config_mapping(config.sandbox, fill_defaults=False) or {}
            )
            if explicit_sandbox_options.get("setup_timeout") is not None:
                setup_timeout = int_config(
                    explicit_sandbox_options, "setup_timeout", setup_timeout
                )

        if config.sandbox is None:
            sandbox: vf.ConfigData | vf.SandboxConfig | bool = {
                "image": "python:3.11-slim",
                "workdir": config.workdir,
                "cpu_cores": 1,
                "memory_gb": 2,
                "disk_size_gb": 5,
                "network_access": True,
                "timeout_minutes": 60,
                "command_timeout": max(config.rlm_exec_timeout + 120, 600),
                "setup_timeout": setup_timeout,
            }
        else:
            sandbox_options = sandbox_config_mapping(config.sandbox) or {}
            sandbox = {
                "workdir": config.workdir,
                "command_timeout": max(config.rlm_exec_timeout + 120, 600),
                **sandbox_options,
                "setup_timeout": setup_timeout,
            }

        skills_install_script = f"""
set -eo pipefail
skills_path={shlex.quote(DEFAULT_RLM_SKILLS_PATH)}
archive_path={shlex.quote(DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH)}
manifest_path="$skills_path/{DEFAULT_RLM_TOOL_SKILLS_MANIFEST_NAME}"
mkdir -p "$skills_path"
if [ -f "$manifest_path" ]; then
  while IFS= read -r skill_name; do
    case "$skill_name" in ""|.*|*/*|*..*) continue ;; esac
    if [ -f "$skills_path/$skill_name/{DEFAULT_RLM_TOOL_SKILL_MARKER}" ]; then
      rm -rf "$skills_path/$skill_name"
    fi
  done < "$manifest_path"
  rm -f "$manifest_path"
fi
if [ -s "$archive_path" ]; then
  tmp_archive="$(mktemp)"
  trap 'rm -f "$tmp_archive"' EXIT
  base64 -d "$archive_path" > "$tmp_archive"
  tar -tzf "$tmp_archive" \\
    | awk -F/ 'NF > 1 && $1 != "" {{print $1}}' \\
    | sort -u > "$manifest_path"
  tar -xzf "$tmp_archive" -C "$skills_path"
fi
"""
        checkout_install_script = f"""
set -eo pipefail
export RLM_CHECKOUT_PATH={shlex.quote(DEFAULT_RLM_CHECKOUT_PATH)}
test -f "$RLM_CHECKOUT_PATH/install.sh"
bash "$RLM_CHECKOUT_PATH/install.sh"
"""
        run_script = f"""
set -eo pipefail
export PATH="$HOME/.local/bin:${{AGENT_PATH:-$PATH}}"
export RLM_MODEL="${{RLM_MODEL:-$OPENAI_MODEL}}"
export OPENAI_API_KEY="${{OPENAI_API_KEY:-intercepted}}"
export RLM_APPEND_TO_SYSTEM_PROMPT="$(cat {shlex.quote(RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH)} 2>/dev/null || true)"
cd "${{AGENT_WORKDIR:-{config.workdir}}}"
rlm "$(cat {shlex.quote(config.instruction_path)})"
"""
        self.program = vf.ProgramConfig.from_command(
            command=["bash", "-lc", run_script],
            program=config.program,
            default_sandbox=config.sandbox,
            files=files,
            dirs=dirs,
            setup=[
                "apt-get -o Acquire::Retries=3 update && "
                "apt-get -o Acquire::Retries=3 install -y --no-install-recommends "
                "ca-certificates curl git && rm -rf /var/lib/apt/lists/*",
                "bash -lc " + shlex.quote(skills_install_script),
                "bash -lc " + shlex.quote(checkout_install_script),
            ],
            env=env,
            artifacts=artifacts,
            sandbox=sandbox,
            setup_timeout=setup_timeout,
        )
        self.model_fields_set.discard("program")
        return self


class RLMEndpoint(vf.Endpoint):
    def trajectory_visibility(self, headers: vf.ConfigMap):
        if str(headers.get("x-rlm-depth", "0")) != "0":
            return "hidden"
        return super().trajectory_visibility(headers)


class RLM(vf.Harness[RLMConfig]):
    config: RLMConfig

    def load_endpoint(self) -> vf.Endpoint:
        return RLMEndpoint(
            use_tunnel=self.program_sandbox_config(self.program) is not None
        )

    @vf.metric
    async def rlm_sub_llm_call_count(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return 0.0
        metrics = cast(vf.ConfigMap, artifacts).get("rlm_metrics")
        if not isinstance(metrics, Mapping):
            return 0.0
        value = cast(vf.ConfigMap, metrics).get("sub_llm_call_count", 0.0)
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            return 0.0
        return float(value or 0.0)

    @vf.metric
    async def rlm_sub_llm_total_turns(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return 0.0
        metrics = cast(vf.ConfigMap, artifacts).get("rlm_metrics")
        if not isinstance(metrics, Mapping):
            return 0.0
        value = cast(vf.ConfigMap, metrics).get("sub_llm_total_turns", 0.0)
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            return 0.0
        return float(value or 0.0)

    @vf.metric
    async def rlm_sub_llm_total_tool_calls(
        self, task: vf.Task, state: vf.State
    ) -> float:
        _ = task
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return 0.0
        metrics = cast(vf.ConfigMap, artifacts).get("rlm_metrics")
        if not isinstance(metrics, Mapping):
            return 0.0
        value = cast(vf.ConfigMap, metrics).get("sub_llm_total_tool_calls", 0.0)
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            return 0.0
        return float(value or 0.0)


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)
