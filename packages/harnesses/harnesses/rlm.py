import shlex

import verifiers as vf

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


class RLMProgramConfig(vf.ProgramConfig):
    sandbox: vf.SandboxConfig | None = None
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

    def resolve(self) -> vf.ProgramConfig:
        files: dict[str, vf.ProgramValue] = {
            self.instruction_path: {
                "fn": "verifiers.v1.utils.prompt_utils:task_text",
                "keys": ["instruction", "question"],
            },
            RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH: self.append_to_system_prompt,
            DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH: {
                "fn": "harnesses.utils.rlm_utils:rlm_tool_skills_archive"
            },
        }
        dirs: dict[str, vf.ProgramValue] = {
            DEFAULT_RLM_CHECKOUT_PATH: {
                "fn": "harnesses.utils.rlm_utils:rlm_checkout_path",
                **(
                    {"local_checkout": self.local_checkout}
                    if self.local_checkout
                    else {}
                ),
                "rlm_repo_url": self.rlm_repo_url,
                "rlm_repo_ref": self.rlm_repo_ref,
                **({"gh_token_var": self.gh_token_var} if self.gh_token_var else {}),
            }
        }
        if self.skills is not None:
            dirs[DEFAULT_RLM_SKILLS_PATH] = self.skills
        else:
            dirs[DEFAULT_RLM_SKILLS_PATH] = {
                "fn": "harnesses.utils.rlm_utils:rlm_skills_dir"
            }

        env: dict[str, vf.ProgramValue] = {
            "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "OPENAI_MODEL": "runtime.model",
            "RLM_MODEL": "runtime.model",
            "RLM_TOOLS": ",".join(self.rlm_tools),
            "RLM_MAX_TURNS": str(self.rlm_max_turns),
            "RLM_EXEC_TIMEOUT": str(self.rlm_exec_timeout),
            "RLM_MAX_DEPTH": str(self.rlm_max_depth),
            **self.env_vars,
        }
        if self.summarize_at_tokens is not None:
            assert self.summarize_at_tokens > 0
            env["RLM_SUMMARIZE_AT_TOKENS"] = str(self.summarize_at_tokens)

        artifacts = vf.ArtifactsConfig.model_validate(
            {
                "rlm_metrics": {
                    "path": f"{self.workdir}/.rlm/sessions/*/meta.json",
                    "format": "json",
                    "key": "metrics",
                    "optional": True,
                }
            }
        )
        command_timeout = max(self.rlm_exec_timeout + 120, 600)
        setup_timeout = command_timeout
        if self.sandbox is not None and "setup_timeout" in self.sandbox.data(
            fill_defaults=False
        ):
            setup_timeout = self.sandbox.setup_timeout

        if self.sandbox is None:
            sandbox = vf.SandboxConfig(
                image="python:3.11-slim",
                workdir=self.workdir,
                cpu_cores=1,
                memory_gb=2,
                disk_size_gb=5,
                network_access=True,
                timeout_minutes=60,
                command_timeout=command_timeout,
                setup_timeout=setup_timeout,
            )
        else:
            sandbox = vf.SandboxConfig.model_validate(
                {
                    "workdir": self.workdir,
                    "command_timeout": command_timeout,
                    **self.sandbox.data(),
                    "setup_timeout": setup_timeout,
                }
            )

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
cd "${{AGENT_WORKDIR:-{self.workdir}}}"
rlm "$(cat {shlex.quote(self.instruction_path)})"
"""
        return self.resolve_command(
            command=["bash", "-lc", run_script],
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


class RLMConfig(vf.HarnessConfig):
    program: RLMProgramConfig = RLMProgramConfig()


class RLMEndpoint(vf.Endpoint):
    def trajectory_visibility(self, headers: dict[str, str]) -> vf.TrajectoryVisibility:
        if str(headers.get("x-rlm-depth", "0")) != "0":
            return "hidden"
        return super().trajectory_visibility(headers)


class RLM(vf.Harness[RLMConfig]):
    config: RLMConfig

    def load_endpoint(self) -> vf.Endpoint:
        return RLMEndpoint(
            use_tunnel=self.program_sandbox_config(self.program_config) is not None
        )

    @vf.metric
    async def rlm_sub_llm_call_count(self, state: vf.State) -> float:
        metrics = state["artifacts"].get("rlm_metrics") or {}
        assert isinstance(metrics, dict)
        value = metrics.get("sub_llm_call_count", 0.0)
        return float(value or 0.0)

    @vf.metric
    async def rlm_sub_llm_total_turns(self, state: vf.State) -> float:
        metrics = state["artifacts"].get("rlm_metrics") or {}
        assert isinstance(metrics, dict)
        value = metrics.get("sub_llm_total_turns", 0.0)
        return float(value or 0.0)

    @vf.metric
    async def rlm_sub_llm_total_tool_calls(self, state: vf.State) -> float:
        metrics = state["artifacts"].get("rlm_metrics") or {}
        assert isinstance(metrics, dict)
        value = metrics.get("sub_llm_total_tool_calls", 0.0)
        return float(value or 0.0)


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)
