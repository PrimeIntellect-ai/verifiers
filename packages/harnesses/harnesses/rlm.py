import shlex

from pydantic import Field, model_validator

import verifiers as vf

from .utils.rlm_utils import (
    DEFAULT_RLM_CHECKOUT_PATH,
    DEFAULT_RLM_SKILLS_PATH,
    DEFAULT_RLM_TOOL_SKILL_MARKER,
    DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH,
    DEFAULT_RLM_TOOL_SKILLS_MANIFEST_NAME,
)


class RLMProgramConfig(vf.ProgramConfig):
    workdir: str = "/workspace"
    """In-sandbox working directory the `rlm` CLI cd's into; must be absolute."""

    instruction_path: str = "/rlm/instruction.txt"
    """In-sandbox path to the rendered task instruction file; must be absolute."""

    repo_url: str = "github.com/PrimeIntellect-ai/rlm-harness.git"
    """Git URL of the RLM harness checkout (ignored if `repo_path` is set)."""

    repo_ref: str = "main"
    """Git ref to check out from `repo_url`."""

    repo_path: str | None = None
    """Local path to an existing RLM checkout. If set, takes precedence over
    `repo_url` / `repo_ref` (no clone is performed)."""

    max_depth: int = Field(default=0, ge=0)
    """Max RLM recursion depth; 0 disables sub-LLM calls."""

    summarize_at_tokens: int | None = Field(default=None, gt=0)
    """Trigger RLM context summarization when the token count exceeds this."""

    append_to_system_prompt: str = ""
    """Extra text appended to the RLM system prompt."""

    tools: list[str] = ["ipython"]
    """RLM tool plugins to load (passed as `RLM_TOOLS`)."""

    allow_git: bool = False
    """Disable RLM's restricted git-history guard (sets `RLM_ALLOW_GIT=1`)."""

    env_vars: dict[str, str] = {}
    """Extra env vars exported into the sandbox."""

    skills: str | None = None
    """Override path to a skills directory; defaults to the taskset's `skills`
    upload dir."""

    @model_validator(mode="after")
    def _check_absolute_paths(self) -> "RLMProgramConfig":
        if not self.workdir.startswith("/"):
            raise ValueError(f"workdir must be absolute, got {self.workdir!r}")
        if not self.instruction_path.startswith("/"):
            raise ValueError(
                f"instruction_path must be absolute, got {self.instruction_path!r}"
            )
        return self

    def resolve(self) -> vf.ProgramConfig:
        append_to_system_prompt_path = "/rlm/append_to_system_prompt.txt"

        files: dict[str, vf.ProgramValue] = {
            self.instruction_path: {
                "fn": "verifiers.v1.utils.prompt_utils:task_text",
                "keys": ["instruction", "question"],
            },
            append_to_system_prompt_path: self.append_to_system_prompt,
            DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH: {
                "fn": "harnesses.utils.rlm_utils:rlm_tool_skills_archive"
            },
        }
        dirs: dict[str, vf.ProgramValue] = {
            DEFAULT_RLM_CHECKOUT_PATH: {
                "fn": "harnesses.utils.rlm_utils:rlm_checkout_path",
                **({"repo_path": self.repo_path} if self.repo_path else {}),
                "repo_url": self.repo_url,
                "repo_ref": self.repo_ref,
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
            "RLM_TOOLS": ",".join(self.tools),
            "RLM_MAX_DEPTH": str(self.max_depth),
            **self.env_vars,
        }
        if self.summarize_at_tokens is not None:
            env["RLM_SUMMARIZE_AT_TOKENS"] = str(self.summarize_at_tokens)
        if self.allow_git:
            env["RLM_ALLOW_GIT"] = "1"

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
        command_timeout = 600
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
export RLM_APPEND_TO_SYSTEM_PROMPT="$(cat {shlex.quote(append_to_system_prompt_path)} 2>/dev/null || true)"
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
