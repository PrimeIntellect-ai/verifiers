from __future__ import annotations

from pydantic import Field

from ...config import HarnessConfig

OPENCODE_DEFAULT_RELEASE_REPO = "PrimeIntellect-ai/opencode"
OPENCODE_DEFAULT_RELEASE_VERSION = "1.1.63-rl2"
OPENCODE_DEFAULT_RELEASE_SHA256 = (
    "47f4102796da50769e27d2c9ea6a9cf7941f76898390cb497278cab39c4b6ed4"
)
OPENCODE_DEFAULT_AGENT_WORKDIR = "/app"
OPENCODE_DEFAULT_INSTRUCTION_PATH = "/opencode/instruction.txt"
OPENCODE_DEFAULT_SYSTEM_PROMPT_PATH = "/opencode/system.txt"
OPENCODE_DEFAULT_LOG_PATH = "/logs/agent/opencode.txt"
OPENCODE_DEFAULT_SYSTEM_PROMPT = """\
You are OpenCode, an interactive CLI tool that helps users with tasks.

Your output is displayed in a command line interface. Be concise and direct.
Use tools to complete tasks. Do not use shell commands or code comments as a
way to communicate with the user.
"""
OPENCODE_DEFAULT_DISABLED_TOOLS = (
    "apply_patch",
    "write",
    "multiedit",
    "glob",
    "todowrite",
    "todoread",
    "websearch",
    "task",
    "batch",
    "list",
    "read",
    "question",
    "webfetch",
    "grep",
    "plan_exit",
    "plan_enter",
    "lsp",
    "codesearch",
    "skill",
)


class OpenCodeConfig(HarnessConfig):
    @classmethod
    def _merge_config_data(
        cls, config: object | None, data: dict[str, object]
    ) -> dict[str, object]:
        system_prompt_disabled = (
            data.get("system_prompt") is None and "system_prompt" in data
        ) or (isinstance(config, OpenCodeConfig) and config.system_prompt is None)
        merged = super()._merge_config_data(config, data)
        if system_prompt_disabled:
            merged["system_prompt"] = None
        return merged

    agent_workdir: str = OPENCODE_DEFAULT_AGENT_WORKDIR
    instruction_path: str = OPENCODE_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = OPENCODE_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = OPENCODE_DEFAULT_LOG_PATH
    system_prompt: object | None = OPENCODE_DEFAULT_SYSTEM_PROMPT
    disabled_tools: list[str] = Field(
        default_factory=lambda: list(OPENCODE_DEFAULT_DISABLED_TOOLS)
    )
    allow_git: bool = False
    disable_compaction: bool = True
    release_repo: str = OPENCODE_DEFAULT_RELEASE_REPO
    release_version: str = OPENCODE_DEFAULT_RELEASE_VERSION
    release_sha256: str = OPENCODE_DEFAULT_RELEASE_SHA256
    install_ripgrep: bool = True
    provider_timeout_ms: int = 3_600_000
    max_turns: int = 4
