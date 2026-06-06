import json

import verifiers.v1 as vf

from .command import CommandHarness, CommandHarnessConfig, shell_command

OPENCODE_DEFAULT_VERSION = "PrimeIntellect-ai/opencode@1.1.63-rl2"
OPENCODE_DEFAULT_WORKDIR = "/app"
OPENCODE_DEFAULT_LOG_PATH = "/logs/agent/opencode.txt"
OPENCODE_DEFAULT_SYSTEM_PROMPT = """\
You are OpenCode, an interactive CLI tool that helps users with tasks.

Your output is displayed in a command line interface. Be concise and direct.
Use tools to complete tasks. Do not use shell commands or code comments as a
way to communicate with the user.
"""
OPENCODE_DEFAULT_DISABLED_TOOLS = [
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
]


class OpenCodeConfig(CommandHarnessConfig):
    system_prompt: vf.SystemPrompt | None = OPENCODE_DEFAULT_SYSTEM_PROMPT
    version: str = OPENCODE_DEFAULT_VERSION
    cwd: str | None = OPENCODE_DEFAULT_WORKDIR
    log_path: str = OPENCODE_DEFAULT_LOG_PATH
    disabled_tools: list[str] = OPENCODE_DEFAULT_DISABLED_TOOLS
    allow_git: bool = False
    disable_compaction: bool = True
    provider_timeout_ms: int = 3_600_000
    max_turns: int = 4


class OpenCode(CommandHarness[OpenCodeConfig]):
    config: OpenCodeConfig

    def command(self, task: vf.Task, state: vf.State) -> list[str]:
        _ = state
        instruction = str(getattr(task, "instruction", ""))
        opencode_config: dict[str, object] = {
            "provider": {
                "openai": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "OpenAI",
                    "options": {
                        "baseURL": "$OPENAI_BASE_URL",
                        "apiKey": "${OPENAI_API_KEY:-intercepted}",
                        "timeout": self.config.provider_timeout_ms,
                    },
                    "models": {
                        "model": {
                            "name": "Model",
                            "modalities": {"input": ["text"], "output": ["text"]},
                        }
                    },
                }
            },
            "model": "openai/model",
            "small_model": "openai/model",
            "agent": {
                "build": {
                    "prompt": vf.messages_text(vf.get_messages(self.system_prompt)),
                    "tools": {name: False for name in self.config.disabled_tools},
                }
            },
        }
        if self.config.disable_compaction:
            opencode_config["compaction"] = {"auto": False, "prune": False}
        config_json = json.dumps(opencode_config)
        script = f"""
set -eo pipefail
mkdir -p "$HOME/.config/opencode" "$(dirname {self.config.log_path!r})"
printf '%s' {config_json!r} > "$HOME/.config/opencode/opencode.json"
printf '%s' {instruction!r} | opencode run 2>&1 | tee {self.config.log_path!r}
"""
        return shell_command(script)


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)
