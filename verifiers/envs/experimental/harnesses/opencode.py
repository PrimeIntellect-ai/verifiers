from __future__ import annotations

import json
import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import verifiers as vf
from verifiers.envs.experimental.harnesses.base import HarnessMonitorRubric
from verifiers.envs.experimental.harnesses.cli_agent import CliHarness
from verifiers.types import AssistantMessage, Messages, Response, State, ToolCall
from verifiers.utils.interception_utils import _truncate as truncate

logger = logging.getLogger(__name__)

# https://opencode.ai/docs/tools/#built-in
OPENCODE_TOOLS = [
    "bash",
    "edit",
    "write",
    "read",
    "grep",
    "glob",
    "skill",
    "todowrite",
    "webfetch",
    "websearch",
    "codesearch",
    "task",
    "question",
]

DEFAULT_SYSTEM_PROMPT = """\
You are OpenCode, the best coding agent on the planet.

You are an interactive CLI tool that helps users with tasks. Use the instructions below and the tools available to you to assist the user.

# Tone and style
- Only use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.
- Your output will be displayed on a command line interface. Your responses should be short and concise. You can use Github-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.
- Output text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate with the user during the session.
- NEVER create files unless they're absolutely necessary for achieving your goal. ALWAYS prefer editing an existing file to creating a new one. This includes markdown files.

# Professional objectivity
Prioritize technical accuracy and truthfulness over validating the user's beliefs. Focus on facts and problem-solving, providing direct, objective technical info without any unnecessary superlatives, praise, or emotional validation. It is best for the user if OpenCode honestly applies the same rigorous standards to all ideas and disagrees when necessary, even if it may not be what the user wants to hear. Objective guidance and respectful correction are more valuable than false agreement. Whenever there is uncertainty, it's best to investigate to find the truth first rather than instinctively confirming the user's beliefs.
"""

TASK_MANAGEMENT_SYSTEM_PROMPT = """\
# Task Management
You have access to the TodoWrite tools to help you manage and plan tasks. Use these tools frequently to ensure that you are tracking your tasks and giving the user visibility into your progress. These tools are also helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable. It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.
"""

DEFAULT_INSTALL_COMMAND = (
    "curl -fsSL https://opencode.ai/install | bash -s -- --version v1.2.15"
)

DEFAULT_RUN_COMMAND_TEMPLATE = """\
set -e

apt-get update && apt-get install -y curl

for install_attempt in 1 2 3; do
    if {install_command}; then
        break
    fi
    if [ "$install_attempt" -eq 3 ]; then
        echo "OpenCode installation failed after 3 attempts" >&2
        exit 1
    fi
    echo "OpenCode install attempt $install_attempt/3 failed, retrying in 5s..." >&2
    sleep 5
done
export PATH="$HOME/.opencode/bin:$PATH"

mkdir -p ~/.config/opencode

SCHEMA_DOLLAR='$'

cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG

cd {agent_workdir}
cat {prompt_path} | opencode run 2>&1 | tee {logs_path}
"""


class OpenCodeMonitorRubric(HarnessMonitorRubric):
    """Monitor rubric that tracks OpenCode tool usage and harness failures."""

    def __init__(self, tool_names: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tool_names = list(tool_names or OPENCODE_TOOLS)

        self.add_metric(self.total_tool_calls)
        self.add_metric(self.unique_tools_used)
        self.add_metric(self.has_tool_calls)
        for tool_name in self.tool_names:
            self.add_metric(self._make_tool_count_metric(tool_name))

    @staticmethod
    def _count_tool_calls(completion: Messages) -> Counter:
        counts: Counter = Counter()
        assert isinstance(completion, list)
        for message in completion:
            if not isinstance(message, AssistantMessage):
                continue
            tool_calls = message.tool_calls
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if isinstance(tool_call, ToolCall):
                    counts[tool_call.name] += 1
        return counts

    async def total_tool_calls(self, completion: Messages) -> float:
        return float(sum(self._count_tool_calls(completion).values()))

    async def unique_tools_used(self, completion: Messages) -> float:
        return float(len(self._count_tool_calls(completion)))

    async def has_tool_calls(self, completion: Messages) -> float:
        return float(bool(self._count_tool_calls(completion)))

    def _make_tool_count_metric(self, tool_name: str) -> Callable:
        async def tool_count(completion: Messages) -> float:
            counts = self._count_tool_calls(completion)
            return float(counts.get(tool_name, 0))

        tool_count.__name__ = f"{tool_name}_calls"
        return tool_count


class OpenCodeHarness(CliHarness):
    """Interceptor-based OpenCode harness running inside a sandbox."""

    DEFAULT_AGENT_WORKDIR = "/app"
    DEFAULT_ASSET_DIR = "/opencode"
    DEFAULT_DISABLED_TOOLS = ["question", "task"]
    DEFAULT_PROVIDER_TIMEOUT_MS = 1_800_000
    DEFAULT_DISABLE_COMPACTION = True
    DEFAULT_ENABLE_INTERLEAVED = True

    def __init__(
        self,
        asset_dir: str = DEFAULT_ASSET_DIR,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        disabled_tools: list[str] | None = None,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        install_command: str = DEFAULT_INSTALL_COMMAND,
        run_command_template: str = DEFAULT_RUN_COMMAND_TEMPLATE,
        disable_compaction: bool = DEFAULT_DISABLE_COMPACTION,
        enable_interleaved: bool = DEFAULT_ENABLE_INTERLEAVED,
        provider_timeout_ms: int = DEFAULT_PROVIDER_TIMEOUT_MS,
        interception_port: int | None = None,
        interception_url: str | None = None,
        poll_interval: float = 1.0,
        timeout_seconds: float = 3600.0,
    ):
        self.asset_dir = asset_dir
        self.agent_workdir = agent_workdir
        self.disabled_tools = list(
            self.DEFAULT_DISABLED_TOOLS if disabled_tools is None else disabled_tools
        )
        self.provider_timeout_ms = provider_timeout_ms

        resolved_system_prompt = system_prompt
        if (
            resolved_system_prompt is not None
            and "todowrite" not in self.disabled_tools
        ):
            resolved_system_prompt += "\n" + TASK_MANAGEMENT_SYSTEM_PROMPT

        run_command = self.build_run_command(
            run_command_template=run_command_template,
            agent_workdir=agent_workdir,
            disabled_tools=self.disabled_tools,
            system_prompt=resolved_system_prompt,
            install_command=install_command,
            disable_compaction=disable_compaction,
            enable_interleaved=enable_interleaved,
        )
        super().__init__(
            run_command=run_command,
            interception_port=interception_port,
            interception_url=interception_url,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )

        self.system_prompt = resolved_system_prompt
        self.tools = [
            tool for tool in OPENCODE_TOOLS if tool not in self.disabled_tools
        ]

    def build_monitor_rubric(self) -> vf.Rubric | None:
        return OpenCodeMonitorRubric(tool_names=self.tools)

    @property
    def remote_system_prompt_path(self) -> str:
        return f"{self.asset_dir}/system.txt"

    @property
    def remote_prompt_path(self) -> str:
        return f"{self.asset_dir}/prompt.txt"

    @property
    def remote_logs_path(self) -> str:
        return f"{self.asset_dir}/logs.txt"

    async def build_env_vars(self, env: Any, state: State) -> dict[str, str]:
        env_vars = await super().build_env_vars(env, state)
        if "websearch" not in self.disabled_tools:
            env_vars["OPENCODE_ENABLE_EXA"] = "1"
        return env_vars

    async def start(self, env: Any, state: State) -> None:
        await self.upload_prompt_assets(env, state)
        await super().start(env, state)

    async def upload_prompt_assets(self, env: Any, state: State) -> None:
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return

        await env.sandbox_client.execute_command(
            sandbox_id,
            f"mkdir -p {self.asset_dir} {self.agent_workdir}",
            working_dir=None,
        )

        prompt = self.build_prompt(state)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as file:
            file.write(prompt)
            local_prompt_path = file.name

        try:
            logger.debug(
                "Uploading OpenCode prompt '%s' from %s to %s",
                truncate(prompt, 50),
                local_prompt_path,
                self.remote_prompt_path,
            )
            await env.sandbox_client.upload_file(
                sandbox_id,
                self.remote_prompt_path,
                local_prompt_path,
            )
        finally:
            Path(local_prompt_path).unlink(missing_ok=True)

        if self.system_prompt is None:
            return

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as file:
            file.write(self.system_prompt)
            local_system_prompt_path = file.name

        try:
            logger.debug(
                "Uploading OpenCode system prompt '%s' from %s to %s",
                truncate(self.system_prompt, 20),
                local_system_prompt_path,
                self.remote_system_prompt_path,
            )
            await env.sandbox_client.upload_file(
                sandbox_id,
                self.remote_system_prompt_path,
                local_system_prompt_path,
            )
        finally:
            Path(local_system_prompt_path).unlink(missing_ok=True)

    def normalize_response(self, env: Any, response: Response) -> Response:
        message = response.message
        normalized_tool_calls = message.tool_calls or []
        if message.tool_calls:
            normalized_tool_calls = []
            for tool_call in message.tool_calls:
                if not isinstance(tool_call, ToolCall):
                    normalized_tool_calls.append(tool_call)
                    continue
                try:
                    compact_arguments = json.dumps(
                        json.loads(tool_call.arguments),
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                except (json.JSONDecodeError, TypeError):
                    compact_arguments = tool_call.arguments
                normalized_tool_calls.append(
                    tool_call.model_copy(
                        update={
                            "name": tool_call.name.lower(),
                            "arguments": compact_arguments,
                        }
                    )
                )

        content = message.content
        if content is None:
            content = ""
        elif isinstance(content, str):
            content = content.rstrip()

        normalized_message = message.model_copy(
            update={
                "content": content,
                "tool_calls": normalized_tool_calls,
                "reasoning_content": message.reasoning_content or None,
            }
        )
        return response.model_copy(update={"message": normalized_message})

    def build_prompt(self, state: State) -> str:
        prompt = state["prompt"][-1]["content"]
        if isinstance(prompt, str):
            return prompt
        return json.dumps(prompt, ensure_ascii=False)

    def build_opencode_config(
        self,
        disabled_tools: list[str] | None = None,
        system_prompt_path: str | None = None,
        disable_compaction: bool = True,
        enable_interleaved: bool = True,
    ) -> str:
        config: dict[str, Any] = {
            "${SCHEMA_DOLLAR}schema": "https://opencode.ai/config.json",
            "provider": {
                "${OPENAI_MODEL%%/*}": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "${OPENAI_MODEL%%/*}",
                    "options": {
                        "baseURL": "$OPENAI_BASE_URL",
                        "apiKey": "intercepted",
                        "timeout": self.provider_timeout_ms,
                    },
                    "models": {
                        "${OPENAI_MODEL##*/}": {
                            "name": "${OPENAI_MODEL##*/}",
                            "modalities": {
                                "input": ["text", "image"],
                                "output": ["text"],
                            },
                            "interleaved": {"field": "reasoning_content"}
                            if enable_interleaved
                            else False,
                        }
                    },
                }
            },
            "model": "$OPENAI_MODEL",
        }

        if disable_compaction:
            config["compaction"] = {"auto": False, "prune": False}

        if system_prompt_path or disabled_tools:
            build_config: dict[str, Any] = {}
            if system_prompt_path:
                build_config["prompt"] = "{file:" + system_prompt_path + "}"
            if disabled_tools:
                build_config["tools"] = {tool: False for tool in disabled_tools}
            config["agent"] = {"build": build_config}

        return json.dumps(config, indent=2)

    def build_run_command(
        self,
        run_command_template: str,
        agent_workdir: str,
        disabled_tools: list[str] | None = None,
        system_prompt: str | None = None,
        install_command: str = DEFAULT_INSTALL_COMMAND,
        disable_compaction: bool = True,
        enable_interleaved: bool = True,
    ) -> str:
        config_json = self.build_opencode_config(
            disabled_tools=disabled_tools,
            system_prompt_path=self.remote_system_prompt_path
            if system_prompt
            else None,
            disable_compaction=disable_compaction,
            enable_interleaved=enable_interleaved,
        )
        return run_command_template.format(
            config_json=config_json,
            agent_workdir=agent_workdir,
            prompt_path=self.remote_prompt_path,
            logs_path=self.remote_logs_path,
            install_command=install_command,
        )
