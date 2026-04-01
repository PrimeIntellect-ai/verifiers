import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Callable

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
from verifiers.types import AssistantMessage, Messages, ToolCall
from verifiers.utils.logging_utils import truncate
from verifiers.utils.path_utils import write_temp_file

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
- Output text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like bash or code comments as means to communicate with the user during the session.
- NEVER create files unless they're absolutely necessary for achieving your goal. ALWAYS prefer editing an existing file to creating a new one. This includes markdown files.

# Professional objectivity
Prioritize technical accuracy and truthfulness over validating the user's beliefs. Focus on facts and problem-solving, providing direct, objective technical info without any unnecessary superlatives, praise, or emotional validation. It is best for the user if OpenCode honestly applies the same rigorous standards to all ideas and disagrees when necessary, even if it may not be what the user wants to hear. Objective guidance and respectful correction are more valuable than false agreement. Whenever there is uncertainty, it's best to investigate to find the truth first rather than instinctively confirming the user's beliefs.
"""

DEFAULT_INSTALL_COMMAND = (
    "curl -fsSL https://opencode.ai/install | bash -s -- --version v1.2.15"
)

DEFAULT_RUN_COMMAND_TEMPLATE = """\
set -eo pipefail

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


class OpenCodeMonitorRubric(vf.Rubric):
    """Monitor rubric that tracks OpenCode tool usage."""

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
        """Count tool calls by name across all assistant messages."""
        counts: Counter = Counter()
        assert isinstance(completion, list)
        for msg in completion:
            if not isinstance(msg, AssistantMessage):
                continue
            tool_calls = msg.tool_calls
            if not isinstance(tool_calls, list):
                continue
            for tc in tool_calls:
                if isinstance(tc, ToolCall):
                    counts[tc.name] += 1
        return counts

    async def total_tool_calls(self, completion: Messages) -> float:
        """Total number of tool calls across all turns."""
        return float(sum(self._count_tool_calls(completion).values()))

    async def unique_tools_used(self, completion: Messages) -> float:
        """Number of distinct tools used."""
        return float(len(self._count_tool_calls(completion)))

    async def has_tool_calls(self, completion: Messages) -> float:
        """Whether the completion has any tool calls (0 or 1)."""
        return float(bool(self._count_tool_calls(completion)))

    def _make_tool_count_metric(self, tool_name: str) -> Callable:
        """Create a metric function that counts calls to a specific tool."""

        async def tool_count(completion: Messages) -> float:
            counts = self._count_tool_calls(completion)
            return float(counts.get(tool_name, 0))

        tool_count.__name__ = f"{tool_name}_calls"
        return tool_count


class OpenCodeEnv(CliAgentEnv):
    """OpenCode environment."""

    DEFAULT_AGENT_WORKDIR = "/app"
    DEFAULT_ASSET_DIR = "/opencode"
    # 'question' requires user interaction
    # 'task' spawns subagents which gives non-linear env histories
    DEFAULT_DISABLED_TOOLS = ["question", "task"]
    DEFAULT_INSTALL_COMMAND = DEFAULT_INSTALL_COMMAND
    DEFAULT_RUN_COMMAND_TEMPLATE = DEFAULT_RUN_COMMAND_TEMPLATE
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
    DEFAULT_PROVIDER_TIMEOUT_MS = 3_600_000  # 1h
    DEFAULT_DISABLE_COMPACTION = True
    DEFAULT_ENABLE_INTERLEAVED = True
    DEFAULT_INCLUDE_TASK_SYSTEM_PROMPT = False
    DEFAULT_TASK_SYSTEM_PROMPT = ""

    def __init__(
        self,
        dataset: Dataset,
        asset_dir: str = DEFAULT_ASSET_DIR,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        disabled_tools: list[str] = DEFAULT_DISABLED_TOOLS,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        install_command: str = DEFAULT_INSTALL_COMMAND,
        run_command_template: str = DEFAULT_RUN_COMMAND_TEMPLATE,
        disable_compaction: bool = DEFAULT_DISABLE_COMPACTION,
        enable_interleaved: bool = DEFAULT_ENABLE_INTERLEAVED,
        provider_timeout_ms: int = DEFAULT_PROVIDER_TIMEOUT_MS,
        task_system_prompt: str = DEFAULT_TASK_SYSTEM_PROMPT,
        include_task_system_prompt: bool = DEFAULT_INCLUDE_TASK_SYSTEM_PROMPT,
        **kwargs,
    ):
        self.asset_dir = asset_dir
        self.agent_workdir = agent_workdir
        self.disabled_tools = disabled_tools
        self.provider_timeout_ms = provider_timeout_ms

        if system_prompt is not None and include_task_system_prompt:
            system_prompt += "\n" + task_system_prompt

        run_command = self.build_run_command(
            run_command_template,
            agent_workdir,
            disabled_tools=disabled_tools,
            system_prompt=system_prompt,
            install_command=install_command,
            disable_compaction=disable_compaction,
            enable_interleaved=enable_interleaved,
        )

        super().__init__(
            run_command=run_command,
            dataset=dataset,
            system_prompt=system_prompt,
            **kwargs,
        )
        self.add_rubric(OpenCodeMonitorRubric())

    @property
    def remote_system_prompt_path(self) -> str:
        return f"{self.asset_dir}/system.txt"

    @property
    def remote_prompt_path(self) -> str:
        return f"{self.asset_dir}/prompt.txt"

    @property
    def remote_logs_path(self) -> str:
        return f"{self.asset_dir}/logs.txt"

    async def post_sandbox_setup(self, state: vf.State) -> None:
        """Upload prompt and optional system prompt after sandbox creation."""
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return

        # Create working directories
        dirs = [self.asset_dir, self.agent_workdir]
        self.logger.debug(f"Creating working directories ({', '.join(dirs)})")
        await self.sandbox_client.execute_command(
            sandbox_id, f"mkdir -p {' '.join(dirs)}", working_dir=None
        )

        prompt = self.build_prompt(state)

        # Upload prompt as file (temp file I/O offloaded to thread)
        local_prompt_path = await asyncio.to_thread(write_temp_file, prompt)

        try:
            logger.debug(
                f"Uploading prompt '{truncate(prompt, 50)}' from {local_prompt_path} to {self.remote_prompt_path}"
            )
            await self.sandbox_client.upload_file(
                sandbox_id, self.remote_prompt_path, local_prompt_path
            )
        finally:
            await asyncio.to_thread(Path(local_prompt_path).unlink, missing_ok=True)

        # Upload system prompt as file, if provided
        if self.system_prompt:
            local_system_prompt_path = await asyncio.to_thread(
                write_temp_file, self.system_prompt
            )

            try:
                logger.debug(
                    f"Uploading system prompt '{truncate(self.system_prompt, 20)}' from {local_system_prompt_path} to {self.remote_system_prompt_path}"
                )
                await self.sandbox_client.upload_file(
                    sandbox_id, self.remote_system_prompt_path, local_system_prompt_path
                )
            finally:
                await asyncio.to_thread(
                    Path(local_system_prompt_path).unlink, missing_ok=True
                )

    async def build_env_vars(self, state: vf.State) -> dict[str, str]:
        """Build environment variables for the sandbox. Override to add custom vars."""
        env_vars = await super().build_env_vars(state)
        if "websearch" not in self.disabled_tools:
            env_vars["OPENCODE_ENABLE_EXA"] = str(1)
        return env_vars

    async def normalize_response(self, response: vf.Response) -> vf.Response:
        """Normalize model response to match OpenCode's message history conventions:
        - Compact JSON arguments
        - Strip trailing newlines from assistant content

        Applying the same normalization to the stored step enables TITO prefix hits.
        """

        def _normalize() -> vf.Response:
            message = response.message
            normalized_tool_calls = message.tool_calls or []
            if message.tool_calls:
                normalized_tool_calls = []
                for tc in message.tool_calls:
                    if not isinstance(tc, ToolCall):
                        normalized_tool_calls.append(tc)
                        continue
                    try:
                        compact_arguments = json.dumps(
                            json.loads(tc.arguments),
                            separators=(",", ":"),
                            ensure_ascii=False,
                        )
                    except (json.JSONDecodeError, TypeError):
                        compact_arguments = tc.arguments
                    normalized_tool_calls.append(
                        tc.model_copy(
                            update={
                                "name": tc.name.lower(),
                                "arguments": compact_arguments,
                            }
                        )
                    )
            content = message.content
            if content is None:
                content = ""
            reasoning_content = message.reasoning_content or None
            normalized_message = message.model_copy(
                update={
                    "content": content,
                    "tool_calls": normalized_tool_calls,
                    "reasoning_content": reasoning_content,
                }
            )
            return response.model_copy(update={"message": normalized_message})

        return await asyncio.to_thread(_normalize)

    async def post_rollout(self, state: vf.State) -> None:
        """Collect agent logs from sandbox before teardown."""
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            try:
                result = await self.sandbox_client.execute_command(
                    sandbox_id,
                    f"cat {self.remote_logs_path} 2>/dev/null || echo '<no logs>'",
                    working_dir=None,
                )
                agent_logs = (result.stdout or "").strip()
                state["agent_logs"] = agent_logs

                # Log agent output on error or empty trajectory for debugging
                num_turns = len(state.get("trajectory", []))
                agent_error = state.get("agent_exit_code", 0) != 0
                if (agent_error or num_turns == 0) and agent_logs:
                    logger.warning(
                        f"Agent logs (example_id={state.get('example_id')}, "
                        f"exit_code={state.get('agent_exit_code')}, turns={num_turns}):\n{agent_logs}"
                    )
            except Exception as e:
                logger.warning(f"Failed to collect agent logs: {e}")

        await super().post_rollout(state)

    def build_prompt(self, state: vf.State) -> str:
        """Build the prompt to be uploaded to OpenCode."""
        return state["prompt"][-1]["content"]

    def build_opencode_config(
        self,
        disabled_tools: list[str] | None = None,
        system_prompt_path: str | None = None,
        disable_compaction: bool = True,
        enable_interleaved: bool = True,
    ) -> str:
        """Build OpenCode config."""
        config: dict = {
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
            build_config: dict = {}
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
        """Build bash script to install and run OpenCode."""

        config_json = self.build_opencode_config(
            disabled_tools,
            self.remote_system_prompt_path if system_prompt else None,
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
