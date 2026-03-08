import json
import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
from verifiers.types import AssistantMessage, Messages, ToolCall
from verifiers.utils.interception_utils import _truncate as truncate

logger = logging.getLogger(__name__)

# Default OpenCode tools to track individually
OPENCODE_TOOLS = [
    "bash",
    "glob",
    "grep",
    "read",
    "edit",
    "write",
    "todowrite",
    "todoread",
    "webfetch",
    "question",
    "task",
]


DEFAULT_OPENCODE_SYSTEM_PROMPT = """\
You are OpenCode, the best coding agent on the planet.

You are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.

IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.

If the user asks for help or wants to give feedback inform them of the following:
- ctrl+p to list available actions
- To give feedback, users should report the issue at
  https://github.com/anomalyco/opencode

# Tone and style
- Only use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.
- Your output will be displayed on a command line interface. Your responses should be short and concise. You can use Github-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.
- Output text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate with the user during the session.
- NEVER create files unless they're absolutely necessary for achieving your goal. ALWAYS prefer editing an existing file to creating a new one. This includes markdown files.

# Professional objectivity
Prioritize technical accuracy and truthfulness over validating the user's beliefs. Focus on facts and problem-solving, providing direct, objective technical info without any unnecessary superlatives, praise, or emotional validation. It is best for the user if OpenCode honestly applies the same rigorous standards to all ideas and disagrees when necessary, even if it may not be what the user wants to hear. Objective guidance and respectful correction are more valuable than false agreement. Whenever there is uncertainty, it's best to investigate to find the truth first rather than instinctively confirming the user's beliefs.

# Task Management
You have access to the TodoWrite tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.
"""

DEFAULT_OPENCODE_RUN_COMMAND_TEMPLATE = """\
set -e

apt-get update && apt-get install -y curl

curl -fsSL https://opencode.ai/install | bash
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
    DEFAULT_DISABLED_TOOLS = ["webfetch", "question"]
    DEFAULT_RUN_COMMAND_TEMPLATE = DEFAULT_OPENCODE_RUN_COMMAND_TEMPLATE
    DEFAULT_SYSTEM_PROMPT = DEFAULT_OPENCODE_SYSTEM_PROMPT

    def __init__(
        self,
        dataset: Dataset,
        asset_dir: str = DEFAULT_ASSET_DIR,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        disabled_tools: list[str] | None = DEFAULT_DISABLED_TOOLS,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        run_command_template: str = DEFAULT_RUN_COMMAND_TEMPLATE,
        **kwargs,
    ):
        self.asset_dir = asset_dir
        self.agent_workdir = agent_workdir
        self.disabled_tools = disabled_tools

        run_command = self.build_run_command(
            run_command_template,
            agent_workdir,
            disabled_tools=disabled_tools,
            system_prompt=system_prompt,
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

        # Upload prompt as file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(prompt)
            local_prompt_path = f.name

        try:
            logger.debug(
                f"Uploading prompt '{truncate(prompt, 50)}' from {local_prompt_path} to {self.remote_prompt_path}"
            )
            await self.sandbox_client.upload_file(
                sandbox_id, self.remote_prompt_path, local_prompt_path
            )
        finally:
            Path(local_prompt_path).unlink(missing_ok=True)

        # Upload system prompt as file, if provided
        if self.system_prompt:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            ) as f:
                f.write(self.system_prompt)
                local_system_prompt_path = f.name

            try:
                logger.debug(
                    f"Uploading system prompt '{truncate(self.system_prompt, 20)}' from {local_system_prompt_path} to {self.remote_system_prompt_path}"
                )
                await self.sandbox_client.upload_file(
                    sandbox_id, self.remote_system_prompt_path, local_system_prompt_path
                )
            finally:
                Path(local_system_prompt_path).unlink(missing_ok=True)

    def build_prompt(self, state: vf.State) -> str:
        """Build the prompt to be uploaded to OpenCode."""
        return state["prompt"][-1]["content"]

    def build_opencode_config(
        self,
        disabled_tools: list[str] | None = None,
        system_prompt_path: str | None = None,
    ) -> str:
        """Build OpenCode config."""
        config: dict = {
            "${SCHEMA_DOLLAR}schema": "https://opencode.ai/config.json",
            "provider": {
                "intercepted": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Intercepted",
                    "options": {
                        "baseURL": "$OPENAI_BASE_URL",
                        "apiKey": "intercepted",
                        "timeout": 600000,
                    },
                    "models": {
                        "model": {
                            "name": "Intercepted Model",
                            "modalities": {
                                "input": ["text", "image"],
                                "output": ["text"],
                            },
                        }
                    },
                }
            },
            "model": "intercepted/model",
        }

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
    ) -> str:
        """Build bash script to install and run OpenCode."""

        config_json = self.build_opencode_config(
            disabled_tools,
            self.remote_system_prompt_path if system_prompt else None,
        )

        return run_command_template.format(
            config_json=config_json,
            agent_workdir=agent_workdir,
            prompt_path=self.remote_prompt_path,
            logs_path=self.remote_logs_path,
        )
