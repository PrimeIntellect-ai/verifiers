import asyncio
import atexit
import os
import threading
from typing import Callable, Dict, List

from datasets import Dataset
from dotenv import load_dotenv

import verifiers as vf
from verifiers.envs.tool_env import ToolEnv
from verifiers.types import Message

from .mcp_server_connection import MCPServerConnection
from .mcp_tool_wrapper import MCPToolWrapper
from .models import MCPServerConfig

load_dotenv()

EXA_FETCH_TOOLS = [
    {
        "name": "exa",
        "command": "npx",
        "args": [
            "-y",
            "exa-mcp-server",
        ],
        "env": {
            "EXA_API_KEY": os.getenv("EXA_API_KEY"),
        },
        "description": "Exa MCP server",
    },
    {
        "name": "fetch",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "description": "Fetch MCP server",
    },
]


class MCPEnv(ToolEnv):
    """Environment for MCP-based tools using the official MCP SDK."""

    def __init__(
        self,
        mcp_servers: List[MCPServerConfig] = [],
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {str(e)}",
        **kwargs,
    ):
        self.mcp_servers = []
        if mcp_servers:
            for server in mcp_servers:
                if isinstance(server, dict):
                    self.mcp_servers.append(MCPServerConfig(**server))
                else:
                    self.mcp_servers.append(server)

        self.server_connections: Dict[str, MCPServerConnection] = {}
        self.mcp_tools: Dict[str, MCPToolWrapper] = {}

        self.error_formatter = error_formatter
        self._setup_complete = False
        self._init_kwargs = kwargs
        self._max_turns = max_turns

        super().__init__(
            tools=[], max_turns=max_turns, error_formatter=error_formatter, **kwargs
        )
        # Start a persistent background event loop and connect synchronously
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._run_loop, args=(self._bg_loop,), daemon=True
        )
        self._bg_thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect_servers(), self._bg_loop)
        fut.result()
        self._setup_complete = True

        # cleanup on exit
        atexit.register(
            lambda: (
                asyncio.run_coroutine_threadsafe(self.cleanup(), self._bg_loop).result(
                    timeout=5
                ),
                self._shutdown_loop(),
            )
        )

    def _run_loop(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _connect_servers(self):
        wrapper_tools = []

        for server_config in self.mcp_servers:
            connection = MCPServerConnection(server_config, self.logger)
            tools = await connection.connect()

            self.server_connections[server_config.name] = connection

            for tool in tools.values():
                wrapper = MCPToolWrapper(server_config.name, tool, connection)
                wrapper_tools.append(wrapper)
                self.mcp_tools[wrapper.__name__] = wrapper
                self.logger.info(
                    f"Registered MCP tool: {wrapper.__name__} from server '{server_config.name}'"
                )

        self.tools = wrapper_tools
        self.oai_tools = [tool.to_oai_tool() for tool in wrapper_tools]
        self.tool_map = {tool.__name__: tool for tool in wrapper_tools}

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> Message:
        if tool_name in self.tool_map:
            tool_wrapper = self.tool_map[tool_name]
            try:
                result = await tool_wrapper(**tool_args)
                return {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call_id,
                }
            except Exception as e:
                return {
                    "role": "tool",
                    "content": self.error_formatter(e),
                    "tool_call_id": tool_call_id,
                }
        else:
            return {
                "role": "tool",
                "content": f"Error: Tool '{tool_name}' not found",
                "tool_call_id": tool_call_id,
            }

    async def cleanup(self):
        for connection in self.server_connections.values():
            await connection.disconnect()

        self.server_connections.clear()
        self.mcp_tools.clear()

    def _shutdown_loop(self):
        self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        self._bg_thread.join(timeout=5)


def load_environment(
    mcp_servers: list = EXA_FETCH_TOOLS, dataset=None, **kwargs
) -> vf.Environment:
    """Load an MCPEnv environment with fetch server for testing."""
    dataset = dataset or Dataset.from_dict(
        {
            "question": [
                "Find out what Prime Intellect's newest announcement was from their website, give me the headline in 2 words. Their url is primeintellect.ai",
            ],
            "answer": ["ENVIRONMENTS HUB"],
        }
    )

    rubric = vf.JudgeRubric(judge_model="gpt-4.1-mini")

    async def judge_reward(judge, prompt, completion, answer, state):
        judge_response = await judge(prompt, completion, answer, state)
        return 1.0 if "yes" in judge_response.lower() else 0.0

    rubric.add_reward_func(judge_reward, weight=1.0)
    vf_env = MCPEnv(
        mcp_servers=mcp_servers,
        dataset=dataset,
        rubric=rubric,
        **kwargs,
    )

    return vf_env
