import asyncio
import atexit
import os
import threading
from typing import Callable, Dict, List

from datasets import Dataset
from dotenv import load_dotenv
from src.mcp_server_connection import MCPServerConnection
from src.mcp_tool_wrapper import MCPToolWrapper
from src.models import MCPServerConfig

import verifiers as vf
from verifiers.envs.tool_env import ToolEnv
from verifiers.types import Message

load_dotenv()

EXA_FETCH_TOOLS = [
    {
        "name": "exa",
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            "@smithery/cli@latest",
            "run",
            "exa",
            "--key",
            os.getenv("SMITHERY_KEY", ""),
            "--profile",
            os.getenv("SMITHERY_PROFILE", ""),
        ],
        "description": "Exa MCP server",
    },
    {
        "name": "fetch",
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "description": "Fetch MCP server",
    },
]

BROWSERBASE_TOOLS = [
    {
        "name": "browserbase",
        "transport": "stdio",
        "command": "npx",
        "args": ["@browserbasehq/mcp-server-browserbase"],
        "env": {
            "BROWSERBASE_API_KEY": os.getenv("BROWSERBASE_API_KEY", ""),
            "BROWSERBASE_PROJECT_ID": os.getenv("BROWSERBASE_PROJECT_ID", ""),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
        },
        "description": "Browserbase MCP (via npx)",
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

        self.logger.info(f"Initializing MCPEnv with {len(self.mcp_servers)} MCP server(s)")

        # Start a persistent background event loop and connect synchronously
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._run_loop, args=(self._bg_loop,), daemon=True
        )
        self._bg_thread.start()
        self.logger.debug("Background event loop started")

        fut = asyncio.run_coroutine_threadsafe(self._connect_servers(), self._bg_loop)
        fut.result()
        self._setup_complete = True
        self.logger.info("MCPEnv initialization complete")

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
        self.logger.info(f"Starting connection to {len(self.mcp_servers)} MCP server(s)")

        for server_config in self.mcp_servers:
            self.logger.info(f"Connecting to MCP server: '{server_config.name}'")
            self.logger.debug(f"  Transport: {server_config.transport}")
            self.logger.debug(f"  Command: {server_config.command}")
            self.logger.debug(f"  Args: {server_config.args}")
            if server_config.env:
                env_keys = list(server_config.env.keys())
                self.logger.debug(f"  Environment variables: {env_keys}")

            try:
                connection = MCPServerConnection(server_config, self.logger)
                tools = await connection.connect()

                self.server_connections[server_config.name] = connection
                self.logger.info(f"✓ Successfully connected to '{server_config.name}', discovered {len(tools)} tool(s)")

                for tool in tools.values():
                    wrapper = MCPToolWrapper(server_config.name, tool, connection)
                    wrapper_tools.append(wrapper)
                    self.mcp_tools[wrapper.__name__] = wrapper
                    self.logger.info(
                        f"  ├─ Registered MCP tool: {wrapper.__name__}"
                    )
            except Exception as e:
                self.logger.error(f"✗ Failed to connect to MCP server '{server_config.name}': {e}")
                raise

        self.tools = wrapper_tools
        self.oai_tools = [tool.to_oai_tool() for tool in wrapper_tools]
        self.tool_map = {tool.__name__: tool for tool in wrapper_tools}
        self.logger.info(f"✓ Total MCP tools registered: {len(self.tool_map)}")

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> Message:
        if tool_name in self.tool_map:
            tool_wrapper = self.tool_map[tool_name]
            self.logger.info(f"Calling tool: {tool_name}")
            self.logger.debug(f"  Arguments: {tool_args}")

            try:
                result = await tool_wrapper(**tool_args)
                result_str = str(result)
                result_preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
                self.logger.info(f"✓ Tool '{tool_name}' completed successfully")
                self.logger.debug(f"  Result preview: {result_preview}")

                return {
                    "role": "tool",
                    "content": result_str,
                    "tool_call_id": tool_call_id,
                }
            except Exception as e:
                self.logger.error(f"✗ Tool '{tool_name}' failed: {e}")
                return {
                    "role": "tool",
                    "content": self.error_formatter(e),
                    "tool_call_id": tool_call_id,
                }
        else:
            self.logger.error(f"✗ Tool '{tool_name}' not found in tool_map")
            return {
                "role": "tool",
                "content": f"Error: Tool '{tool_name}' not found",
                "tool_call_id": tool_call_id,
            }

    async def cleanup(self):
        self.logger.info(f"Cleaning up {len(self.server_connections)} MCP server connection(s)")

        for name, connection in self.server_connections.items():
            try:
                self.logger.debug(f"Disconnecting from MCP server: '{name}'")
                await connection.disconnect()
                self.logger.info(f"✓ Disconnected from MCP server: '{name}'")
            except Exception as e:
                self.logger.error(f"✗ Error disconnecting from MCP server '{name}': {e}")

        self.server_connections.clear()
        self.mcp_tools.clear()
        self.logger.info("Cleanup complete")

    def _shutdown_loop(self):
        self.logger.debug("Shutting down background event loop")
        self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        self._bg_thread.join(timeout=5)
        self.logger.debug("Background event loop stopped")


def load_environment(
    mcp_servers: list = EXA_FETCH_TOOLS + BROWSERBASE_TOOLS, dataset=None, **kwargs
) -> vf.Environment:
    """Load an MCPEnv environment with fetch server for testing."""
    dataset = dataset or Dataset.from_dict(
        {
            "question": [
                "Find out what Prime Intellect's newest announcement was from their website, give me the headline in 2 words. Their url is primeintellect.ai. Use the browserbase tools to get the information.",
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
