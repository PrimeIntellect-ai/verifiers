from collections.abc import Iterable
from pathlib import Path
import sys
from typing import cast

import verifiers as vf

SYSTEM_PROMPT = "Use the available MCP tools to answer the question."

DEFAULT_EXAMPLES = [
    {
        "query": "ceramic battery recycling",
        "question": "Use the MCP tools to find the record about ceramic battery recycling. What is the record title?",
        "answer": "Kiln Battery Loop",
    },
    {
        "query": "ocean drone algae bloom",
        "question": "Use the MCP tools to find the record about ocean drones and algae blooms. What is the record title?",
        "answer": "Tide Scout",
    },
    {
        "query": "library robot sorting",
        "question": "Use the MCP tools to find the record about library robot sorting. What is the record title?",
        "answer": "Stacks Navigator",
    },
    {
        "query": "green roof insulation",
        "question": "Use the MCP tools to find the record about green roof insulation. What is the record title?",
        "answer": "Moss Blanket",
    },
    {
        "query": "satellite wildfire mapping",
        "question": "Use the MCP tools to find the record about satellite wildfire mapping. What is the record title?",
        "answer": "Ember Atlas",
    },
    {
        "query": "fermentation sensor brewery",
        "question": "Use the MCP tools to find the record about fermentation sensors in a brewery. What is the record title?",
        "answer": "Yeast Whisper",
    },
    {
        "query": "rail tunnel airflow",
        "question": "Use the MCP tools to find the record about airflow in rail tunnels. What is the record title?",
        "answer": "Tunnel Pulse",
    },
    {
        "query": "museum climate microgrid",
        "question": "Use the MCP tools to find the record about museum climate control and microgrids. What is the record title?",
        "answer": "Gallery Grid",
    },
    {
        "query": "orchard frost prediction",
        "question": "Use the MCP tools to find the record about orchard frost prediction. What is the record title?",
        "answer": "Frost Lantern",
    },
    {
        "query": "city curb delivery",
        "question": "Use the MCP tools to find the record about city curb delivery routing. What is the record title?",
        "answer": "Curb Queue",
    },
]
MCP_SERVER_PATH = str(Path(__file__).with_name("mcp_server.py"))
DEFAULT_MCP_SERVERS: list[vf.ConfigData] = [
    {
        "name": "records",
        "command": sys.executable,
        "args": [MCP_SERVER_PATH],
        "description": "Synthetic search-record MCP server",
    },
]


class MCPSearchTasksetConfig(vf.TasksetConfig):
    rewards: list[str] = ["exact_title_reward"]
    mcp_servers: list[vf.ConfigData] | None = None
    max_turns: int = 6
    examples: list[vf.ConfigData] | None = None


class MCPSearchTaskset(vf.Taskset[MCPSearchTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(
            examples=self.config.examples, max_turns=self.config.max_turns
        )

    def load_system_prompt(self, config: MCPSearchTasksetConfig) -> vf.SystemPrompt:
        _ = config
        return SYSTEM_PROMPT

    def load_toolsets(self, config: MCPSearchTasksetConfig) -> vf.Toolsets:
        servers = config.mcp_servers or [dict(server) for server in DEFAULT_MCP_SERVERS]
        return {
            "records": vf.Toolset(
                tools=[
                    vf.MCPTool(
                        command=str(server["command"]),
                        args=[
                            str(arg)
                            for arg in cast(
                                Iterable[str | int | float | bool],
                                server.get("args") or [],
                            )
                        ],
                        env=cast(dict[str, str] | None, server.get("env")),
                        cwd=cast(str | None, server.get("cwd")),
                    )
                    for server in servers
                ]
            )
        }


def load_tasks(
    examples: Iterable[vf.JsonData] | None = None,
    *,
    max_turns: int = 6,
):
    records = examples if examples is not None else DEFAULT_EXAMPLES
    for index, record in enumerate(records):
        question = str(record["question"])
        yield {
            **dict(record),
            "example_id": index,
            "max_turns": max_turns,
            "prompt": [{"role": "user", "content": question}],
        }


@vf.reward(weight=1.0)
async def exact_title_reward(task: vf.Task, state: vf.State) -> float:
    completion = state.get("completion") or []
    messages = (
        vf.get_messages(completion, role="assistant")
        if isinstance(completion, list)
        else []
    )
    response = str(messages[-1].content or "") if messages else ""
    return float(str(task["answer"]).lower() in response.lower())


class MCPSearchEnvConfig(vf.EnvConfig):
    taskset: MCPSearchTasksetConfig = MCPSearchTasksetConfig()
    harness: vf.HarnessConfig = vf.HarnessConfig()


def load_environment(config: MCPSearchEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=MCPSearchTaskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )
