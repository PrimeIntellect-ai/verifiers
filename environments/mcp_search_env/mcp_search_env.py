from collections.abc import Iterable
from pathlib import Path
import sys
from typing import cast

from datasets import Dataset

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
    source: str = f"{__name__}:source"
    system_prompt: str = SYSTEM_PROMPT
    rewards: list[vf.CallableConfig] = [
        vf.CallableConfig(fn=f"{__name__}:exact_title_reward")
    ]
    mcp_servers: list[vf.ConfigData] = DEFAULT_MCP_SERVERS
    max_turns: int = 6
    examples: list[vf.ConfigData] = DEFAULT_EXAMPLES


class MCPSearchTaskset(vf.Taskset):
    config_type = MCPSearchTasksetConfig


def default_mcp_servers() -> list[vf.ConfigData]:
    return [dict(server) for server in DEFAULT_MCP_SERVERS]


def default_dataset() -> Dataset:
    return Dataset.from_list(DEFAULT_EXAMPLES)


def source(
    examples: Iterable[vf.ConfigMap] | None = None,
    *,
    max_turns: int = 6,
):
    rows = examples if examples is not None else default_dataset()
    for index, row in enumerate(rows):
        row = cast(vf.ConfigMap, row)
        question = str(row["question"])
        yield {
            **dict(row),
            "example_id": index,
            "max_turns": max_turns,
            "prompt": [{"role": "user", "content": question}],
        }


def mcp_tool_from_config(config: vf.ConfigMap) -> vf.MCPTool:
    return vf.MCPTool(
        command=str(config["command"]),
        args=[
            str(arg)
            for arg in cast(
                Iterable[str | int | float | bool], config.get("args") or []
            )
        ],
        env=cast(dict[str, str] | None, config.get("env")),
        cwd=cast(str | None, config.get("cwd")),
    )


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


def load_toolset(
    mcp_servers: Iterable[vf.ConfigMap] | None = None,
    config: vf.ToolsetConfig | None = None,
) -> vf.Toolset:
    servers = mcp_servers or default_mcp_servers()
    return vf.Toolset(
        tools=[mcp_tool_from_config(server) for server in servers],
        config=config,
    )


def load_taskset(
    config: MCPSearchTasksetConfig = MCPSearchTasksetConfig(),
) -> MCPSearchTaskset:
    taskset = MCPSearchTaskset(config=config)
    taskset.add_toolset(load_toolset(mcp_servers=config.mcp_servers))
    return taskset


def load_harness(config: vf.HarnessConfig = vf.HarnessConfig()):
    return vf.Harness(config=config)


class MCPSearchEnvConfig(vf.EnvConfig):
    taskset: MCPSearchTasksetConfig = MCPSearchTasksetConfig()
    harness: vf.HarnessConfig = vf.HarnessConfig()


def load_environment(config: MCPSearchEnvConfig = MCPSearchEnvConfig()) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
