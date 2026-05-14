from collections.abc import Iterable
from pathlib import Path
import sys
from typing import cast

from datasets import Dataset
from pydantic import Field

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


class MCPSearchTasksetConfig(vf.TasksetConfig):
    mcp_servers: list[vf.ConfigData] | None = None
    max_turns: int = 6
    examples: list[vf.ConfigData] = Field(
        default_factory=lambda: [dict(example) for example in DEFAULT_EXAMPLES]
    )


class MCPSearchTaskset(vf.Taskset):
    config_type = MCPSearchTasksetConfig


def default_mcp_servers() -> list[vf.ConfigData]:
    return [
        {
            "name": "records",
            "command": sys.executable,
            "args": [str(Path(__file__).with_name("mcp_server.py"))],
            "description": "Synthetic search-record MCP server",
        },
    ]


def default_dataset() -> Dataset:
    return Dataset.from_list(DEFAULT_EXAMPLES)


def source(
    dataset: Iterable[vf.ConfigMap] | None = None,
    *,
    max_turns: int = 6,
):
    rows = dataset if dataset is not None else default_dataset()
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


def response_text(state: vf.State) -> str:
    completion = state.get("completion") or []
    if not isinstance(completion, list):
        return ""
    messages = vf.get_messages(completion, role="assistant")
    return str(messages[-1].content or "") if messages else ""


@vf.reward(weight=1.0)
async def exact_title_reward(task: vf.Task, state: vf.State) -> float:
    return float(str(task["answer"]).lower() in response_text(state).lower())


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
    config: MCPSearchTasksetConfig | None = None,
    dataset: Iterable[vf.ConfigMap] | None = None,
    mcp_servers: Iterable[vf.ConfigMap] | None = None,
    max_turns: int | None = None,
) -> MCPSearchTaskset:
    taskset_overrides: vf.ConfigData = {}
    if mcp_servers is not None:
        taskset_overrides["mcp_servers"] = [dict(server) for server in mcp_servers]
    if max_turns is not None:
        taskset_overrides["max_turns"] = max_turns
    taskset_config = MCPSearchTasksetConfig(config, **taskset_overrides)
    return MCPSearchTaskset(
        source=lambda: source(
            dataset if dataset is not None else taskset_config.examples,
            max_turns=taskset_config.max_turns,
        ),
        system_prompt=SYSTEM_PROMPT,
        rewards=[exact_title_reward],
        toolsets=[load_toolset(mcp_servers=taskset_config.mcp_servers)],
        config=taskset_config,
    )


def load_harness(config: vf.HarnessConfig | None = None):
    return vf.Harness(config=config)


def load_environment(
    config: vf.EnvConfig,
    dataset: Iterable[vf.ConfigMap] | None = None,
    mcp_servers: Iterable[vf.ConfigMap] | None = None,
    max_turns: int | None = None,
) -> vf.Env:
    taskset_overrides: vf.ConfigData = {}
    if mcp_servers is not None:
        taskset_overrides["mcp_servers"] = [dict(server) for server in mcp_servers]
    if max_turns is not None:
        taskset_overrides["max_turns"] = max_turns
    config = vf.EnvConfig(
        config,
        taskset=MCPSearchTasksetConfig(**taskset_overrides),
    )
    taskset_config = (
        None if config.taskset is None else MCPSearchTasksetConfig(config.taskset)
    )
    harness_config = (
        None if config.harness is None else vf.HarnessConfig(config.harness)
    )
    return vf.Env(
        taskset=load_taskset(
            config=taskset_config,
            dataset=dataset,
        ),
        harness=load_harness(config=harness_config),
    )
