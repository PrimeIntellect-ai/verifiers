from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any, cast

from datasets import Dataset
from openai import AsyncOpenAI

import verifiers.v1 as vf

SYSTEM_PROMPT = "Use the available MCP tools to answer the question."
JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the response is correct.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only.
"""


def default_mcp_servers() -> list[dict[str, Any]]:
    return [
        {
            "name": "exa",
            "command": "npx",
            "args": ["-y", "--loglevel=silent", "exa-mcp-server"],
            "env": {"EXA_API_KEY": os.getenv("EXA_API_KEY", "")},
            "description": "Exa MCP server",
        },
        {
            "name": "fetch",
            "command": "uvx",
            "args": ["mcp-server-fetch"],
            "description": "Fetch MCP server",
        },
    ]


def default_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "question": [
                "Use the fetch tool to inspect https://example.com. What is the page title? Answer with exactly two words.",
            ],
            "answer": ["Example Domain"],
        }
    )


def source(dataset=None, max_turns: int = 10):
    rows = dataset or default_dataset()
    for index, row in enumerate(rows):
        row = cast(Mapping[str, object], row)
        question = str(row["question"])
        yield {
            **dict(row),
            "example_id": index,
            "runtime": {"max_turns": max_turns},
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        }


def mcp_tool_from_config(config: Mapping[str, object]) -> vf.MCPTool:
    return vf.MCPTool(
        command=str(config["command"]),
        args=[str(arg) for arg in cast(Iterable[object], config.get("args") or [])],
        env=cast(dict[str, str] | None, config.get("env")),
        cwd=cast(str | None, config.get("cwd")),
    )


def judge_reward_factory(
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
):
    @vf.reward(weight=1.0)
    async def judge_reward(task, state) -> float:
        completion = state.get("completion") or []
        response = ""
        for message in reversed(completion):
            if message.get("role") == "assistant":
                response = str(message.get("content") or "")
                break
        if str(task["answer"]).lower() in response.lower():
            return 1.0
        prompt = JUDGE_PROMPT.format(
            question=task["question"],
            answer=task["answer"],
            response=response,
        )
        judge_client = AsyncOpenAI(
            base_url=judge_base_url,
            api_key=os.getenv(judge_api_key_var, ""),
        )
        try:
            result = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
            )
        finally:
            await judge_client.close()
        text = result.choices[0].message.content or ""
        return float("yes" in text.lower())

    return judge_reward


def load_taskset(
    dataset=None,
    mcp_servers: list[Mapping[str, object]] | None = None,
    max_turns: int = 10,
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    config=None,
):
    return vf.Taskset(
        source=lambda: source(dataset, max_turns=max_turns),
        rewards=[
            judge_reward_factory(
                judge_model=judge_model,
                judge_base_url=judge_base_url,
                judge_api_key_var=judge_api_key_var,
            )
        ],
        toolsets=[load_toolset(mcp_servers=mcp_servers)],
        config=config,
    )


def load_toolset(mcp_servers: list[Mapping[str, object]] | None = None, config=None):
    return vf.Toolset(
        tools=[
            mcp_tool_from_config(server)
            for server in mcp_servers or default_mcp_servers()
        ],
        config=config,
    )


def load_v1_environment(
    mcp_servers: list[Mapping[str, object]] | None = None,
    dataset=None,
    max_turns: int = 10,
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    **kwargs,
) -> vf.Env:
    if kwargs:
        raise TypeError(f"Unsupported v1 args: {sorted(kwargs)}")
    return vf.Env(
        taskset=load_taskset(
            dataset=dataset,
            mcp_servers=mcp_servers,
            max_turns=max_turns,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key_var=judge_api_key_var,
        )
    )
