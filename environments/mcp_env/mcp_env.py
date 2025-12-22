import os

from datasets import Dataset
from dotenv import load_dotenv

import verifiers as vf
from verifiers.envs.experimental.mcp_env import MCPEnv

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
