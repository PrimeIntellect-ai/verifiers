import verifiers as vf
from verifiers.utils.mcp_utils.models import MCPServerConfig
from datasets import Dataset


def load_environment(**kwargs):
    ds = Dataset.from_dict(
        {
            "question": [
                "Check out what tools are available and try one that looks interesting to you",
            ],
            "answer": ["Hello World"],
        }
    )

    rubric = vf.JudgeRubric(judge_model="gpt-4.1-mini")

    async def judge_reward(judge, prompt, completion, answer, state):
        judge_response = await judge(prompt, completion, answer, state)
        return 1.0 if "yes" in judge_response.lower() else 0.0

    rubric.add_reward_func(judge_reward, weight=1.0)

    env = vf.MCPEnv(
        mcp_servers=[
            MCPServerConfig(
                name="everything-mcp",
                command="npx",
                args=[
                    "@modelcontextprotocol/server-everything",
                    "streamableHttp",
                ],
                env={
                    "PORT": "8000",
                },
                setup_commands=[
                    "apt update",
                    "apt upgrade -y",
                    "apt install -y git curl",
                    "curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -",
                    "apt-get install -y nodejs",
                    "npm install -g @modelcontextprotocol/server-everything@latest",
                ],
            )
        ],
        transport_type="sandbox",
        sandbox_image="python:3.11-slim",
        sandbox_start_command="tail -f /dev/null",
        sandbox_cpu_cores=1,
        sandbox_memory_gb=2,
        sandbox_disk_size_gb=5,
        sandbox_timeout_minutes=15,
        sandbox_port_to_expose=8000,  # Port the MCP server listens on
        # Standard env options
        dataset=ds,
        rubric=rubric,
        max_turns=10,
        **kwargs
    )

    return env
