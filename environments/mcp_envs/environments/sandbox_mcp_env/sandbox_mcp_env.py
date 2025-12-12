"""
Sandbox MCP Environment Example

This example demonstrates how to create an MCP environment that runs inside
isolated sandboxes. Each rollout gets its own sandbox with an MCP server,
allowing agents to have completely stateful private workspaces.

Use cases:
- File system environments where agents can read/write files safely
- Code execution environments with isolation between rollouts
- Any MCP server that needs stateful isolation
"""
import verifiers as vf
from datasets import Dataset


def load_environment(**kwargs):
    """
    Creates a sandboxed MCP environment with a filesystem server.

    The sandbox transport:
    1. Creates a new sandbox for each rollout (connection_scope="rollout")
    2. Exposes the specified port for MCP communication
    3. Starts the MCP server command inside the sandbox
    4. Cleans up the sandbox when the rollout completes
    """
    ds = Dataset.from_dict(
        {
            "question": [
                "Use the echo tool to echo back the message 'Hello World'",
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
            {
                "name": "everything-mcp",
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-everything@latest",
                    "streamableHttp",
                ],
            }
        ],
        transport_type="sandbox",
        connection_scope="rollout",  # Each rollout gets its own sandbox
        # Sandbox configuration
        sandbox_image="python:3.11-slim",
        sandbox_start_command="apt-get update && apt-get install -y nodejs npm && tail -f /dev/null",
        sandbox_cpu_cores=1,
        sandbox_memory_gb=2,
        sandbox_disk_size_gb=5,
        sandbox_timeout_minutes=15,
        sandbox_port_to_expose=3000,  # Port the MCP server listens on
        # Standard env options
        dataset=ds,
        rubric=rubric,
        max_turns=10,
        **kwargs
    )

    return env
