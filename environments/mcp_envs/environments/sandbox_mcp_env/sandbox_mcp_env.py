"""
Sandbox MCP Environment Example

This example demonstrates how to create an MCP environment that runs inside
isolated sandboxes. Each rollout gets its own sandbox with an MCP server,
allowing agents to have completely stateful private workspaces.

Use cases:
- Jupyter notebook environments where agents can execute code
- File system environments where agents can read/write files safely
- Any MCP server that needs isolation between rollouts
"""
import verifiers as vf
from datasets import Dataset


def load_environment(**kwargs):
    """
    Creates a sandboxed MCP environment with a Jupyter notebook server.

    The sandbox transport:
    1. Creates a new sandbox for each rollout (connection_scope="rollout")
    2. Starts the MCP server command inside the sandbox
    3. Exposes the specified port so the MCP client can connect
    4. Cleans up the sandbox when the rollout completes
    """
    ds = Dataset.from_dict(
        {
            "question": [
                "Create a Python script that calculates the first 10 Fibonacci numbers and saves them to a file called 'fibonacci.txt'",
            ],
            "answer": ["0, 1, 1, 2, 3, 5, 8, 13, 21, 34"],
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
                "name": "jupyter-mcp",
                "command": "uvx",
                "args": ["jupyter-mcp-server@latest"],
                "env": {
                    "JUPYTER_URL": "http://localhost:8888",
                    "JUPYTER_TOKEN": "sandbox-token",
                }
            }
        ],
        transport_type="sandbox",
        connection_scope="rollout",  # Each rollout gets its own sandbox
        # Sandbox configuration
        sandbox_image="python:3.11-slim",
        sandbox_start_command="jupyter notebook --ip=0.0.0.0 --port=8888 --NotebookApp.token=sandbox-token --no-browser &",
        sandbox_cpu_cores=2,
        sandbox_memory_gb=4,
        sandbox_disk_size_gb=10,
        sandbox_timeout_minutes=30,
        sandbox_port_to_expose=8888,  # Expose the MCP server port
        sandbox_environment_vars={
            "PYTHONUNBUFFERED": "1",
        },
        # Standard env options
        dataset=ds,
        rubric=rubric,
        max_turns=15,
        **kwargs
    )

    return env


def load_filesystem_environment(**kwargs):
    """
    Alternative example: A sandboxed filesystem MCP server.

    This allows agents to safely read/write files in an isolated environment.
    """
    ds = Dataset.from_dict(
        {
            "question": [
                "Create a directory structure for a Python project with src/, tests/, and docs/ folders, and add a README.md file",
            ],
            "answer": ["Project structure created"],
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
                "name": "filesystem-mcp",
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/workspace"
                ],
            }
        ],
        transport_type="sandbox",
        connection_scope="rollout",
        sandbox_image="node:20-slim",
        sandbox_start_command="tail -f /dev/null",  # Keep sandbox alive
        sandbox_cpu_cores=1,
        sandbox_memory_gb=2,
        sandbox_disk_size_gb=5,
        sandbox_timeout_minutes=15,
        sandbox_port_to_expose=3000,  # Default MCP server port
        dataset=ds,
        rubric=rubric,
        max_turns=10,
        **kwargs
    )

    return env
