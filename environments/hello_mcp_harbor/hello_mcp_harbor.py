from __future__ import annotations

import json
import logging
from pathlib import Path

from verifiers.envs.experimental.harbor_env import (
    HarborEnv,
    HarborMCPHealthcheck,
)

logger = logging.getLogger("verifiers.envs.HelloMCPHarborEnv")


def _build_run_command(agent_workdir: str) -> str:
    """Install OpenCode and point it at the task.toml-declared MCP server."""
    config: dict = {
        "${SCHEMA_DOLLAR}schema": "https://opencode.ai/config.json",
        "provider": {
            "intercepted": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Intercepted",
                "options": {
                    "baseURL": "$OPENAI_BASE_URL",
                    "apiKey": "intercepted",
                    "timeout": 600000,
                },
                "models": {
                    "model": {
                        "name": "Intercepted Model",
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                    }
                },
            }
        },
        "model": "intercepted/model",
        "mcp": {
            "mcp-server": {
                "type": "remote",
                "url": "http://mcp-server:8000/mcp",
            }
        },
    }
    config_json = json.dumps(config, indent=2)

    return f"""
set -e

apt-get update && apt-get install -y curl

curl -fsSL https://opencode.ai/install | bash
export PATH="$HOME/.opencode/bin:$PATH"

mkdir -p ~/.config/opencode

SCHEMA_DOLLAR='$'
cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG

mkdir -p /logs/agent
cd {agent_workdir}
opencode run "$(cat /task/instruction.md)" 2>&1 | tee /logs/agent/opencode.txt
"""


class HelloMCPHarborEnv(HarborEnv):
    """HarborEnv subclass that uploads the MCP server code before it's started."""

    async def pre_mcp_setup(self, state) -> None:
        """Install fastmcp + upload server.py before the MCP server starts."""
        sandbox_id = state["sandbox_id"]

        logger.info("Installing fastmcp + staging MCP server code…")
        await self.sandbox_client.execute_command(
            sandbox_id,
            "pip install --quiet --root-user-action=ignore fastmcp",
            working_dir=None,
            timeout=180,
        )
        await self.sandbox_client.execute_command(
            sandbox_id,
            "mkdir -p /opt/mcp-server",
            working_dir=None,
        )
        await self.sandbox_client.upload_file(
            sandbox_id,
            "/opt/mcp-server/server.py",
            str(Path(__file__).parent / "mcp_server" / "server.py"),
        )


def load_environment(
    dataset_path: Path | str = Path(__file__).parent / "tasks",
    tasks: list[str] | None = None,
    agent_workdir: str = "/app",
    docker_image: str = "python:3.11-slim",
    timeout_seconds: float = 900.0,
    cpu_cores: int = 2,
    memory_gb: int = 4,
    disk_size_gb: int = 10,
    max_turns: int = 8,
) -> HelloMCPHarborEnv:
    return HelloMCPHarborEnv(
        run_command=_build_run_command(agent_workdir),
        dataset_path=dataset_path,
        tasks=tasks,
        agent_workdir=agent_workdir,
        docker_image=docker_image,
        mcp_launch_commands={
            "mcp-server": "python /opt/mcp-server/server.py",
        },
        mcp_healthcheck=HarborMCPHealthcheck(
            retries=10,
            interval_sec=1.0,
            start_period_sec=3.0,
            timeout_sec=5.0,
        ),
        timeout_seconds=timeout_seconds,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        max_turns=max_turns,
    )
