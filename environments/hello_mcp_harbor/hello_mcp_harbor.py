from __future__ import annotations

import json
import logging
from pathlib import Path

from verifiers.envs.experimental.harbor_env import (
    HarborEnv,
    HarborMCPHealthcheck,
    HarborMCPLauncher,
)

MCP_SERVER_SOURCE = Path(__file__).parent / "mcp_server" / "server.py"

logger = logging.getLogger("verifiers.envs.HelloMCPHarborEnv")


def _build_run_command(agent_workdir: str) -> str:
    """OpenCode install + config that registers the framework-managed MCP server.

    ``$HARBOR_MCP_MCP_SERVER_URL`` is published by ``HarborEnv.build_env_vars``
    for every ``[[environment.mcp_servers]]`` entry that's active in the
    current phase. OpenCode's config supports ``$VAR`` substitution, so we
    reference it verbatim — no Python-side URL rewriting needed.
    """
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
                "url": "$HARBOR_MCP_MCP_SERVER_URL",
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


_MCP_LAUNCHERS: dict[str, HarborMCPLauncher] = {
    # task.toml declares the `mcp-server` entry (name/transport/url).
    # HarborEnv doesn't know how to start it in-container — that's this
    # launcher's job. Keeping launcher config on the Python side means
    # task.toml stays pure Harbor format.
    "mcp-server": HarborMCPLauncher(
        command="python /opt/mcp-server/server.py",
        phases=["agent"],
        healthcheck=HarborMCPHealthcheck(
            retries=10,
            interval_sec=1.0,
            start_period_sec=3.0,
            start_interval_sec=1.0,
            timeout_sec=5.0,
        ),
    ),
}


class HelloMCPHarborEnv(HarborEnv):
    """HarborEnv subclass that uploads the MCP server code before it's started.

    The ``mcp-server`` entry declared in ``task.toml`` is matched to the
    launcher in ``_MCP_LAUNCHERS`` (passed as ``mcp_launchers=`` to the base
    constructor). ``pre_mcp_setup`` puts ``server.py`` on the sandbox
    filesystem and installs its one Python dependency before the framework
    fires that launcher.
    """

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
            str(MCP_SERVER_SOURCE),
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
    timeout_minutes: int = 30,
    max_turns: int = 8,
) -> HelloMCPHarborEnv:
    return HelloMCPHarborEnv(
        run_command=_build_run_command(agent_workdir),
        dataset_path=dataset_path,
        tasks=tasks,
        agent_workdir=agent_workdir,
        docker_image=docker_image,
        mcp_launchers=_MCP_LAUNCHERS,
        timeout_seconds=timeout_seconds,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        timeout_minutes=timeout_minutes,
        max_turns=max_turns,
    )
