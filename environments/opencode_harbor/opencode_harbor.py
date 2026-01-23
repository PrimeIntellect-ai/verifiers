import logging
from pathlib import Path

from verifiers.envs.experimental.harbor_env import HarborEnv

logger = logging.getLogger("verifiers.envs.OpenCodeHarborEnv")


def _build_run_command(agent_workdir: str) -> str:
    return f"""
set -e

echo "Starting OpenCode agent..."
echo "Base URL: $OPENAI_BASE_URL"

apt-get update && apt-get install -y curl

# TODO: Add opencode to prebuilt images so we don't need to install at runtime
curl -fsSL https://opencode.ai/install | bash
export PATH="$HOME/.opencode/bin:$PATH"

# Create opencode config directory
mkdir -p ~/.config/opencode

# Create opencode.json config with intercepted provider
cat > ~/.config/opencode/opencode.json << EOFCONFIG
{{
  "\\$schema": "https://opencode.ai/config.json",
  "provider": {{
    "intercepted": {{
      "npm": "@ai-sdk/openai-compatible",
      "name": "Intercepted",
      "options": {{
        "baseURL": "$OPENAI_BASE_URL",
        "apiKey": "intercepted",
        "timeout": 600000
      }},
      "models": {{
        "model": {{
          "name": "Intercepted Model",
          "modalities": {{
            "input": ["text", "image"],
            "output": ["text"]
          }}
        }}
      }}
    }}
  }},
  "model": "intercepted/model"
}}
EOFCONFIG

mkdir -p /logs/agent

# Run OpenCode with task instruction
cd {agent_workdir}
opencode run "$(cat /task/instruction.md)" 2>&1 | tee /logs/agent/opencode.txt
"""


class OpenCodeHarborEnv(HarborEnv):
    def __init__(
        self,
        dataset_path: str | Path,
        tasks: list[str] | None = None,
        agent_workdir: str = "/app",
        docker_image: str = "python:3.11-slim",
        **kwargs,
    ):
        super().__init__(
            run_command=_build_run_command(agent_workdir),
            dataset_path=dataset_path,
            tasks=tasks,
            agent_workdir=agent_workdir,
            docker_image=docker_image,
            **kwargs,
        )


def load_environment(
    dataset_path: str | Path = Path(__file__).parent / "tasks",
    tasks: list[str] | None = None,
    agent_workdir: str = "/app",
    docker_image: str = "python:3.11-slim",
    timeout_seconds: float = 900.0,
    cpu_cores: int = 2,
    memory_gb: int = 4,
    disk_size_gb: int = 10,
    timeout_minutes: int = 120,
    max_turns: int = 4,
) -> OpenCodeHarborEnv:
    return OpenCodeHarborEnv(
        dataset_path=dataset_path,
        tasks=tasks,
        agent_workdir=agent_workdir,
        docker_image=docker_image,
        timeout_seconds=timeout_seconds,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        timeout_minutes=timeout_minutes,
        max_turns=max_turns,
    )
