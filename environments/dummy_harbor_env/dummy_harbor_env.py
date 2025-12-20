import textwrap
from pathlib import Path

import verifiers as vf

_AGENT_SCRIPT = textwrap.dedent(
    """
    import os
    import subprocess
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key="dummy-key"
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    # Read task instruction
    with open("/task/instruction.md") as f:
        instruction = f.read().strip()

    messages = [
        {"role": "system", "content": "You are a coding agent. Respond with a bash command to complete the task. Output ONLY the command, nothing else."},
        {"role": "user", "content": instruction}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
    )
    command = response.choices[0].message.content.strip()

    # Execute the LLM's command
    os.makedirs("/app", exist_ok=True)
    subprocess.run(command, shell=True, cwd="/app")
    """
)


def _build_run_command() -> str:
    """Build the run command that executes the dummy agent script."""
    return f"""
set -e
pip install -q openai
cat > /tmp/agent_script.py << 'AGENT_EOF'
{_AGENT_SCRIPT}
AGENT_EOF
python -u /tmp/agent_script.py
"""


class DummyHarborEnv(vf.HarborEnv):
    def __init__(
        self,
        dataset_path: str | Path = Path(__file__).parent / "tasks",
        tasks: list[str] | None = None,
        agent_workdir: str = "/app",
        docker_image: str = "python:3.11-slim",
        **kwargs,
    ):
        super().__init__(
            run_command=_build_run_command(),
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
    timeout_seconds: float = 300.0,
    cpu_cores: int = 1,
    memory_gb: int = 2,
    disk_size_gb: int = 5,
    timeout_minutes: int = 30,
    max_turns: int = -1,
) -> DummyHarborEnv:
    return DummyHarborEnv(
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
