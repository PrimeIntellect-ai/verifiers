from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Harbor task.toml template
TASK_TOML_TEMPLATE = """version = "1.0"

[metadata]
author_name = "Endless Terminals"
author_email = ""
difficulty = "{difficulty}"
category = "terminal"
tags = ["generated", "terminal", "endless-terminals"]

[verifier]
timeout_sec = {verifier_timeout}

[agent]
timeout_sec = {agent_timeout}

[environment]
docker_image = "{docker_image}"
"""

# Harbor test.sh template that writes reward to /logs/verifier/reward.txt
TEST_SH_TEMPLATE = """#!/bin/bash

apt-get update
apt-get install -y curl
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env

# Run pytest tests
cd /home/user
uvx \\
  --python 3.12 \\
  --with pytest==8.4.1 \\
  pytest /tests/test_state.py -v

# Check exit code and write reward
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

# Default Dockerfile template when no conversion is available
DEFAULT_DOCKERFILE_TEMPLATE = """FROM ubuntu:22.04

# Install basic dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-venv \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Install pytest
RUN pip3 install pytest

# Create user directory
RUN mkdir -p /home/user && chmod 755 /home/user

WORKDIR /home/user
"""


def convert_task_to_harbor(
    task_dir: Path,
    output_dir: Path,
    docker_image: Optional[str] = None,
    verifier_timeout: float = 300.0,
    agent_timeout: float = 600.0,
    difficulty: str = "medium",
) -> Optional[Path]:
    task_dir = Path(task_dir)
    output_dir = Path(output_dir)

    # Check required files exist
    task_json = task_dir / "task.json"
    container_def = task_dir / "container.def"
    test_final = task_dir / "test_final_state.py"

    if not task_json.exists():
        logger.warning(f"Missing task.json in {task_dir}")
        return None

    if not test_final.exists():
        logger.warning(f"Missing test_final_state.py in {task_dir}")
        return None

    # Read task.json
    try:
        with open(task_json) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse task.json: {e}")
        return None

    # Create output directory
    task_name = task_dir.name
    harbor_dir = output_dir / task_name
    harbor_dir.mkdir(parents=True, exist_ok=True)

    # Generate docker image name if not provided
    if docker_image is None:
        docker_image = f"endless-task-{task_name.lower().replace('_', '-')}"

    # 1. Create instruction.md from task description
    description = task_data.get("description", "Complete the terminal task.")
    instruction_file = harbor_dir / "instruction.md"
    instruction_file.write_text(description)
    logger.debug("Created instruction.md")

    # 2. Create task.toml
    task_toml = TASK_TOML_TEMPLATE.format(
        difficulty=difficulty,
        verifier_timeout=verifier_timeout,
        agent_timeout=agent_timeout,
        docker_image=docker_image,
    )
    task_toml_file = harbor_dir / "task.toml"
    task_toml_file.write_text(task_toml)
    logger.debug("Created task.toml")

    # 3. Create environment directory
    env_dir = harbor_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)

    # Copy container.def if it exists (for potential later conversion)
    if container_def.exists():
        shutil.copy(container_def, env_dir / "container.def")

        # Try to convert to Dockerfile (basic conversion)
        dockerfile = convert_def_to_dockerfile(container_def)
        if dockerfile:
            (env_dir / "Dockerfile").write_text(dockerfile)
            logger.debug("Created Dockerfile from container.def")
        else:
            # Use default Dockerfile
            (env_dir / "Dockerfile").write_text(DEFAULT_DOCKERFILE_TEMPLATE)
            logger.debug("Using default Dockerfile")
    else:
        # Use default Dockerfile if no container.def
        (env_dir / "Dockerfile").write_text(DEFAULT_DOCKERFILE_TEMPLATE)
        logger.debug("Using default Dockerfile (no container.def)")

    # 4. Create tests directory
    tests_dir = harbor_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Copy test_final_state.py as test_state.py
    shutil.copy(test_final, tests_dir / "test_state.py")
    logger.debug("Copied test_final_state.py to tests/test_state.py")

    # Create test.sh
    test_sh_file = tests_dir / "test.sh"
    test_sh_file.write_text(TEST_SH_TEMPLATE)
    os.chmod(test_sh_file, 0o755)
    logger.debug("Created test.sh")

    logger.info(f"Converted {task_name} to Harbor format at {harbor_dir}")
    return harbor_dir


def convert_def_to_dockerfile(def_path: Path) -> Optional[str]:
    try:
        content = def_path.read_text()
    except Exception as e:
        logger.warning(f"Failed to read {def_path}: {e}")
        return None

    lines = content.split("\n")
    dockerfile_lines = []
    in_post = False
    in_environment = False
    post_commands = []
    env_vars = []
    base_image = "ubuntu:22.04"  # Default

    for line in lines:
        stripped = line.strip()

        # Parse Bootstrap/From for base image
        if stripped.lower().startswith("from:"):
            base_image = stripped.split(":", 1)[1].strip()
        elif stripped.lower().startswith("bootstrap:"):
            # bootstrap: docker means we use the From line as-is
            pass

        # Track sections
        elif stripped.lower() == "%post":
            in_post = True
            in_environment = False
        elif stripped.lower() == "%environment":
            in_post = False
            in_environment = True
        elif stripped.startswith("%"):
            in_post = False
            in_environment = False

        # Collect commands
        elif in_post and stripped:
            post_commands.append(stripped)
        elif in_environment and stripped:
            # Parse environment variable exports
            if stripped.startswith("export "):
                var_part = stripped[7:].strip()
                if "=" in var_part:
                    env_vars.append(var_part)

    # Build Dockerfile
    dockerfile_lines.append(f"FROM {base_image}")
    dockerfile_lines.append("")

    # Add environment variables
    for env_var in env_vars:
        dockerfile_lines.append(f"ENV {env_var}")
    if env_vars:
        dockerfile_lines.append("")

    # Add post commands
    if post_commands:
        # Group commands that should run together
        dockerfile_lines.append("RUN " + " && \\\n    ".join(post_commands[:10]))

        # Add remaining commands in batches
        for i in range(10, len(post_commands), 10):
            batch = post_commands[i : i + 10]
            dockerfile_lines.append("RUN " + " && \\\n    ".join(batch))

    dockerfile_lines.append("")
    dockerfile_lines.append("WORKDIR /home/user")

    return "\n".join(dockerfile_lines)


def convert_batch(
    source_dir: Path,
    output_dir: Path,
    **kwargs,
) -> list[Path]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    converted = []

    for task_dir in sorted(source_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        if not task_dir.name.startswith("task_"):
            logger.debug(f"Skipping {task_dir.name}: not a task directory")
            continue

        result = convert_task_to_harbor(task_dir, output_dir, **kwargs)
        if result:
            converted.append(result)

    logger.info(f"Converted {len(converted)} tasks to Harbor format")
    return converted
