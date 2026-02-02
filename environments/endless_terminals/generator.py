from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

logger = logging.getLogger(__name__)

# Default endless-terminals repo
ENDLESS_TERMINALS_REPO = "https://github.com/kcoopermiller/endless-terminals"

# Setup script for the sandbox
SETUP_SCRIPT = """
set -e

# Update and install dependencies
apt-get update
apt-get install -y software-properties-common git curl python3 python3-pip python3-venv

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install Apptainer
add-apt-repository -y ppa:apptainer/ppa
apt-get update
apt-get install -y apptainer-suid

# Remount /proc for Apptainer compatibility
mount -o remount,hidepid=0 /proc || true

# Clone endless-terminals
cd /workspace
git clone {repo_url} endless-terminals
cd endless-terminals

# Install dependencies with uv
uv sync

# Pull base Ubuntu container for Apptainer
apptainer pull ubuntu_22.04.sif docker://ubuntu:22.04

echo "Setup complete!"
"""

# Task generation script template
GENERATE_SCRIPT = """
set -e
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/endless-terminals

# Set OpenAI API key
export OPENAI_API_KEY="{api_key}"

# Generate tasks
uv run python generate_tasks.py \
    --num-tasks {num_tasks} \
    --out-dir /workspace/generated_tasks \
    --model {model} \
    --jobs {jobs} \
    --verbose

echo "Task generation complete!"
"""

# Harbor conversion script template
CONVERT_SCRIPT = """
set -e
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/endless-terminals

# Set OpenAI API key for Dockerfile conversion
export OPENAI_API_KEY="{api_key}"

# Convert tasks to Harbor format
uv run python -c "
import json
import os
import shutil
from pathlib import Path

TASK_TOML_TEMPLATE = '''version = \"1.0\"

[metadata]
author_name = \"Endless Terminals\"
author_email = \"\"
difficulty = \"medium\"
category = \"terminal\"
tags = [\"generated\", \"terminal\"]

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 600.0

[environment]
docker_image = \"endless-task-{task_name}\"
'''

TEST_SH_TEMPLATE = '''#!/bin/bash

apt-get update
apt-get install -y curl
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source \\$HOME/.local/bin/env

# Run pytest tests
cd /home/user
uvx \\
  --python 3.12 \\
  --with pytest==8.4.1 \\
  pytest /tests/test_state.py -v

# Check exit code and write reward
if [ \\$? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
'''

generated_dir = Path('/workspace/generated_tasks')
output_dir = Path('/workspace/harbor_tasks')
output_dir.mkdir(parents=True, exist_ok=True)

for task_dir in sorted(generated_dir.iterdir()):
    if not task_dir.is_dir() or not task_dir.name.startswith('task_'):
        continue
    
    task_json = task_dir / 'task.json'
    container_def = task_dir / 'container.def'
    test_final = task_dir / 'test_final_state.py'
    
    if not all(f.exists() for f in [task_json, container_def, test_final]):
        print(f'Skipping {task_dir.name}: missing required files')
        continue
    
    # Create Harbor task directory
    harbor_dir = output_dir / task_dir.name
    harbor_dir.mkdir(parents=True, exist_ok=True)
    
    # Read task.json
    with open(task_json) as f:
        task_data = json.load(f)
    
    # Create instruction.md from description
    instruction = task_data.get('description', 'Complete the terminal task.')
    (harbor_dir / 'instruction.md').write_text(instruction)
    
    # Create task.toml
    task_toml = TASK_TOML_TEMPLATE.format(task_name=task_dir.name.lower().replace('_', '-'))
    (harbor_dir / 'task.toml').write_text(task_toml)
    
    # Create environment directory with Dockerfile placeholder
    env_dir = harbor_dir / 'environment'
    env_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy container.def for later conversion
    shutil.copy(container_def, env_dir / 'container.def')
    
    # Create tests directory
    tests_dir = harbor_dir / 'tests'
    tests_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy test_final_state.py as test_state.py
    shutil.copy(test_final, tests_dir / 'test_state.py')
    
    # Create test.sh
    (tests_dir / 'test.sh').write_text(TEST_SH_TEMPLATE)
    os.chmod(tests_dir / 'test.sh', 0o755)
    
    print(f'Converted {task_dir.name} to Harbor format')

print('Harbor conversion complete!')
"

# Now convert container.def files to Dockerfiles using LLM
uv run python -m generator.convert_to_harbor.convert_sif_docker \
    --task-dir /workspace/harbor_tasks \
    --skip-build \
    --skip-tests \
    --model {model} \
    --provider openai

echo "Dockerfile conversion complete!"
"""


async def generate_tasks(
    num_tasks: int = 10,
    out_dir: Path | str = Path("tasks"),
    model: str = "gpt-4o-mini",
    openai_api_key: str | None = None,
    sandbox_timeout_minutes: int = 120,
    cleanup_sandbox: bool = True,
    jobs: int = 4,
    repo_url: str = ENDLESS_TERMINALS_REPO,
) -> list[Path]:
    """
    Generate endless-terminals tasks using Prime Sandbox.

    This function:
    1. Creates a Prime Sandbox with Ubuntu
    2. Installs Apptainer and clones endless-terminals
    3. Runs the task generation pipeline
    4. Converts tasks to Harbor format
    5. Downloads the tasks locally

    Args:
        num_tasks: Number of tasks to generate
        out_dir: Local directory to save generated tasks
        model: OpenAI model for task generation
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        sandbox_timeout_minutes: Sandbox timeout in minutes
        cleanup_sandbox: Whether to delete the sandbox after generation
        jobs: Number of parallel jobs for task generation
        repo_url: URL of the endless-terminals repository

    Returns:
        List of paths to generated task directories
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get API key
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable "
            "or pass openai_api_key parameter."
        )

    sandbox_id: Optional[str] = None
    generated_tasks: list[Path] = []

    async with AsyncSandboxClient() as sandboxes:
        try:
            # 1. Create sandbox
            logger.info("Creating Prime Sandbox...")
            request = CreateSandboxRequest(
                name=f"endless-terminals-gen-{os.getpid()}",
                docker_image="ubuntu:22.04",
                timeout_minutes=sandbox_timeout_minutes,
                cpu_cores=4,
                memory_gb=16,
            )
            sandbox = await sandboxes.create(request)
            sandbox_id = sandbox.id
            logger.info(f"Sandbox created: {sandbox_id}")

            await sandboxes.wait_for_creation(sandbox_id)
            logger.info("Sandbox is ready")

            # 2. Run setup script
            logger.info("Setting up sandbox (installing Apptainer, cloning repo)...")
            setup_script = SETUP_SCRIPT.format(repo_url=repo_url)

            # Use background job for long-running setup
            job = await sandboxes.start_background_job(sandbox_id, setup_script)
            logger.info(f"Setup job started: {job.job_id}")

            # Poll for completion
            while True:
                status = await sandboxes.get_background_job(sandbox_id, job)
                if status.completed:
                    if status.exit_code != 0:
                        logger.error(f"Setup failed: {status.stdout}\n{status.stderr}")
                        raise RuntimeError(
                            f"Sandbox setup failed with exit code {status.exit_code}"
                        )
                    logger.info("Setup complete")
                    break
                await asyncio.sleep(10)

            # 3. Generate tasks
            logger.info(f"Generating {num_tasks} tasks...")
            generate_script = GENERATE_SCRIPT.format(
                api_key=api_key,
                num_tasks=num_tasks,
                model=model,
                jobs=jobs,
            )

            job = await sandboxes.start_background_job(sandbox_id, generate_script)
            logger.info(f"Generation job started: {job.job_id}")

            while True:
                status = await sandboxes.get_background_job(sandbox_id, job)
                if status.completed:
                    if status.exit_code != 0:
                        logger.warning(f"Task generation had errors: {status.stdout}")
                    else:
                        logger.info("Task generation complete")
                    break
                await asyncio.sleep(30)

            # 4. Convert to Harbor format
            logger.info("Converting tasks to Harbor format...")
            convert_script = CONVERT_SCRIPT.format(
                api_key=api_key,
                model=model,
            )

            job = await sandboxes.start_background_job(sandbox_id, convert_script)
            logger.info(f"Conversion job started: {job.job_id}")

            while True:
                status = await sandboxes.get_background_job(sandbox_id, job)
                if status.completed:
                    if status.exit_code != 0:
                        logger.warning(f"Conversion had errors: {status.stdout}")
                    else:
                        logger.info("Conversion complete")
                    break
                await asyncio.sleep(15)

            # 5. List generated tasks
            logger.info("Listing generated tasks...")
            result = await sandboxes.execute_command(
                sandbox_id,
                "ls -1 /workspace/harbor_tasks",
                timeout=60,
            )

            if result.exit_code != 0:
                logger.error(f"Failed to list tasks: {result.stderr}")
                return []

            task_names = [
                name.strip()
                for name in result.stdout.strip().split("\n")
                if name.strip() and name.strip().startswith("task_")
            ]

            if not task_names:
                logger.warning("No tasks were generated")
                return []

            logger.info(f"Found {len(task_names)} tasks to download")

            # 6. Download tasks
            for task_name in task_names:
                logger.info(f"Downloading {task_name}...")
                local_task_dir = out_dir / task_name
                local_task_dir.mkdir(parents=True, exist_ok=True)

                # List files in the task directory
                result = await sandboxes.execute_command(
                    sandbox_id,
                    f"find /workspace/harbor_tasks/{task_name} -type f",
                    timeout=60,
                )

                if result.exit_code != 0:
                    logger.warning(f"Failed to list files for {task_name}")
                    continue

                files = [
                    f.strip() for f in result.stdout.strip().split("\n") if f.strip()
                ]

                for remote_file in files:
                    # Calculate relative path within task directory
                    rel_path = remote_file.replace(
                        f"/workspace/harbor_tasks/{task_name}/", ""
                    )
                    local_file = local_task_dir / rel_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        await sandboxes.download_file(
                            sandbox_id,
                            remote_file,
                            str(local_file),
                        )
                    except Exception as e:
                        logger.warning(f"Failed to download {remote_file}: {e}")

                generated_tasks.append(local_task_dir)
                logger.info(f"Downloaded {task_name}")

            logger.info(f"Successfully downloaded {len(generated_tasks)} tasks")

        finally:
            # Cleanup sandbox if requested
            if cleanup_sandbox and sandbox_id:
                logger.info("Cleaning up sandbox...")
                try:
                    await sandboxes.delete(sandbox_id)
                    logger.info("Sandbox deleted")
                except Exception as e:
                    logger.warning(f"Failed to delete sandbox: {e}")

    return generated_tasks


def generate_tasks_sync(
    num_tasks: int = 10,
    out_dir: Path | str = Path("tasks"),
    model: str = "gpt-4o-mini",
    openai_api_key: str | None = None,
    sandbox_timeout_minutes: int = 120,
    cleanup_sandbox: bool = True,
    jobs: int = 4,
    repo_url: str = ENDLESS_TERMINALS_REPO,
) -> list[Path]:
    """Synchronous version of generate_tasks for use in non-async contexts."""
    return asyncio.run(
        generate_tasks(
            num_tasks=num_tasks,
            out_dir=out_dir,
            model=model,
            openai_api_key=openai_api_key,
            sandbox_timeout_minutes=sandbox_timeout_minutes,
            cleanup_sandbox=cleanup_sandbox,
            jobs=jobs,
            repo_url=repo_url,
        )
    )


# CLI


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Endless Terminals tasks using Prime Sandbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=10,
        help="Number of tasks to generate",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tasks"),
        help="Output directory for generated tasks",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for task generation",
    )

    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )

    parser.add_argument(
        "--sandbox-timeout",
        type=int,
        default=120,
        help="Sandbox timeout in minutes",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel jobs for task generation",
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't delete sandbox after generation (useful for debugging)",
    )

    parser.add_argument(
        "--repo-url",
        type=str,
        default=ENDLESS_TERMINALS_REPO,
        help="URL of the endless-terminals repository",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    # Get API key
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required.", file=sys.stderr)
        print(
            "Set OPENAI_API_KEY environment variable or use --openai-api-key",
            file=sys.stderr,
        )
        return 1

    print(f"Generating {args.num_tasks} tasks...")
    print(f"Output directory: {args.out_dir}")
    print(f"Model: {args.model}")
    print(f"Sandbox timeout: {args.sandbox_timeout} minutes")
    print()

    try:
        generated = await generate_tasks(
            num_tasks=args.num_tasks,
            out_dir=args.out_dir,
            model=args.model,
            openai_api_key=api_key,
            sandbox_timeout_minutes=args.sandbox_timeout,
            cleanup_sandbox=not args.no_cleanup,
            jobs=args.jobs,
            repo_url=args.repo_url,
        )

        print()
        print(f"Successfully generated {len(generated)} tasks:")
        for task_path in generated:
            print(f"  - {task_path.name}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    args = _parse_args(argv)
    _setup_logging(args.verbose)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
