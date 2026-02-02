from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from .generator import generate_tasks

from verifiers.envs.experimental.local_harbor_env import LocalHarborEnv

logger = logging.getLogger(__name__)


class EndlessTerminalsEnv(LocalHarborEnv):
    def __init__(
        self,
        dataset_path: str | Path = Path(__file__).parent / "tasks",
        tasks: list[str] | None = None,
        generate_on_init: bool = False,
        num_tasks: int = 10,
        generation_model: str = "gpt-4o-mini",
        openai_api_key: str | None = None,
        sandbox_timeout_minutes: int = 120,
        cleanup_sandbox: bool = True,
        generation_jobs: int = 4,
        agent_name: str = "terminus-2",
        model_name: str = "anthropic/claude-sonnet-4",
        environment_type: str = "prime",
        timeout_seconds: float = 3600.0,
        timeout_multiplier: float = 1.0,
        delete_sandbox: bool = True,
        disable_verifier: bool = False,
        trials_dir: Path | None = None,
        max_turns: int | None = None,
        interception_port: int = 8765,
        **kwargs,
    ):
        self.dataset_path = Path(dataset_path)
        self.generate_on_init = generate_on_init
        self.num_tasks = num_tasks
        self.generation_model = generation_model
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.sandbox_timeout_minutes = sandbox_timeout_minutes
        self.cleanup_sandbox = cleanup_sandbox
        self.generation_jobs = generation_jobs

        # Generate tasks if requested
        if generate_on_init:
            self._generate_tasks()

        # Verify dataset exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {self.dataset_path}. "
                "Either provide an existing path or set generate_on_init=True."
            )

        super().__init__(
            dataset_path=self.dataset_path,
            tasks=tasks,
            agent_name=agent_name,
            model_name=model_name,
            environment_type=environment_type,
            timeout_seconds=timeout_seconds,
            timeout_multiplier=timeout_multiplier,
            delete_sandbox=delete_sandbox,
            disable_verifier=disable_verifier,
            trials_dir=trials_dir,
            max_turns=max_turns,
            interception_port=interception_port,
            **kwargs,
        )

    def _generate_tasks(self) -> None:
        """Generate tasks using Prime Sandbox."""
        logger.info(
            f"Generating {self.num_tasks} tasks with model {self.generation_model}..."
        )

        # Run async generation
        generated = asyncio.run(
            generate_tasks(
                num_tasks=self.num_tasks,
                out_dir=self.dataset_path,
                model=self.generation_model,
                openai_api_key=self.openai_api_key,
                sandbox_timeout_minutes=self.sandbox_timeout_minutes,
                cleanup_sandbox=self.cleanup_sandbox,
                jobs=self.generation_jobs,
            )
        )

        logger.info(f"Generated {len(generated)} tasks at {self.dataset_path}")

    async def generate_additional_tasks(
        self,
        num_tasks: int,
        reload_dataset: bool = True,
    ) -> list[Path]:
        generated = await generate_tasks(
            num_tasks=num_tasks,
            out_dir=self.dataset_path,
            model=self.generation_model,
            openai_api_key=self.openai_api_key,
            sandbox_timeout_minutes=self.sandbox_timeout_minutes,
            cleanup_sandbox=self.cleanup_sandbox,
            jobs=self.generation_jobs,
        )

        if reload_dataset:
            # Reload the Harbor dataset
            self.dataset = self._load_harbor_dataset()
            logger.info(f"Reloaded dataset with {len(self.dataset)} total tasks")

        return generated


def load_environment(
    dataset_path: str | Path = Path(__file__).parent / "tasks",
    tasks: list[str] | None = None,
    generate_on_init: bool = False,
    num_tasks: int = 10,
    generation_model: str = "gpt-4o-mini",
    openai_api_key: str | None = None,
    sandbox_timeout_minutes: int = 120,
    cleanup_sandbox: bool = True,
    generation_jobs: int = 4,
    agent_name: str = "terminus-2",
    model_name: str = "anthropic/claude-sonnet-4",
    environment_type: str = "prime",
    timeout_seconds: float = 3600.0,
    timeout_multiplier: float = 1.0,
    delete_sandbox: bool = True,
    disable_verifier: bool = False,
    trials_dir: Path | None = None,
    max_turns: int | None = None,
    interception_port: int = 8765,
) -> EndlessTerminalsEnv:
    return EndlessTerminalsEnv(
        dataset_path=dataset_path,
        tasks=tasks,
        generate_on_init=generate_on_init,
        num_tasks=num_tasks,
        generation_model=generation_model,
        openai_api_key=openai_api_key,
        sandbox_timeout_minutes=sandbox_timeout_minutes,
        cleanup_sandbox=cleanup_sandbox,
        generation_jobs=generation_jobs,
        agent_name=agent_name,
        model_name=model_name,
        environment_type=environment_type,
        timeout_seconds=timeout_seconds,
        timeout_multiplier=timeout_multiplier,
        delete_sandbox=delete_sandbox,
        disable_verifier=disable_verifier,
        trials_dir=trials_dir,
        max_turns=max_turns,
        interception_port=interception_port,
    )
