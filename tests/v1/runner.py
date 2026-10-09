"""Run an environment in-process for the e2e tests, saving episodes for replay."""

import asyncio
from pathlib import Path

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.cli.output import output_path, save_config
from verifiers.v1.clients import ClientConfig, EvalClientConfig, ModelContext
from verifiers.v1.configs.cli.env import narrowed_env_annotation, resolve_env_field
from verifiers.v1.configs.cli.run import RunConfig, default_run_name
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.select import SelectCLIConfig
from verifiers.v1.envs.single_agent import SingleAgentEnvConfig
from verifiers.v1.episode import Episode, EvalRunInfo
from verifiers.v1.types import SamplingConfig
from verifiers.v1.utils.loaders import load_environment
from verifiers.v1.utils.trace_store import append_episode


class RunnerConfig(BaseConfig):
    """The environment, model, and task selection used by the test fixtures."""

    env: SerializeAsAny[EnvConfig] = SingleAgentEnvConfig()
    run: RunConfig = Field(default_factory=RunConfig)
    model: str
    client: ClientConfig = EvalClientConfig()
    sampling: SamplingConfig = SamplingConfig()
    select: SelectCLIConfig = SelectCLIConfig()
    num_rollouts: int = 1
    max_concurrent: int | None = 128
    output_dir: Path = Path("outputs")

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, data):
        return resolve_env_field(data, narrowed_env_annotation(cls))

    @model_validator(mode="after")
    def auto_setup_run_name(self):
        if self.run.name is None:
            self.run.name = default_run_name(self.env, self.model)
        if self.run.dir is None:
            self.run.dir = self.run.name
        return self


async def run_episodes(config: RunnerConfig) -> list[Episode]:
    """Execute the selected tasks and persist each completed episode."""
    env = load_environment(config.env)
    tasks = list(env.taskset.select(config.select))
    out = output_path(config)
    save_config(config, out, "run.json")
    write_lock = asyncio.Lock()

    async def on_complete(episode: Episode) -> None:
        episode.record_run(EvalRunInfo(id=config.run.id, name=config.run.name))
        await append_episode(out, episode, write_lock)

    ctx = ModelContext(
        client=config.client, model=config.model, sampling=config.sampling
    )
    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    async with env.serving():
        return list(
            await asyncio.gather(
                *(
                    env.run_slot(slot, ctx, semaphore, on_complete)
                    for task in tasks
                    for slot in env.slots(task, config.num_rollouts)
                )
            )
        )
