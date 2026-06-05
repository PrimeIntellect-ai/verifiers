"""The eval runner: run an environment's tasks against a model and save results.

`EvalConfig` is the single config object (a `prime-pydantic-config` `BaseConfig`),
so the CLI gets `@ file.toml`, dotted `--a.b` flags, and defaults<toml<cli for
free. `run_eval` fans rollouts out with bounded concurrency, aggregates, and
writes the full transcripts to `results.jsonl` + a summary `metadata.json`.
"""

import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import Field, model_validator
from pydantic_config import BaseConfig

from verifiers.nano.clients import Client, OpenAIChatCompletionsClient
from verifiers.nano.environment import EnvConfig, Environment
from verifiers.nano.output import EvalMetadata, save_results
from verifiers.nano.transcript import Transcript
from verifiers.nano.types import SamplingConfig

PRIME_INFERENCE_HOST = "pinference.ai"
PRIME_TEAM_ID_HEADER = "X-Prime-Team-ID"


class ClientConfig(BaseConfig):
    """An OpenAI-compatible endpoint. The API key is read from an env var."""

    base_url: str = "https://api.pinference.ai/api/v1"
    api_key_var: str = "PRIME_API_KEY"
    headers: dict[str, str] = Field(default_factory=dict)
    """Extra HTTP headers sent on every request."""

    @model_validator(mode="after")
    def add_prime_team_id(self) -> "ClientConfig":
        # Prime inference bills the personal balance unless a team is named; on
        # that endpoint, route billing to PRIME_TEAM_ID when set (explicit wins).
        team_id = os.environ.get("PRIME_TEAM_ID")
        if PRIME_INFERENCE_HOST in self.base_url and team_id:
            self.headers.setdefault(PRIME_TEAM_ID_HEADER, team_id)
        return self


def resolve_client(config: ClientConfig) -> Client:
    return OpenAIChatCompletionsClient(
        AsyncOpenAI(
            base_url=config.base_url,
            api_key=os.environ.get(config.api_key_var, "EMPTY"),
            default_headers=config.headers or None,
        )
    )


class EvalConfig(BaseConfig):
    id: str = ""
    """Environment id (set from the CLI positional argument)."""
    model: str = "deepseek/deepseek-v4-flash"
    client: ClientConfig = ClientConfig()
    sampling: SamplingConfig = SamplingConfig()
    num_tasks: int = 1
    num_rollouts: int = 1
    max_concurrent: int | None = 8
    output_dir: Path | None = None
    env: EnvConfig = EnvConfig()


async def run_eval(
    env: Environment, config: EvalConfig
) -> tuple[list[Transcript], EvalMetadata]:
    client = resolve_client(config.client)
    tasks = env.tasks()[: config.num_tasks]
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def rollout(task) -> Transcript:
        async with semaphore:
            return await env.run_rollout(task, client, config.model, config.sampling)

    jobs = [rollout(task) for task in tasks for _ in range(config.num_rollouts)]
    start = time.time()
    transcripts = await asyncio.gather(*jobs)
    duration = time.time() - start
    await client.close()

    n = len(transcripts)

    def average(field: str) -> dict[str, float]:
        keys = sorted({key for t in transcripts for key in getattr(t, field)})
        return {
            key: sum(getattr(t, field).get(key, 0.0) for t in transcripts) / n
            for key in keys
        }

    metadata = EvalMetadata(
        env_id=config.id,
        model=config.model,
        base_url=config.client.base_url,
        num_tasks=len(tasks),
        num_rollouts=config.num_rollouts,
        sampling=config.sampling,
        date=datetime.now(timezone.utc).isoformat(),
        duration=duration,
        avg_reward=sum(t.reward for t in transcripts) / n if n else 0.0,
        avg_rewards=average("rewards") if n else {},
        avg_metrics=average("metrics") if n else {},
    )
    default_dir = Path("outputs") / config.id / config.model.replace("/", "_")
    save_results(transcripts, metadata, config.output_dir or default_dir)
    return transcripts, metadata
