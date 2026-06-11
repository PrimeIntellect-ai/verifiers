"""The `EvalConfig`: the single config object the eval CLI parses."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field

from verifiers.v1.clients import ClientConfig, OpenAIClientConfig
from verifiers.v1.env import EnvConfig
from verifiers.v1.types import SamplingConfig


class EvalConfig(EnvConfig):
    """The eval run plus its environment: inherits the env's fields (`taskset`, `harness`,
    `max_turns`, token limits, timeouts) so they're top-level flags (`--taskset.id`, `--harness.id`,
    `--harness.runtime.*`, …) with no `--env.` prefix, and adds the run knobs (model,
    sampling, counts, …)."""

    uuid: str = Field(default_factory=lambda: str(uuid4()), exclude=True)
    """Auto-generated run id — the leaf of the output dir, so runs never overwrite.
    Excluded from the saved config so re-running `@ config.toml` lands in a fresh dir."""
    model: str = Field(
        "deepseek/deepseek-v4-flash", validation_alias=AliasChoices("model", "m")
    )
    """Model id."""
    client: ClientConfig = OpenAIClientConfig()
    sampling: SamplingConfig = SamplingConfig()
    num_tasks: int | None = Field(
        None,
        validation_alias=AliasChoices("batch_size", "num_examples", "num_tasks", "n"),
    )
    """How many tasks to evaluate (None = all)."""
    num_rollouts: int = Field(
        1,
        validation_alias=AliasChoices(
            "group_size", "rollouts_per_example", "num_rollouts", "r"
        ),
    )
    """Rollouts per task. A taskset with `@group_reward`s requires >=2)."""
    shuffle: bool = Field(False, validation_alias=AliasChoices("shuffle", "s"))
    """Shuffle tasks before taking the first `num_tasks`."""
    max_concurrent: int | None = Field(
        128, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Max rollouts in flight at once."""
    num_workers: int = Field(0, validation_alias=AliasChoices("num_workers", "w"))
    """Run the eval through an env-server worker pool of up to this many processes (0 = off,
    in-process). >0 spawns the router + workers and drives rollouts over ZMQ — the same
    path prime-rl trains through, so it exercises the pool e2e for v1 and legacy v0 envs.
    The pool starts at one worker and scales up to this cap on demand."""
    multiplex: int = 128
    """Per-worker capacity for the pool's scale-up trigger: it spawns the next worker once
    in-flight rollouts reach 90% of `workers * multiplex` (until `num_workers` is hit)."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of the default info."""
    dry_run: bool = False
    """Resolve + validate the config and dump it, then exit."""
    rich: bool = True
    """Show a live dashboard instead of per-rollout logs."""
    output_dir: Path | None = Field(
        None, validation_alias=AliasChoices("output_dir", "o")
    )
    """Where to write the run (config.toml + results.jsonl). None = a fresh per-run dir
    under `outputs/<env>--<model>--<harness>/<uuid>` (so runs never overwrite each other)."""
