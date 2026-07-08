"""The `EvalConfig`: the single config object the eval CLI parses."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field, model_validator

from verifiers.v1.clients import ClientConfig, EvalClientConfig
from verifiers.v1.env import EnvServerConfig
from verifiers.v1.types import SamplingConfig


class EvalConfig(EnvServerConfig):
    """The eval run plus its environment: inherits the env's fields (`taskset`, `harness`,
    `max_turns`, token limits, timeouts) and the worker `pool` so they're top-level flags
    (`--taskset.id`, `--harness.id`, `--harness.runtime.*`, `--pool.*`, …) with no `--env.`
    prefix, and adds the run knobs (model, sampling, counts, …). Rollouts run in-process by
    default; `--server` drives them through the env-server worker pool (sized by `pool`) — the
    path prime-rl trains through. `--rich` adds a live dashboard (in-process only)."""

    uuid: str = Field(default_factory=lambda: str(uuid4()), exclude=True)
    """Auto-generated run id — the leaf of the output dir, so runs never overwrite.
    Excluded from the saved config so re-running `@ config.toml` lands in a fresh dir."""
    model: str = Field(
        "deepseek/deepseek-v4-flash", validation_alias=AliasChoices("model", "m")
    )
    """Model id."""
    client: ClientConfig = EvalClientConfig()
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
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of the default info."""
    dry_run: bool = False
    """Resolve + validate the config and dump it, then exit."""
    rich: bool = True
    """Show a live dashboard instead of per-rollout logs (in-process only)."""
    server: bool = False
    """Drive rollouts through the env-server worker pool (sized by `--pool.*`) instead of
    in-process — the path prime-rl trains through. Incompatible with `--rich`."""
    push: bool = True
    """Upload the finished run to the Prime Intellect platform (the private Evaluations
    tab) at the end of the eval. On by default; disable with `--no-push`. Needs
    `$PRIME_API_KEY` or `prime login`."""
    output_dir: Path | None = Field(
        None, validation_alias=AliasChoices("output_dir", "o")
    )
    """Where to write the run (config.toml + results.jsonl). None = a fresh per-run dir
    under `outputs/<env>--<model>--<harness>/<uuid>` (so runs never overwrite each other)."""
    resume: Path | None = Field(None, exclude=True)
    """Set by `--resume <dir>`: re-run only the rollouts a previous run left missing or
    errored, appending to that run's own results. The run's saved config is loaded verbatim,
    so `--resume` takes no other arguments. Excluded from the saved config."""

    @model_validator(mode="after")
    def reject_rich_with_server(self):
        if self.server and self.rich:
            raise ValueError(
                "`--rich` (the live dashboard) runs in-process and can't be combined with "
                "`--server`; pass `--no-rich` with `--server`."
            )
        return self
