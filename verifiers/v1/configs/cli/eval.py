"""The `EvalConfig`: the single config object the eval CLI parses."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig, EvalClientConfig
from verifiers.v1.configs.cli.env import (
    narrowed_env_annotation,
    resolve_env_field,
    single_agent_env_config,
)
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.legacy import (
    LegacyEnvConfig,
    is_legacy,
    refuse_mixed_run,
    run_env_id,
)
from verifiers.v1.configs.serve import ServingConfig
from verifiers.v1.types import SamplingConfig


class EvalConfig(BaseConfig):
    env: SerializeAsAny[EnvConfig] = Field(default_factory=single_agent_env_config)
    """The environment — which env, its seed taskset, each agent, its knobs. Narrowed to
    the selected env's config class by the env id, else the taskset id."""
    serve: ServingConfig = ServingConfig()
    """How the env is hosted under `--server`: the worker pool, each worker's episode
    bound. Ignored by an in-process run."""
    legacy: LegacyEnvConfig = LegacyEnvConfig()
    """A classic (v0) environment to evaluate through the bridge instead of `[env]`."""
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
        ge=1,
        validation_alias=AliasChoices("batch_size", "num_examples", "num_tasks", "n"),
    )
    """How many tasks to evaluate (None = all)."""
    num_rollouts: int = Field(
        1,
        ge=1,
        validation_alias=AliasChoices(
            "group_size", "rollouts_per_example", "num_rollouts", "r"
        ),
    )
    """Independent episodes per task — the trainer's group size."""
    shuffle: bool = Field(False, validation_alias=AliasChoices("shuffle", "s"))
    """Shuffle tasks before taking the first `num_tasks`."""
    max_concurrent: int | None = Field(
        128, ge=1, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Episodes in flight at once, `None` for no limit. An episode plays its agents one
    at a time, so this is the live agent runs too — until `--env.max-concurrent-agents`
    says otherwise. Under `--server` it seeds each worker's bound, unless
    `--serve.max-concurrent` pins one."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of the default info."""
    dry_run: bool = Field(False, exclude=True)
    """Resolve + validate the config and dump it, then exit. Excluded from the saved
    config so re-running `@ config.toml` (or resuming/replaying the dir) actually runs."""
    rich: bool = True
    """Show a live dashboard instead of per-rollout logs (in-process only; an unset
    `rich` defaults off under `--server`)."""
    server: bool = False
    """Drive rollouts through the env-server worker pool (sized by `[serve]`) instead of
    in-process — the path prime-rl trains through. Incompatible with `--rich`."""
    push: bool = True
    """Upload the finished run to the Prime Intellect platform (the private Evaluations
    tab) at the end of the eval. On by default; disable with `--no-push`. Needs
    `$PRIME_API_KEY` or `prime login`."""
    output_dir: Path | None = Field(
        None, validation_alias=AliasChoices("output_dir", "o")
    )
    """Where to write the run (config.toml + traces.jsonl). None = a fresh per-run dir
    under `outputs/<env>--<model>--<harness>/<uuid>` (so runs never overwrite each other)."""
    resume: Path | None = Field(None, exclude=True)
    """Set by `--resume <dir>`: re-run missing or errored rollouts, or an incomplete
    group-scored task as a whole group, appending to that run's own results. The run's saved
    config is loaded verbatim, so `--resume` takes no other arguments. Excluded from the
    saved config."""

    @model_validator(mode="after")
    def reject_rich_with_server(self):
        """The dashboard reads live in-process run slots, so it can't ride the
        worker pool: an unset `rich` defaults off under `--server`; an explicit
        `--rich --server` is refused."""
        if self.server and self.rich:
            if "rich" not in self.model_fields_set:
                self.rich = False
                return self
            raise ValueError(
                "`--rich` (the live dashboard) runs in-process and can't be combined with "
                "`--server`; drop `--rich`."
            )
        return self

    @property
    def is_legacy(self) -> bool:
        return is_legacy(self.env, self.legacy)

    @property
    def env_id(self) -> str:
        return run_env_id(self.env, self.legacy)

    @property
    def worker_max_concurrent(self) -> int | None:
        """A served worker's episode bound: its own pin, else the run's `--max-concurrent`."""
        return (
            self.serve.max_concurrent
            if self.serve.max_concurrent is not None
            else self.max_concurrent
        )

    @model_validator(mode="after")
    def _refuse_mixed_run(self):
        refuse_mixed_run(self.env, self.legacy)
        return self

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, data):
        return resolve_env_field(data, narrowed_env_annotation(cls))
