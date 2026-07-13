"""The `EvalConfig`: the single config object the eval CLI parses."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field, model_validator

from verifiers.v1.clients import ClientConfig, EvalClientConfig
from verifiers.v1.env import EnvServerConfig
from verifiers.v1.types import SamplingConfig


class EvalConfig(EnvServerConfig):
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
        ge=1,
        validation_alias=AliasChoices(
            "group_size", "rollouts_per_example", "num_rollouts", "r"
        ),
    )
    """Independent invocations per seed task (one single-agent rollout, or one
    topology `go` whose traces are flattened on disk)."""
    shuffle: bool = Field(False, validation_alias=AliasChoices("shuffle", "s"))
    """Shuffle tasks before taking the first `num_tasks`."""
    max_concurrent: int | None = Field(
        128, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Max agent runs in flight in-process, or graph requests through the server."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of the default info."""
    dry_run: bool = False
    """Resolve + validate the config and dump it, then exit."""
    rich: bool = True
    """Show a live dashboard instead of per-rollout logs (in-process only)."""
    server: bool = False
    """Drive rollouts through the env-server worker pool (sized by `--pool.*`) instead of
    in-process — the path prime-rl trains through. Incompatible with `--rich` and with
    explicit `--topology.id` (topologies are local-eval only)."""
    push: bool = True
    """Upload the finished run to the Prime Intellect platform (the private Evaluations
    tab) at the end of the eval. On by default; disable with `--no-push`. Needs
    `$PRIME_API_KEY` or `prime login`. Explicit topologies force `--no-push`."""
    output_dir: Path | None = Field(
        None, validation_alias=AliasChoices("output_dir", "o")
    )
    """Where to write the run (config.toml + traces.jsonl). None = a fresh per-run dir
    under `outputs/<env>--<model>--<harness>/<uuid>` (so runs never overwrite each other)."""
    resume: Path | None = Field(None, exclude=True)
    """Set by `--resume <dir>`: re-run missing or errored rollouts, appending to that
    run's own results. The saved config is loaded verbatim, so `--resume` takes no other
    arguments. Excluded from the saved config. Not supported for explicit topologies."""

    @model_validator(mode="after")
    def reject_rich_with_server(self):
        if self.server and self.rich:
            raise ValueError(
                "`--rich` (the live dashboard) runs in-process and can't be combined with "
                "`--server`; pass `--no-rich` with `--server`."
            )
        return self

    @model_validator(mode="after")
    def topology_local_eval_only(self):
        """Explicit topologies are for local inspection: same runner, flat traces on
        disk, no platform/server/resume surface yet."""
        if self.topology is None:
            return self
        if self.server:
            raise ValueError(
                "explicit `--topology.id` is local-eval only; drop `--server` "
                "(taskset × harness is the supported server-backed path)"
            )
        if self.resume is not None:
            raise ValueError(
                "explicit `--topology.id` does not support `--resume`; "
                "resume the taskset × harness path instead"
            )
        if self.push:
            # Default push=True would otherwise reject every topology eval; force local.
            self.push = False
        return self
