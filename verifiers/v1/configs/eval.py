"""The `EvalConfig`: the single config object the eval CLI parses."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator

from verifiers.v1.clients import ClientConfig, EvalClientConfig
from verifiers.v1.env import EnvServerConfig
from verifiers.v1.topology import TopologyConfig
from verifiers.v1.types import SamplingConfig


class EvalConfig(EnvServerConfig):
    uuid: str = Field(default_factory=lambda: str(uuid4()), exclude=True)
    """Auto-generated run id — the leaf of the output dir, so runs never overwrite.
    Excluded from the saved config so re-running `@ config.toml` lands in a fresh dir."""
    topology: SerializeAsAny[TopologyConfig] | None = None
    """A multi-agent topology to run *instead of* the single `taskset` × `harness` pair
    (see `verifiers.v1.topology`), selected by `--topology.id`. Seeds come from its `taskset`
    slot (`--topology.taskset.id <id>`); each agent binds its own harness/routing
    (`--topology.<agent>.harness.id`, ...); `num_rollouts` then means topology instances per
    seed task, and `max_concurrent` still bounds rollouts in flight."""
    model: str = Field("deepseek/deepseek-v4-flash", validation_alias=AliasChoices("model", "m"))
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
        validation_alias=AliasChoices("group_size", "rollouts_per_example", "num_rollouts", "r"),
    )
    """Independent topology invocations per seed task."""
    shuffle: bool = Field(False, validation_alias=AliasChoices("shuffle", "s"))
    """Shuffle tasks before taking the first `num_tasks`."""
    max_concurrent: int | None = Field(128, validation_alias=AliasChoices("max_concurrent", "c"))
    """Max agent runs in flight in-process, or graph requests through the server."""
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
    output_dir: Path | None = Field(None, validation_alias=AliasChoices("output_dir", "o"))
    """Where to write the run (config.toml + traces.jsonl). None = a fresh per-run dir
    under `outputs/<env>--<model>--<harness>/<uuid>` (so runs never overwrite each other)."""
    resume: Path | None = Field(None, exclude=True)
    """Set by `--resume <dir>`: re-run missing or errored graph invocations, appending
    to that run's own results. The saved config is loaded verbatim, so `--resume` takes
    no other arguments. Excluded from the saved config."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_topology(cls, data):
        """Narrow `topology` to the config type its `id` resolves to (mirrors the
        taskset/harness narrowing in `EnvConfig`), so agent fields and interaction knobs
        validate typed from a toml alone. Runs before `EnvConfig._resolve_plugins`
        manufactures the default harness, so a *user-supplied* `harness` alongside a
        topology is still distinguishable — and refused, since it would be silently
        ignored (agents bind their own harnesses)."""
        if isinstance(data, dict) and data.get("topology"):
            if data.get("harness"):
                raise ValueError(
                    "`--harness.*` is ignored under a topology — each agent binds its "
                    "own harness (`--topology.<agent>.harness.*`); drop the flag."
                )
            from verifiers.v1.loaders import narrow_plugin_field, topology_config_type

            narrow_plugin_field(data, "topology", topology_config_type)
        return data

    @model_validator(mode="after")
    def reject_rich_with_server(self):
        if self.server and self.rich:
            raise ValueError(
                "`--rich` (the live dashboard) runs in-process and can't be combined with "
                "`--server`; pass `--no-rich` with `--server`."
            )
        return self

    @model_validator(mode="after")
    def check_topology(self):
        """A topology replaces the eval's own taskset × harness pair — reject combinations
        that would silently ignore half the config."""
        if self.topology is None:
            return self
        if self.taskset.id or self.id:
            raise ValueError(
                "`--topology.id` runs the topology's own agents; drop `--taskset.id` / "
                "`--id` (choose the seed tasks via --topology.taskset.id)."
            )
        return self
