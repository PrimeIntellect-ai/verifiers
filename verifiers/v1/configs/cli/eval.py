"""The `EvalConfig`: the single config object the eval CLI parses."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field, PrivateAttr, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig, EvalClientConfig
from verifiers.v1.configs.cli.env import narrowed_env_annotation, resolve_env_field
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.serve import ServeConfig
from verifiers.v1.envs.single_agent import SingleAgentEnvConfig
from verifiers.v1.types import SamplingConfig


def default_run_name(env: EnvConfig, model: str) -> str:
    """The auto-generated run name: `<env>--<model>--<harness>--<short-id>`, a
    descriptive leaf for the run directory `output_dir / run.name`. The short-id
    suffix keeps repeated invocations from colliding."""
    taskset = env.taskset
    name = taskset.name if taskset.id else "no-taskset"
    if taskset.id and env.id:
        # Same compounding as `EnvConfig.env_id`: a `best-of-n+gsm8k` run must
        # not share a name with a plain `gsm8k` one.
        name = f"{env.id}+{name}"
    # Every seat's resolved harness, distinct, in role order.
    harness = "+".join(dict.fromkeys(h.name for h in env.agent_harnesses().values()))
    slug = (
        f"{name}--{model.replace('/', '--')}--{harness or 'default'}--{uuid4().hex[:8]}"
    )
    return slug.lower()


class RichConfig(BaseConfig):
    """The live dashboard."""

    show_logs: bool = False
    """Replace the dashboard's per-rollout rows with a live tail of the attempt's log
    file (`logs/latest/eval.log`) — the env's own log lines included."""


class RunConfig(BaseConfig):
    name: str | None = None
    """Run name. Auto-generated as `<env>--<model>--<harness>--<short-id>` when unset."""

    dir: str | None = None
    """Run directory name — the run writes to `output_dir / dir`. Defaults to `run.name`;
    set it only when the directory should differ from the display name."""

    _id: str = PrivateAttr(default_factory=lambda: str(uuid4()))
    """The platform's run id once `adopt_id` has been called; a local uuid until then."""

    @property
    def id(self) -> str:
        return self._id

    def adopt_id(self, run_id: str) -> None:
        """Take the platform's run id, before the first rollout, so every trace carries it."""
        self._id = run_id


class EvalConfig(BaseConfig):
    env: SerializeAsAny[EnvConfig] = SingleAgentEnvConfig()
    """The environment — which env, its seed taskset, each agent, its knobs. Narrowed to
    the selected env's config class by the env id, else the taskset id."""
    serve: ServeConfig | None = Field(default_factory=ServeConfig)
    """How the env is hosted: the env-server worker pool (elastic by default) and each
    worker's episode bound — the path prime-rl trains through. `--no-serve` runs the
    rollouts in-process instead."""
    run: RunConfig = Field(default_factory=RunConfig)
    """Run identity: `run.name` is the display name, `run.dir` names the directory
    under `output_dir`, and `run.id` is stamped on traces."""
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
    says otherwise. Under `[serve]` it also seeds each worker's bound, unless
    `--serve.max-concurrent` pins one."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of the default info."""
    dry_run: bool = Field(False, exclude=True)
    """Resolve + validate the config and dump it, then exit. Excluded from the saved
    config so re-running `@ configs/eval.json` (or resuming/replaying the dir) actually runs."""
    clean: bool = Field(False, exclude=True)
    """Delete the run directory (`output_dir / run.dir`) before running, overwriting a
    previous run's results. Excluded from the saved config."""
    rich: RichConfig | None = Field(default_factory=RichConfig)
    """The live dashboard (on by default; `--no-rich` streams logs to the console
    instead). A served run has no live per-turn view, so its rollout rows fill in as
    each episode completes; `--rich.show-logs` swaps the rows for the run's logs."""
    push: bool = True
    """Upload the finished run to the Prime Intellect platform (the private Evaluations
    tab) at the end of the eval. On by default; disable with `--no-push`. Needs
    `$PRIME_API_KEY` or `prime login`."""
    output_dir: Path = Field(
        Path("outputs"), validation_alias=AliasChoices("output_dir", "o")
    )
    """Directory that groups related runs. The run itself (`configs/eval.json` +
    `traces.jsonl`) writes to `output_dir / run.dir`."""
    resume: bool = Field(False, exclude=True)
    """Re-run the run's missing/errored rollouts in place instead of starting fresh. The
    run dir comes from the resolved config (`output_dir / run.dir`), so resume with the
    run's own config — e.g. `uv run eval @ <run-dir>/configs/eval.json --resume`. Excluded
    from the saved config."""

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
