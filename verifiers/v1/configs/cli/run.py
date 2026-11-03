"""The `RunnerConfig` the `vf-run` CLI parses, and the run identity every CLI that
writes a run directory shares."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field, PrivateAttr, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.cli.env import narrowed_env_annotation, resolve_env_field
from verifiers.v1.configs.client import ClientConfig, EvalClientConfig
from verifiers.v1.configs.env import EnvConfig, SingleAgentEnvConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.configs.select import SelectCLIConfig


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


class RunConfig(BaseConfig):
    name: str | None = None
    """Run name. Auto-generated as `<env>--<model>--<harness>--<short-id>` when unset."""

    dir: str | None = None
    """Run directory name — the run writes to `output_dir / dir`. Defaults to `run.name`;
    set it only when the directory should differ from the display name."""

    _id: str = PrivateAttr(default_factory=lambda: uuid4().hex)

    @property
    def id(self) -> str:
        """Stamped on every trace and exported as `$VF_RUN_ID`."""
        return self._id


class RunnerConfig(BaseConfig):
    env: SerializeAsAny[EnvConfig] = SingleAgentEnvConfig()
    """The environment — which env, its seed taskset, each agent, its knobs. Narrowed to
    the selected env's config class by the env id, else the taskset id."""
    run: RunConfig = Field(default_factory=RunConfig)
    model: str = Field(
        "deepseek/deepseek-v4-flash", validation_alias=AliasChoices("model", "m")
    )
    client: ClientConfig = EvalClientConfig()
    sampling: SamplingConfig = SamplingConfig()
    select: SelectCLIConfig = SelectCLIConfig()
    """Which tasks to run (`-n` sets `select.limit`, `-s` sets `select.shuffle`)."""
    num_rollouts: int = Field(
        1, ge=1, validation_alias=AliasChoices("num_rollouts", "r")
    )
    """Episodes per task."""
    max_concurrent: int | None = Field(
        128, ge=1, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Episodes in flight at once, `None` for no limit."""
    output_dir: Path = Field(
        Path("outputs"), validation_alias=AliasChoices("output_dir", "o")
    )
    """The run writes `configs/` and `traces.jsonl` to `output_dir / run.dir`."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    dry_run: bool = Field(False, exclude=True)
    """Resolve and validate the config, write it to the run directory, and exit."""

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
