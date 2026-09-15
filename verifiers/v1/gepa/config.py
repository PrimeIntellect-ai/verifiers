"""The `GEPAConfig`: the single config object the `gepa` CLI parses.

GEPA optimizes one taskset's `Task.system_prompt` by alternating rollouts (`evaluate`) with a
teacher LM reflecting on the reflective dataset (`make_reflective_dataset`) — see
`verifiers.v1.gepa.adapter.GEPAAdapter`. It owns an `env` field (the
environment: its taskset, seats, limits) and adds the optimization loop's own knobs (model,
reflection model, train/val split, budget). There is no `[serve]` block here — GEPA
always runs in-process, since its adapter protocol is itself synchronous (see
`GEPAAdapter`)."""

from pathlib import Path

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import EvalClientConfig
from verifiers.v1.configs.cli.env import narrowed_env_annotation, resolve_env_field
from verifiers.v1.configs.cli.run import RunConfig, default_run_name
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.envs.single_agent import SingleAgentEnvConfig
from verifiers.v1.types import SamplingConfig


class GEPAConfig(BaseConfig):
    """The GEPA run plus its environment. `model` runs the rollouts under optimization;
    `reflection_model` (defaults to `model`) proposes new system prompts from the reflective
    dataset."""

    env: SerializeAsAny[EnvConfig] = SingleAgentEnvConfig()
    """The environment under optimization — the same `[env]` block as an eval's
    (`--env.taskset.*`, seats, limits), narrowed to the selected env's config class."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, data):
        return resolve_env_field(data, narrowed_env_annotation(cls))

    run: RunConfig = Field(default_factory=RunConfig)
    """Run identity: `run.name` is the display name and `run.dir` names the directory
    under `output_dir`; both auto-generate like an eval's when unset."""
    model: str = Field(
        "deepseek/deepseek-v4-flash", validation_alias=AliasChoices("model", "m")
    )
    """Model id for rollouts under optimization."""
    client: EvalClientConfig = EvalClientConfig()
    sampling: SamplingConfig = SamplingConfig()
    reflection_model: str | None = None
    """Teacher model that proposes new system prompts. None = reuse `model`."""
    reflection_client: EvalClientConfig | None = None
    """Endpoint for `reflection_model`. None = reuse `client`."""

    num_train: int = Field(100, ge=1)
    """Tasks reserved for reflection minibatches (GEPA never scores the full trainset at once)."""
    num_val: int = Field(50, ge=1)
    """Tasks held out to score each candidate system prompt for the pareto frontier."""
    shuffle: bool = Field(True, validation_alias=AliasChoices("shuffle", "s"))
    """Shuffle tasks before splitting into train/val — v1 tasksets have no generic train/val
    split, so GEPA carves one out of `Taskset.select` the way an eval samples (fixed
    seed, so the split is reproducible across runs)."""
    seed: int = 0
    """Seed for GEPA's optimizer (candidate selection / minibatch sampling). Task shuffling
    uses a fixed seed, matching eval — so this doesn't change the train/val split."""

    max_total_rollouts: int = Field(500)
    """Total rollouts GEPA may spend across the whole optimization run."""
    reflection_minibatch_size: int = 3
    """Train tasks sampled per reflection step."""
    reflection_columns: list[str] = Field(default_factory=list)
    """Extra per-trace fields (from `trace.info`, else `task`) to surface to the teacher LM."""
    initial_prompt: str | None = None
    """Seed system prompt. None = the first loaded task's `Task.system_prompt`, if any task
    sets one (see `resolve_gepa_seed_prompt`)."""

    max_concurrent: int | None = Field(
        128, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Max rollouts in flight at once, across the whole run."""
    output_dir: Path = Field(
        Path("outputs"), validation_alias=AliasChoices("output_dir", "o")
    )
    """Directory that groups related runs. The run (`configs/gepa.json` + the streamed
    `traces.jsonl`, alongside GEPA's own `candidates.json` / `run_log.json`) writes to
    `output_dir / run.dir`."""
    save_results: bool = True
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    dry_run: bool = Field(False, exclude=True)
    """Resolve + validate the config and dump it, then exit. Excluded from the
    saved config so re-running `@ configs/gepa.json` runs for real."""

    @model_validator(mode="after")
    def auto_setup_run_name(self):
        if self.run.name is None:
            self.run.name = default_run_name(self.env, self.model)
        if self.run.dir is None:
            self.run.dir = self.run.name
        return self
