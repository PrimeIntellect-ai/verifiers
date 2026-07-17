"""The `GEPAConfig`: the single config object the `gepa` CLI parses.

GEPA optimizes the taskset config-layer system prompt (`taskset.system_prompt` /
`system_prompt_file`) by alternating rollouts (`evaluate`) with a teacher LM reflecting
on the reflective dataset (`make_reflective_dataset`) — see
`verifiers.v1.gepa.adapter.GEPAAdapter`. This inherits `EnvConfig`'s fields (`taskset`,
`harness`, `max_turns`, token limits, timeouts) as top-level flags, the same way `EvalConfig`
does, and adds the optimization loop's own knobs (model, reflection model, train/val split,
budget). There is no worker pool here (`EnvServerConfig` is not a base) — GEPA always runs
in-process, since its adapter protocol is itself synchronous (see `GEPAAdapter`)."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field

from verifiers.v1.clients import EvalClientConfig
from verifiers.v1.env import EnvConfig
from verifiers.v1.types import SamplingConfig


class GEPAConfig(EnvConfig):
    """The GEPA run plus its environment. `model` runs the rollouts under optimization;
    `reflection_model` (defaults to `model`) proposes new system prompts from the reflective
    dataset. `model` defaults to the same id as `EvalConfig`."""

    uuid: str = Field(default_factory=lambda: str(uuid4()), exclude=True)
    """Auto-generated run id — the leaf of the output dir, so runs never overwrite.
    Excluded from the saved config so re-running `@ config.toml` lands in a fresh dir."""
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
    split, so GEPA carves one out of `Taskset.select` the way `run_eval` samples (fixed
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

    max_concurrent: int | None = Field(
        128, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Max rollouts in flight at once, across the whole run."""
    output_dir: Path | None = Field(
        None, validation_alias=AliasChoices("output_dir", "o")
    )
    """Where to write results (config.toml + the streamed traces.jsonl, alongside GEPA's own
    candidates.json / run_log.json / best_system_prompt.txt). None = a fresh per-run dir under
    `outputs/<taskset>--<model>--<harness>/<uuid>` (via `output_path`)."""
    save_results: bool = True
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    dry_run: bool = False
    """Resolve + validate the config and dump it, then exit."""
