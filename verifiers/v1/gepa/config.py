"""The `GEPAConfig`: the single config object the `gepa` CLI parses.

GEPA optimizes one taskset's `Task.system_prompt` by alternating rollouts (`evaluate`) with a
teacher LM reflecting on the reflective dataset (`make_reflective_dataset`) — see
`verifiers.v1.gepa.adapter.GEPAv1Adapter`. This inherits `EnvConfig`'s fields (`taskset`,
`harness`, `max_turns`, token limits, timeouts) as top-level flags, the same way `EvalConfig`
does, and adds the optimization loop's own knobs (model, reflection model, train/val split,
budget). There is no worker pool here (`EnvServerConfig` is not a base) — GEPA always runs
in-process, since its adapter protocol is itself synchronous (see `GEPAv1Adapter`)."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field

from verifiers.v1.clients import ClientConfig, EvalClientConfig
from verifiers.v1.env import EnvConfig
from verifiers.v1.types import SamplingConfig


class GEPAConfig(EnvConfig):
    """The GEPA run plus its environment. `model` runs the rollouts under optimization;
    `reflection_model` (defaults to `model`) proposes new system prompts from the reflective
    dataset. No default for `model` — a GEPA run can spend a large eval budget, so pick it
    explicitly rather than silently optimizing (and spending) against a fallback."""

    uuid: str = Field(default_factory=lambda: str(uuid4()), exclude=True)
    """Auto-generated run id — the leaf of the output dir, so runs never overwrite.
    Excluded from the saved config so re-running `@ config.toml` lands in a fresh dir."""
    model: str = Field(..., validation_alias=AliasChoices("model", "m"))
    """Model id for rollouts under optimization."""
    client: ClientConfig = EvalClientConfig()
    sampling: SamplingConfig = SamplingConfig()
    reflection_model: str | None = None
    """Teacher model that proposes new system prompts. None = reuse `model`."""
    reflection_client: ClientConfig | None = None
    """Endpoint for `reflection_model`. None = reuse `client`."""

    num_train: int = Field(100, ge=1)
    """Tasks reserved for reflection minibatches (GEPA never scores the full trainset at once)."""
    num_val: int = Field(50, ge=1)
    """Tasks held out to score each candidate system prompt for the pareto frontier."""
    shuffle: bool = Field(True, validation_alias=AliasChoices("shuffle", "s"))
    """Shuffle tasks before splitting into train/val — v1 tasksets have no generic train/val
    split, so GEPA carves one out of `taskset.load()` the way `run_eval` samples (fixed
    seed, so the split is reproducible across runs)."""
    seed: int = 0
    """Seed for GEPA's optimizer (candidate selection / minibatch sampling). Task shuffling
    uses a fixed seed, matching eval — so this doesn't change the train/val split."""

    max_metric_calls: int = Field(
        500, validation_alias=AliasChoices("max_metric_calls", "B")
    )
    """Total rollouts GEPA may spend across the whole optimization run."""
    reflection_minibatch_size: int = 3
    """Train tasks sampled per reflection step."""
    perfect_score: float | None = None
    """Skip reflecting on a minibatch that already scores this well."""
    state_columns: list[str] = Field(default_factory=list)
    """Extra per-trace fields (from `trace.info`, else `task`) to surface to the teacher LM."""
    initial_prompt: str | None = None
    """Seed system prompt. None = the first loaded task's `Task.system_prompt`, if any task
    sets one (see `resolve_gepa_seed_prompt`)."""

    max_concurrent: int | None = Field(
        32, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Max rollouts in flight at once, across the whole run."""
    output_dir: Path | None = Field(
        None, validation_alias=AliasChoices("output_dir", "o")
    )
    """Where to write results (config.toml + the streamed traces.jsonl, alongside GEPA's own
    candidates.json / run_log.json). None = a fresh per-run dir under
    `outputs/<taskset>--<model>--<harness>/<uuid>` (via `output_path`)."""
    save_results: bool = True
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    dry_run: bool = False
    """Resolve + validate the config and dump it, then exit."""
