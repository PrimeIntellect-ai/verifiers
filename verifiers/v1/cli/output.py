"""On-disk output: results.jsonl (one full trace per line) + config.toml.

The trace is the full data dump — written verbatim, consumed by the platform
(visualization) and prime-rl (training). config.toml is the run's resolved EvalConfig,
written in the same format the CLI reads (`@ config.toml`), so a run is re-runnable from
its own output. Aggregates (avg reward, etc.) are cheap to recompute from results, so
they aren't stored.

The runner writes `config.toml` once up front (`save_config`) and then appends each
trace to `results.jsonl` as it completes (`append_trace`), so a long run's results are
durable as they land rather than only at the end.
"""

from pathlib import Path

import tomli_w

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import Trace


def output_path(config: EvalConfig) -> Path:
    """Where this run writes: `outputs/<taskset>--<model>--<harness>/<uuid>` (or the explicit
    `--output-dir`). The per-run `uuid` leaf means runs never overwrite each other."""
    if config.output_dir is not None:
        return config.output_dir
    name = f"{config.taskset.name}--{config.model.replace('/', '--')}--{config.harness.name}"
    return Path("outputs") / name / config.uuid


def save_config(config: EvalConfig, results_dir: Path) -> None:
    """Set up the run's output dir: write `config.toml` and start a fresh (empty)
    `results.jsonl`. Call once up front, before traces start landing."""
    results_dir.mkdir(parents=True, exist_ok=True)
    # mode="json" makes values TOML-friendly (Path -> str, etc.); exclude_none drops the
    # nulls TOML can't represent.
    toml = tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    (results_dir / "config.toml").write_text(toml)
    (results_dir / "results.jsonl").write_text(
        ""
    )  # fresh; appended to as traces complete


def append_trace(results_dir: Path, trace: Trace) -> None:
    """Append one finished trace to `results.jsonl` (one full trace per line). Called per
    trace as it completes — a synchronous, single-line append, so concurrent rollouts in
    one event loop never interleave."""
    with (results_dir / "results.jsonl").open("a") as f:
        f.write(trace.model_dump_json(exclude_none=True) + "\n")
