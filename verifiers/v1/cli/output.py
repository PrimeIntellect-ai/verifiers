"""On-disk output: config.toml + results.jsonl (one full trace per line).

The trace is the full data dump — written verbatim, consumed by the platform
(visualization) and prime-rl (training). config.toml is the run's resolved EvalConfig,
written in the same format the CLI reads (`@ config.toml`), so a run is re-runnable from
its own output. Aggregates (avg reward, etc.) are cheap to recompute from results, so
they aren't stored.

The runner writes `config.toml` once up front (`save_config`) and then appends each
trace to `results.jsonl` as it completes (`append_trace`), so a long run's results are
durable as they land rather than only at the end.
"""

import asyncio
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, TypeAdapter, ValidationError

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import Trace, WireTrace
from verifiers.v1.utils.aio import run_shielded


class _ResultRecord(dict[str, Any]):
    line: int


@dataclass(frozen=True)
class InvalidResultLine:
    line: int
    reason: str


@dataclass(frozen=True)
class EvalUploadData:
    """A normalized eval payload ready for host applications to upload."""

    eval_name: str
    model_name: str
    env: str
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    results: list[dict[str, Any]]
    invalid_results: list[InvalidResultLine]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("invalid_results")
        return data


def read_config(results_dir: Path) -> dict[str, Any]:
    """Read a native run's resolved config without importing its plugins."""
    path = results_dir / "config.toml"
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Invalid Verifiers eval config: {path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Invalid Verifiers eval config: {path}")
    return config


def read_results(path: Path) -> tuple[list[dict[str, Any]], list[InvalidResultLine]]:
    """Read JSONL results while reporting incomplete or invalid records to the caller."""
    results: list[dict[str, Any]] = []
    invalid: list[InvalidResultLine] = []
    # errors="replace": a torn multibyte write corrupts one line, not the whole read —
    # the mangled line then fails json.loads and is reported as invalid
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                invalid.append(InvalidResultLine(line_number, "invalid JSON"))
                continue
            if not isinstance(result, dict):
                invalid.append(
                    InvalidResultLine(
                        line_number, f"expected object, got {type(result).__name__}"
                    )
                )
                continue
            record = _ResultRecord(result)
            record.line = line_number
            results.append(record)
    return results, invalid


def convert_results_for_upload(
    samples: list[dict[str, Any]],
    invalid_results: list[InvalidResultLine] | None = None,
) -> list[dict[str, Any]]:
    """Convert v1 traces to the sample schema while preserving legacy results."""
    trace_fields = WireTrace.model_fields
    node_fields = MessageNode.model_fields
    rollout_counts: dict[int, int] = {}
    converted: list[dict[str, Any]] = []

    for sample_number, sample in enumerate(samples, start=1):
        # legacy rows never carry `nodes`/`rewards`; a row with either marker is a
        # v1 trace and must validate fully — partial rows are reported, not uploaded
        if "nodes" not in sample and "rewards" not in sample:
            legacy_sample = dict(sample)
            if "id" in legacy_sample and "example_id" not in legacy_sample:
                legacy_sample["example_id"] = legacy_sample["id"]
            converted.append(legacy_sample)
            continue

        trace_data = {
            key: value for key, value in sample.items() if key in trace_fields
        }
        raw_nodes = sample.get("nodes")
        if isinstance(raw_nodes, list):
            trace_data["nodes"] = [
                {key: value for key, value in node.items() if key in node_fields}
                if isinstance(node, dict)
                else node
                for node in raw_nodes
            ]
        try:
            trace = WireTrace.model_validate(trace_data)
        except ValidationError as exc:
            if invalid_results is not None:
                errors = exc.errors()
                reason = errors[0]["msg"] if errors else str(exc)
                invalid_results.append(
                    InvalidResultLine(
                        getattr(sample, "line", sample_number),
                        f"invalid trace: {reason}",
                    )
                )
            continue
        task = trace.task.model_dump(mode="json", exclude_none=True)
        branches = trace.branches
        main_messages = (
            [
                message.model_dump(mode="json", exclude_none=True)
                for message in branches[-1].messages
            ]
            if branches
            else []
        )
        # v0-style split: prompt = messages before the first assistant turn
        first_assistant = next(
            (i for i, m in enumerate(main_messages) if m.get("role") == "assistant"),
            len(main_messages),
        )
        # the trace reward belongs to the completed (final) branch; earlier branches
        # carry no per-step reward, matching v0 samples
        trajectory = [
            {
                "messages": [
                    message.model_dump(mode="json", exclude_none=True)
                    for message in branch.messages
                ],
                "reward": trace.reward if branch is branches[-1] else None,
                "num_input_tokens": branch.num_input_tokens,
                "num_output_tokens": branch.num_output_tokens,
            }
            for branch in branches
        ]
        example_id = trace.task.idx
        rollout_counts[example_id] = rollout_counts.get(example_id, 0) + 1
        info = dict(trace.info)
        info.update(
            {key: value for key, value in sample.items() if key not in trace_fields}
        )

        converted.append(
            {
                "sample_id": trace.id,
                "example_id": example_id,
                "rollout_number": rollout_counts[example_id],
                "task": task,
                "prompt": main_messages[:first_assistant],
                "completion": main_messages[first_assistant:],
                "answer": task.get("answer"),
                "reward": trace.reward,
                "timing": trace.timing.model_dump(mode="json", exclude_none=True),
                "is_completed": trace.is_completed,
                "is_truncated": trace.is_truncated,
                "metrics": trace.metrics,
                "error": (
                    trace.error.model_dump(mode="json", exclude_none=True)
                    if trace.error
                    else None
                ),
                "stop_condition": trace.stop_condition,
                "trajectory": trajectory,
                "token_usage": (
                    trace.usage.model_dump(mode="json", exclude_none=True)
                    if trace.usage
                    else None
                ),
                "num_steps": trace.num_turns,
                "info": info or None,
            }
        )

    return converted


def has_eval_artifacts(directory: Path) -> bool:
    """Return whether a directory contains native or legacy eval artifacts."""
    if (directory / "config.toml").is_file():
        try:
            read_config(directory)
        except ValueError:
            return False
        return (directory / "results.jsonl").is_file()
    return (directory / "metadata.json").exists() and (
        directory / "results.jsonl"
    ).exists()


def resolve_eval_artifact_dir(path: str | Path) -> Path:
    """Validate and return the evaluation artifact directory. A path to one of the run's
    files (config.toml, results.jsonl, ...) resolves to its directory."""
    artifact_path = Path(path)

    if artifact_path.is_file():
        if artifact_path.name not in (
            "config.toml",
            "results.jsonl",
            "eval.log",
            "metadata.json",
        ):
            raise ValueError(
                f"Expected a directory path, but got file: {artifact_path}\n"
                "Pass a directory containing a native V1 run or metadata.json/results.jsonl"
            )
        artifact_path = artifact_path.parent

    if not artifact_path.is_dir():
        raise FileNotFoundError(f"Path not found: {artifact_path}")
    if has_eval_artifacts(artifact_path):
        return artifact_path

    if (artifact_path / "config.toml").exists():
        raise ValueError(
            f"Directory '{artifact_path}' is not a complete Verifiers run artifact"
        )
    has_metadata = (artifact_path / "metadata.json").exists()
    has_results = (artifact_path / "results.jsonl").exists()
    if has_metadata and not has_results:
        raise ValueError(f"Directory '{artifact_path}' is missing results.jsonl")
    if has_results and not has_metadata:
        raise ValueError(f"Directory '{artifact_path}' is missing metadata.json")
    raise ValueError(
        f"Directory '{artifact_path}' is missing both metadata.json and results.jsonl"
    )


def discover_eval_artifact_dirs(outputs_dir: Path = Path("outputs")) -> list[Path]:
    """Find native or legacy eval output directories under an outputs root."""
    if not outputs_dir.exists():
        return []
    candidates = {
        artifact.parent
        for pattern in ("config.toml", "metadata.json")
        for artifact in outputs_dir.rglob(pattern)
    }
    return sorted(
        directory for directory in candidates if has_eval_artifacts(directory)
    )


def read_upload_data(results_dir: Path) -> EvalUploadData:
    """Read native or legacy eval artifacts into a host-uploadable payload."""
    if (results_dir / "config.toml").is_file():
        samples, invalid = read_results(results_dir / "results.jsonl")
        config = read_config(results_dir)
        taskset = config.get("taskset")
        env = (taskset.get("id") if isinstance(taskset, dict) else None) or config.get(
            "id"
        )
        model = config.get("model")
        if not isinstance(env, str) or not isinstance(model, str):
            raise ValueError("Missing taskset.id/id or model in config.toml")
        results = convert_results_for_upload(samples, invalid)
        rewards = [
            row["reward"]
            for row in results
            if isinstance(row.get("reward"), (int, float))
        ]
        metrics = {"reward": sum(rewards) / len(rewards)} if rewards else {}
        metadata: dict[str, Any] = {
            "framework": "verifiers",
            "run_id": results_dir.name,
            "num_examples": config.get("num_tasks"),
            "rollouts_per_example": config.get("num_rollouts"),
            "resolved_config": config,
        }
    else:
        metadata_path = results_dir / "metadata.json"
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        env = raw.get("env_id") or raw.get("env")
        model = raw.get("model")
        if not env or not model:
            raise ValueError(
                f"Missing required 'env_id' or 'model' field in {metadata_path}"
            )
        samples, invalid = read_results(results_dir / "results.jsonl")
        results = convert_results_for_upload(samples, invalid)
        # legacy metadata carries aggregates as avg_<metric> keys
        metrics = {
            key.removeprefix("avg_"): value
            for key, value in raw.items()
            if key.startswith("avg_")
        }
        metadata = {
            key: value for key, value in raw.items() if not key.startswith("avg_")
        }

    return EvalUploadData(
        eval_name=f"{env}-{model}",
        model_name=model,
        env=env,
        metrics=metrics,
        metadata=metadata,
        results=results,
        invalid_results=invalid,
    )


def output_path(config: EvalConfig) -> Path:
    """Where this run writes: `outputs/<taskset>--<model>--<harness>/<uuid>` (or the explicit
    `--output-dir`). The per-run `uuid` leaf means runs never overwrite each other."""
    if config.output_dir is not None:
        return config.output_dir
    if config.is_legacy:
        from verifiers.v1.types import env_name

        assert config.id is not None
        name = f"{env_name(config.id)}--{config.model.replace('/', '--')}--legacy"
        return Path("outputs") / name / config.uuid
    name = f"{config.taskset.name}--{config.model.replace('/', '--')}--{config.harness.name}"
    return Path("outputs") / name / config.uuid


def write_config(config: BaseModel, results_dir: Path) -> Path:
    """Write the run's resolved `config.toml` (re-readable via `@ config.toml`); return its
    path. mode="json" makes values TOML-friendly (Path -> str, etc.); exclude_none drops the
    nulls TOML can't represent."""
    results_dir.mkdir(parents=True, exist_ok=True)
    toml = tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    config_path = results_dir / "config.toml"
    config_path.write_text(toml)
    return config_path


def save_config(config: BaseModel, results_dir: Path) -> None:
    """Set up the run's output dir: write `config.toml` and start a fresh (empty)
    `results.jsonl`. Call once up front, before traces start landing."""
    write_config(config, results_dir)
    (results_dir / "results.jsonl").write_text(
        ""
    )  # fresh; appended to as traces complete


def write_trace(results_dir: Path, trace: Trace) -> None:
    """Serialize and append one trace in the worker thread."""
    data = TypeAdapter(type(trace)).dump_json(trace, exclude_none=True)
    with (results_dir / "results.jsonl").open("ab") as f:
        f.write(data + b"\n")


def read_traces(results_dir: Path, trace_type: type) -> list[Trace]:
    """Load a run's saved traces from `results.jsonl`, typed as `trace_type` — the inverse of
    `write_trace`. Used by `replay` to re-score finished rollouts (pass the taskset's typed
    `Trace[...]`, or `Trace[WireTask, ...]` to read any taskset's traces without importing it).
    Streams line-by-line so a large (multi-GB) results file isn't loaded into memory at once."""
    adapter = TypeAdapter(trace_type)
    traces: list[Trace] = []
    with (results_dir / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(adapter.validate_python(json.loads(line)))
    return traces


async def append_trace(results_dir: Path, trace: Trace, lock: asyncio.Lock) -> None:
    """Append one finished trace without blocking the event loop. The run's shared lock
    preserves whole-line ordering, and awaiting the worker preserves per-trace durability."""

    async def persist() -> None:
        async with lock:
            await asyncio.to_thread(write_trace, results_dir, trace)

    # Run lock acquisition and the worker to completion even under cancellation, so
    # finalized traces are never lost mid-write (`run_shielded` re-raises the cancellation).
    await run_shielded(persist())
