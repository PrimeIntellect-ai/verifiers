"""On-disk output: manifest.json + config.toml + results.jsonl + eval.log.

The trace is the full data dump — written verbatim, consumed by the platform
(visualization) and prime-rl (training). config.toml is the run's resolved EvalConfig,
written in the same format the CLI reads (`@ config.toml`), so a run is re-runnable from
its own output. Aggregates (avg reward, etc.) are cheap to recompute from results, so
they aren't stored.

The eval process initializes `config.toml`, `results.jsonl`, and `manifest.json` before
execution. Runners append each trace as it completes, so long-run results are durable as
they land; manifest replacements expose only complete lifecycle records to readers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w
from pydantic import TypeAdapter

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import Trace

if TYPE_CHECKING:
    from collections.abc import Iterator

    from verifiers.v1.cli.eval.resolver import EvalInvocation

EVAL_PROTOCOL_VERSION = 1
TRACE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_SCHEMA = "verifiers.eval-run/v1"


def output_path(config: EvalConfig) -> Path:
    """Where this run writes: `outputs/<taskset>--<model>--<harness>/<uuid>` (or the explicit
    `--output-dir`). The per-run `uuid` leaf means runs never overwrite each other."""
    if config.output_dir is not None:
        return config.output_dir
    if config.is_legacy:
        from verifiers.v1.utils.install import env_name

        assert config.id is not None
        name = f"{env_name(config.id)}--{config.model.replace('/', '--')}--legacy"
        return Path("outputs") / name / config.uuid
    name = f"{config.taskset.name}--{config.model.replace('/', '--')}--{config.harness.name}"
    return Path("outputs") / name / config.uuid


def write_config(config: EvalConfig, results_dir: Path) -> Path:
    """Write the run's resolved `config.toml` (re-readable via `@ config.toml`); return its
    path. mode="json" makes values TOML-friendly (Path -> str, etc.); exclude_none drops the
    nulls TOML can't represent."""
    results_dir.mkdir(parents=True, exist_ok=True)
    toml = tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    config_path = results_dir / "config.toml"
    config_path.write_text(toml)
    return config_path


def save_config(config: EvalConfig, results_dir: Path) -> None:
    """Set up the run's output dir: write `config.toml` and start a fresh (empty)
    `results.jsonl`. Call once up front, before traces start landing."""
    write_config(config, results_dir)
    (results_dir / "results.jsonl").write_text(
        ""
    )  # fresh; appended to as traces complete


def load_manifest(results_dir: Path) -> dict[str, Any] | None:
    """Load a versioned run manifest, or None for a pre-manifest run."""
    path = results_dir / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit(f"--resume: unsupported manifest schema in {path}")
    return manifest


def _write_manifest(results_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically replace ``manifest.json`` so readers never observe a partial update."""
    results_dir.mkdir(parents=True, exist_ok=True)
    temporary = results_dir / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(results_dir / "manifest.json")


@contextlib.contextmanager
def run_artifacts(invocation: EvalInvocation) -> Iterator[None]:
    """Initialize one run's files and track its manifest through a terminal state."""
    now = datetime.now(UTC).isoformat()
    manifest = load_manifest(invocation.output_dir) if invocation.resume else None
    if manifest:
        manifest.update(
            status="running",
            attempt=manifest["attempt"] + 1,
            updated_at=now,
            started_at=now,
            finished_at=None,
            error=None,
        )
    else:
        if invocation.resume:
            created_at = datetime.fromtimestamp(
                (invocation.output_dir / "config.toml").stat().st_mtime, UTC
            ).isoformat()
            results_path = invocation.output_dir / "results.jsonl"
            if not results_path.exists():
                results_path.touch()
        else:
            save_config(invocation.config, invocation.output_dir)
            created_at = now
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "protocol_version": EVAL_PROTOCOL_VERSION,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "run_id": invocation.run_id,
            "status": "running",
            "attempt": 2 if invocation.resume else 1,
            "created_at": created_at,
            "updated_at": now,
            "started_at": now,
            "finished_at": None,
            "artifacts": {
                "config": "config.toml",
                "results": "results.jsonl",
                "log": "eval.log",
            },
            "error": None,
        }
    _write_manifest(invocation.output_dir, manifest)

    status, error = "completed", None
    try:
        yield
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
            status = "cancelled"
        elif not isinstance(exc, SystemExit) or exc.code not in (None, 0):
            status = "failed"
            error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        now = datetime.now(UTC).isoformat()
        manifest.update(status=status, updated_at=now, finished_at=now, error=error)
        _write_manifest(invocation.output_dir, manifest)


def write_trace(results_dir: Path, trace: Trace) -> None:
    """Serialize and append one trace in the worker thread."""
    data = TypeAdapter(type(trace)).dump_json(trace, exclude_none=True)
    with (results_dir / "results.jsonl").open("ab") as f:
        f.write(data + b"\n")


async def append_trace(results_dir: Path, trace: Trace, lock: asyncio.Lock) -> None:
    """Append one finished trace without blocking the event loop. The run's shared lock
    preserves whole-line ordering, and awaiting the worker preserves per-trace durability."""

    async def persist() -> None:
        async with lock:
            await asyncio.to_thread(write_trace, results_dir, trace)

    # Shield lock acquisition and the worker so finalized traces survive cancellation.
    persist_task = asyncio.create_task(persist())
    try:
        await asyncio.shield(persist_task)
    except asyncio.CancelledError:
        await persist_task
        raise
