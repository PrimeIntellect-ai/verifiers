"""Durable output and resume primitives for model-free validation runs."""

import json
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic_core import from_json

from verifiers.v1.cli.eval.resume import task_key
from verifiers.v1.cli.output import CONFIG_FILE, write_config
from verifiers.v1.configs.cli.validate import ValidateConfig

RESULTS_FILE = "results.jsonl"
SUMMARY_FILE = "summary.json"
LOG_FILE = "validate.log"
FINAL_REASONS = frozenset({"valid", "invalid"})
REASONS = ("valid", "invalid", "error", "timeout")

ResultRow = dict[str, Any]
TaskIdentity = tuple[int, str]


def validation_mode(config: ValidateConfig) -> str:
    if config.only_gold:
        return "gold"
    if config.only_setup:
        return "setup"
    return "all"


def output_path(config: ValidateConfig) -> Path:
    """Return an exact explicit output dir, else a fresh eval-shaped run dir."""
    if config.output_dir is not None:
        return config.output_dir
    return Path("outputs") / f"{config.name}--validate" / config.uuid


def identity(position: int, data: Mapping) -> TaskIdentity:
    """Identify a selected task by stable selection position and eval's content key."""
    return position, task_key(data)


def _write_rows(path: Path, rows: Sequence[ResultRow]) -> None:
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def append_result(results_dir: Path, row: ResultRow) -> None:
    """Append one whole task result. A torn final line is ignored on resume."""
    data = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    with (results_dir / RESULTS_FILE).open("ab") as f:
        f.write(data + b"\n")


def _is_final(row: object, target: dict[int, str], mode: str) -> bool:
    if not isinstance(row, dict):
        return False
    position = row.get("task_position")
    reason = row.get("reason")
    return (
        isinstance(position, int)
        and target.get(position) == row.get("task_key")
        and row.get("mode") == mode
        and reason in FINAL_REASONS
        and row.get("valid") is (reason == "valid")
    )


def load_results(
    results_dir: Path, selected: Sequence[TaskIdentity], mode: str
) -> tuple[list[ResultRow], list[int]]:
    """Keep one final valid/invalid row per selected task and return owed positions.

    Missing, malformed, error, and timeout rows are owed. The JSONL is atomically
    canonicalized to its kept rows before new work starts, so resumed runs never
    accumulate duplicate final records.
    """
    path = results_dir / RESULTS_FILE
    target = dict(selected)
    kept: dict[int, ResultRow] = {}
    if path.exists():
        with path.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = from_json(line)
                except ValueError:
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                if _is_final(row, target, mode):
                    position = row["task_position"]
                    if position not in kept:
                        kept[position] = row
    rows = [kept[position] for position, _ in selected if position in kept]
    _write_rows(path, rows)
    owed = [position for position, _ in selected if position not in kept]
    return rows, owed


def summarize(rows: Sequence[ResultRow], total: int, mode: str) -> dict[str, Any]:
    """Build the partial or final full-run report written to summary.json."""
    counts = Counter(row.get("reason") for row in rows)
    missing = max(0, total - len(rows))
    outcomes = {reason: counts[reason] for reason in REASONS}
    outcomes["missing"] = missing
    terminal = outcomes["valid"] + outcomes["invalid"]
    summary: dict[str, Any] = {
        "mode": mode,
        "total": total,
        "recorded": len(rows),
        "terminal": terminal,
        "owed": missing + outcomes["error"] + outcomes["timeout"],
        "outcomes": outcomes,
        "valid_rate": round(outcomes["valid"] / total, 6) if total else None,
    }
    if mode == "all":
        checks: dict[str, dict[str, int]] = {}
        for check in ("gold", "setup"):
            check_counts = Counter(
                row.get(check, {}).get("reason")
                for row in rows
                if isinstance(row.get(check), dict)
            )
            checks[check] = {reason: check_counts[reason] for reason in REASONS}
            checks[check]["missing"] = missing
        summary["checks"] = checks
    return summary


def write_summary(results_dir: Path, summary: Mapping[str, Any]) -> None:
    path = results_dir / SUMMARY_FILE
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def save_run(config: ValidateConfig, results_dir: Path, total: int) -> None:
    """Create a fresh run before dispatching any validation task."""
    write_config(config, results_dir)
    (results_dir / RESULTS_FILE).write_text("")
    write_summary(results_dir, summarize([], total, validation_mode(config)))


def split_resume(argv: list[str]) -> tuple[Path | None, list[str]]:
    for i, arg in enumerate(argv):
        if arg == "--resume":
            if i + 1 >= len(argv):
                raise SystemExit(
                    "--resume needs an output dir: uv run validate --resume <dir>"
                )
            return Path(argv[i + 1]), argv[:i] + argv[i + 2 :]
        if arg.startswith("--resume="):
            return Path(arg.split("=", 1)[1]), argv[:i] + argv[i + 1 :]
    return None, argv


def load_resume_config(resume_dir: Path) -> ValidateConfig:
    """Replay the resolved saved config and point output back at the same run."""
    path = resume_dir / CONFIG_FILE
    if not path.exists():
        raise SystemExit(
            f"--resume: no config.toml in {resume_dir} - not a validate output dir"
        )
    config = ValidateConfig.model_validate(tomllib.loads(path.read_text()))
    config.resume = resume_dir
    config.output_dir = resume_dir
    return config
