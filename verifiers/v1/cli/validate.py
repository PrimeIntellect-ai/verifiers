"""Model-free task-validation CLI."""

import asyncio
import contextlib
import json
import logging
import shutil
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic_config import cli
from pydantic_core import from_json

import verifiers.v1 as vf
from verifiers.v1.cli.dashboard import TaskProgress, validate_dashboard
from verifiers.v1.cli.output import write_config
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_taskset_config,
    plugin_errors,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.cli.resume import distribute
from verifiers.v1.configs.cli.validate import ValidateConfig
from verifiers.v1.runtimes import make_runtime
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace, TraceTask
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.decorators import invoke
from verifiers.v1.utils.interrupt import install_interrupt
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

RESULTS_FILE = "results.jsonl"
SUMMARY_FILE = "summary.json"
LOG_FILE = "logs/validate.log"
FINAL_REASONS = frozenset({"valid", "invalid"})
REASONS = ("valid", "invalid", "error", "timeout")

ResultRow = dict[str, Any]

USAGE = (
    "usage: uv run validate [<taskset-id>] [--only-setup | --only-gold] "
    "[-o <output-dir>] [--runtime.type subprocess] [options] [@ file.toml]\n"
    "       uv run validate @ <run-dir>/config.toml --resume   (re-run missing/errored/timed-out tasks)\n"
    "       runs persisted gold and setup-only checks per task (no model)"
)


def _narrow(argv: list[str]) -> type[ValidateConfig]:
    """`ValidateConfig` with `taskset` narrowed to the config type of the id on the CLI — so
    the single `cli()` parse stays typed and `-h` renders the taskset's fields. Absent an id
    (a `@ file.toml` may carry it) the base type is left for the validator to resolve."""
    return narrow_taskset_config(ValidateConfig, extract_id(argv, "taskset"))


def validation_mode(config: ValidateConfig) -> str:
    if config.only_gold:
        return "gold"
    if config.only_setup:
        return "setup"
    return "all"


def output_path(config: ValidateConfig) -> Path:
    """Where this run writes: `output_dir / run.dir` — the same grouping convention as
    eval and training."""
    assert config.run.dir is not None
    return config.output_dir / config.run.dir


def _write_rows(path: Path, rows: Sequence[ResultRow]) -> None:
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def append_result(results_dir: Path, row: ResultRow) -> None:
    data = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    with (results_dir / RESULTS_FILE).open("ab") as f:
        f.write(data + b"\n")


def _is_final(row: object, key: str, mode: str) -> bool:
    if not isinstance(row, dict):
        return False
    reason = row.get("reason")
    return (
        row.get("task_key") == key
        and row.get("mode") == mode
        and reason in FINAL_REASONS
        and row.get("valid") is (reason == "valid")
    )


def load_results(
    results_dir: Path, selected_keys: list[str], mode: str
) -> tuple[list[ResultRow], dict[str, int]]:
    """Keep final rows by eval task-content key; return counts owed per key."""
    path = results_dir / RESULTS_FILE
    targets = Counter(selected_keys)
    good: dict[str, list[ResultRow]] = defaultdict(list)
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
                if not isinstance(row, dict):
                    continue
                key = row.get("task_key")
                if (
                    isinstance(key, str)
                    and key in targets
                    and len(good[key]) < targets[key]
                    and _is_final(row, key, mode)
                ):
                    good[key].append(row)

    owed = {
        key: target - len(good.get(key, []))
        for key, target in targets.items()
        if len(good.get(key, [])) < target
    }
    # Eval spreads debt over content-identical selected tasks in order. Assign the
    # interchangeable kept rows to the remaining positions so reporting positions
    # stay unique even when duplicate task content exists.
    counts = distribute(selected_keys, owed, 1)
    rows = []
    used = Counter()
    for position, (key, count) in enumerate(zip(selected_keys, counts)):
        if count:
            continue
        row = good[key][used[key]]
        used[key] += 1
        rows.append({**row, "task_position": position})
    _write_rows(path, rows)
    return rows, owed


def summarize(rows: Sequence[ResultRow], total: int, mode: str) -> dict[str, Any]:
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
    write_config(config, results_dir, "validate.json")
    (results_dir / RESULTS_FILE).write_text("")
    write_summary(results_dir, summarize([], total, validation_mode(config)))


def _classify(valid: bool, exc: BaseException | None) -> str:
    if valid:
        return "valid"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if exc is not None:
        return "error"
    return "invalid"


def _row(
    task, mode: str, valid: bool, exc: BaseException | None, start: float
) -> ResultRow:
    return {
        "index": task.data.idx,
        "name": task.data.name,
        "mode": mode,
        "valid": bool(valid),
        "reason": _classify(valid, exc),
        "elapsed": round(time.time() - start, 2),
        "error": str(exc) if exc is not None else None,
        "error_type": type(exc).__name__ if exc is not None else None,
    }


async def _run_gold(task: Task, config: ValidateConfig) -> ResultRow:
    start = time.time()
    runtime = make_runtime(
        resolve_runtime_config(config.runtime, task),
        name=f"validate-gold-{task.data.idx}-{uuid4().hex[:8]}",
    )
    setup_timeout = (
        config.timeout.setup
        if config.timeout.setup is not None
        else task.data.timeout.setup
    )
    valid, exc = False, None
    try:
        runtime.env = dict(task.runtime_env())
        trace = Trace(
            task=TraceTask(
                type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
            ),
            state=state_cls(type(task))(),
            # No agent runs here — the info only records the runtime policy.
            agent=vf.AgentInfo(
                config=vf.AgentConfig(runtime=config.runtime),
                name="validate",
                trainable=False,
            ),
        )
        await runtime.start()
        await asyncio.wait_for(
            invoke(task.setup, {"trace": trace, "runtime": runtime}),
            setup_timeout,
        )
        valid = await asyncio.wait_for(task.validate(runtime), config.timeout.total)
    except Exception as e:  # noqa: BLE001 - validation reports plugin failures per task
        exc = e
    finally:
        try:
            await runtime.stop()
        except Exception:
            logger.warning(
                "runtime teardown failed (task %s)", task.data.idx, exc_info=True
            )
    return _row(task, "gold", valid, exc, start)


async def _run_setup(task: Task, config: ValidateConfig) -> ResultRow:
    start = time.time()
    runtime = make_runtime(
        resolve_runtime_config(config.runtime, task),
        name=f"validate-setup-{task.data.idx}-{uuid4().hex[:8]}",
    )
    setup_timeout = (
        config.timeout.setup
        if config.timeout.setup is not None
        else task.data.timeout.setup
    )
    valid, exc = False, None
    try:
        runtime.env = dict(task.runtime_env())
        trace = Trace(
            task=TraceTask(
                type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
            ),
            state=state_cls(type(task))(),
            # No agent runs here — the info only records the runtime policy.
            agent=vf.AgentInfo(
                config=vf.AgentConfig(runtime=config.runtime),
                name="validate",
                trainable=False,
            ),
        )
        await runtime.start()
        await asyncio.wait_for(
            invoke(task.setup, {"trace": trace, "runtime": runtime}),
            setup_timeout,
        )
        valid = True
    except Exception as e:  # noqa: BLE001 - validation reports plugin failures per task
        exc = e
    finally:
        try:
            await runtime.stop()
        except Exception:
            logger.warning(
                "runtime teardown failed (task %s)", task.data.idx, exc_info=True
            )
    return _row(task, "setup", valid, exc, start)


def _all_reason(rows: list[ResultRow]) -> str:
    if all(row["valid"] for row in rows):
        return "valid"
    reasons = {str(row["reason"]) for row in rows}
    if "error" in reasons:
        return "error"
    if "timeout" in reasons:
        return "timeout"
    return "invalid"


def _all_error(rows: list[ResultRow]) -> tuple[str | None, str | None]:
    failed = [row for row in rows if not row["valid"]]
    parts = [f"{row['mode']}: {row['error'] or row['reason']}" for row in failed]
    error_types = {str(row["error_type"]) for row in failed if row["error_type"]}
    return ("; ".join(parts) or None, "+".join(sorted(error_types)) or None)


async def _run_all(task: Task, config: ValidateConfig) -> ResultRow:
    start = time.time()
    gold = await _run_gold(task, config)
    setup = await _run_setup(task, config)
    rows = [gold, setup]
    error, error_type = _all_error(rows)
    return {
        "index": task.data.idx,
        "name": task.data.name,
        "mode": "all",
        "valid": all(row["valid"] for row in rows),
        "reason": _all_reason(rows),
        "elapsed": round(time.time() - start, 2),
        "error": error,
        "error_type": error_type,
        "gold": gold,
        "setup": setup,
    }


async def _validate_task(task: Task, config: ValidateConfig) -> ResultRow:
    if config.only_gold:
        return await _run_gold(task, config)
    if config.only_setup:
        return await _run_setup(task, config)
    return await _run_all(task, config)


async def run_validate(config: ValidateConfig) -> list[dict]:
    taskset = vf.load_taskset(config.taskset)
    if config.num_tasks is None and taskset.INFINITE:
        raise ValueError(
            f"{type(taskset).__name__} is infinite - bound the run with -n"
        )
    selected = taskset.shuffle() if config.shuffle else taskset
    if config.num_tasks is not None:
        selected = selected.head(config.num_tasks)
    tasks = list(selected)
    if isinstance(config.runtime, vf.SubprocessConfig) and any(
        type(t).NEEDS_CONTAINER or t.data.image for t in tasks
    ):
        raise SystemExit(
            "taskset needs a container runtime to validate - pass --runtime.type docker (or prime)"
        )
    mode = validation_mode(config)
    checks = "gold+setup" if mode == "all" else mode
    out = output_path(config)
    selected_keys = [task.hash for task in tasks]
    if not config.resume:
        save_run(config, out, len(tasks))
        rows: list[ResultRow] = []
        counts = [1] * len(tasks)
    else:
        rows, owed = load_results(out, selected_keys, mode)
        counts = distribute(selected_keys, owed, 1)
        write_summary(out, summarize(rows, len(tasks), mode))
    plan = [
        (position, task, key)
        for position, (task, key, count) in enumerate(zip(tasks, selected_keys, counts))
        if count
    ]
    logger.info(
        "%s %d/%d task(s) from %s on the %s runtime (%s)",
        "resuming" if config.resume else "validating",
        len(plan),
        len(tasks),
        config.name,
        config.runtime.type,
        checks,
    )
    logger.info("results: %s", out)

    sem = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    states = [TaskProgress(idx=t.data.idx, name=t.data.name) for t in tasks]
    for row in rows:
        state = states[row["task_position"]]
        state.state = row["reason"]

    write_lock = asyncio.Lock()

    async def _one(position: int, task: Task, key: str) -> ResultRow:
        st = states[position]
        async with sem or contextlib.nullcontext():
            st.start = time.time()
            st.state = "running"
            row = await _validate_task(task, config)
        row["task_position"] = position
        row["task_key"] = key

        async def persist() -> None:
            async with write_lock:
                await asyncio.to_thread(append_result, out, row)
                rows.append(row)
                await asyncio.to_thread(
                    write_summary,
                    out,
                    summarize(rows, len(tasks), mode),
                )

        await run_shielded(persist())
        st.end, st.state = time.time(), row["reason"]
        if not config.rich:  # the dashboard shows this live; otherwise log each task
            detail = f" - {row['error']}" if row["error"] else ""
            logger.info(
                "idx=%s valid=%s reason=%s (%.1fs)%s",
                row["index"],
                row["valid"],
                row["reason"],
                row["elapsed"],
                detail,
            )
        return row

    display = (
        validate_dashboard(states, config, time.time())
        if config.rich
        else contextlib.nullcontext()
    )
    async with display:
        if not plan:
            logger.info(
                "nothing to resume: all %d task(s) are valid or invalid", len(tasks)
            )
            return rows
        await asyncio.gather(
            *(_one(position, task, key) for position, task, key in plan)
        )
    rows.sort(key=lambda row: row["task_position"])
    write_summary(out, summarize(rows, len(tasks), mode))
    return rows


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(
        list(sys.argv[1:]) if argv is None else list(argv), flag="--taskset.id"
    )

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        with plugin_errors():
            cli(_narrow(argv))  # full option help, narrowed to the given taskset
        return
    if not extract_id(argv, "taskset") and not references_config_file(argv):
        raise SystemExit(
            USAGE
        )  # need a taskset (positional / --taskset.id) or a @ file.toml

    with plugin_errors():
        config_type = _narrow(argv)
        sys.argv = [
            sys.argv[0],
            *argv,
        ]  # let prime-pydantic-config render help/errors
        config = cli(config_type)
    out = output_path(config)
    # A named run directory is re-entered only by `--resume` or wiped by `--clean`: any
    # other write into it would overwrite the previous run's results.
    if config.clean and not config.resume and out.exists():
        output_dir = config.output_dir.resolve()
        resolved_out = out.resolve()
        if resolved_out == output_dir or not resolved_out.is_relative_to(output_dir):
            raise SystemExit("--clean requires run.dir to name a child of output_dir")
        shutil.rmtree(out)
    results_file = out / RESULTS_FILE
    if not config.resume and results_file.exists() and results_file.stat().st_size > 0:
        raise SystemExit(
            f"run directory {out} already contains results - append --resume to re-run "
            "its missing/errored tasks, overwrite it with --clean, or pick another --run.name"
        )
    setup_logging(
        "DEBUG" if config.verbose else "INFO",
        log_file=str(out / LOG_FILE),
        console=not config.rich,
    )
    # Graceful shutdown: first Ctrl-C/SIGTERM unwinds each task's teardown `finally`
    # (containers/sandboxes); a second is swallowed so it can't orphan them mid-cleanup.
    install_interrupt()
    try:
        asyncio.run(run_validate(config))
    except KeyboardInterrupt:
        print(f"interrupted; partial results: {out}", file=sys.stderr)
        raise SystemExit(130)
    summary = json.loads((out / SUMMARY_FILE).read_text())
    print(f"results: {out}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
