"""The validate entrypoint: `uv run validate <taskset-id> [--runtime.type subprocess] [options]`.

Registered as the `validate` console script — the model-free sibling of `eval`. Where `eval`
runs a model rollout per task, `validate` runs each task's `validate` hook: a per-task check
that the ground truth holds (a SWE row's gold patch makes its tests pass, gsm8k's verifier
accepts the gold answer), in a runtime with the taskset's `setup` already applied. Each task
is provisioned, set up, validated, and torn down independently with bounded concurrency;
results stream to `results.jsonl` as they land (so a crash keeps partial work), with a
`summary.json` at the end.
"""

import asyncio
import contextlib
import json
import logging
import random
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import tomli_w
from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.log import setup_logging
from verifiers.v1.cli.resolve import (
    extract_id,
    local_examples,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.validate import ValidateConfig
from verifiers.v1.env import resolve_runtime_config
from verifiers.v1.runtimes import RetryingRuntime, make_runtime
from verifiers.v1.taskset import Taskset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run validate [<taskset-id>] [--runtime.type subprocess] [options] [@ file.toml]\n"
    "       runs each task's `validate` hook (per-task validation, no model)"
)


def _narrow(argv: list[str]) -> type[ValidateConfig]:
    """`ValidateConfig` with `taskset` narrowed to the config type of the id on the CLI — so
    the single `cli()` parse stays typed and `-h` renders the taskset's fields. Absent an id
    (a `@ file.toml` may carry it) the base type is left for the validator to resolve."""
    taskset_id = extract_id(argv, "taskset")
    if not taskset_id:
        return ValidateConfig
    ftype = vf.taskset_config_type(taskset_id)
    return type(
        ValidateConfig.__name__,
        (ValidateConfig,),
        {"__annotations__": {"taskset": ftype}, "taskset": ftype(id=taskset_id)},
    )


def _output_path(config: ValidateConfig) -> Path:
    """Where this run writes: `outputs/<taskset>--validate/<uuid>` (or `--output-dir`)."""
    if config.output_dir is not None:
        return config.output_dir
    return Path("outputs") / f"{config.name}--validate" / config.uuid


def _write_config(config: ValidateConfig, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.toml").write_text(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
    )


def _classify(valid: bool, exc: BaseException | None) -> str:
    """The outcome reason: `valid` (passed), `invalid` (returned False), `timeout` (a stage
    timed out), or `error` (raised) — the error's message carries the detail."""
    if valid:
        return "valid"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if exc is not None:
        return "error"
    return "invalid"


async def _validate_task(taskset: Taskset, task, config: ValidateConfig) -> dict:
    """Validate one task: provision its runtime, run `setup` then `validate` (each under its
    stage timeout), tear the runtime down, and return the result row. A raised error is
    captured onto the row (one bad task is data, not a crash) — never re-raised."""
    start = time.time()
    runtime = make_runtime(
        resolve_runtime_config(config.runtime, task), name=f"validate-{task.idx}"
    )
    if config.retries.runtime.max_retries > 0:
        runtime = RetryingRuntime(runtime, config.retries.runtime.max_retries)
    setup_timeout = (
        config.setup_timeout if config.setup_timeout is not None else task.setup_timeout
    )
    valid, exc = False, None
    try:
        await runtime.start()
        await asyncio.wait_for(taskset.setup(task, runtime), setup_timeout)
        valid = await asyncio.wait_for(
            taskset.validate(task, runtime), config.validate_timeout
        )
    except Exception as e:
        exc = e
    finally:
        try:
            await runtime.stop()
        except Exception:
            logger.warning("runtime teardown failed (task %s)", task.idx, exc_info=True)
    return {
        "index": task.idx,
        "name": task.name,
        "valid": bool(valid),
        "reason": _classify(valid, exc),
        "elapsed": round(time.time() - start, 2),
        "error": str(exc) if exc is not None else None,
        "error_type": type(exc).__name__ if exc is not None else None,
    }


def _read_prior(path: Path) -> tuple[list[dict], set[int]]:
    """Parse a prior `results.jsonl` into `(rows, validated_indices)` for `--resume`. Malformed
    lines and rows without an integer `index` are skipped."""
    rows, done = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("index"), int):
            rows.append(row)
            done.add(row["index"])
    return rows, done


async def run_validate(
    taskset: Taskset, config: ValidateConfig, out: Path
) -> list[dict]:
    """Run each task's `validate` hook with bounded concurrency, streaming rows to
    `results.jsonl` as they complete. Returns all rows (prior + new on `--resume`)."""
    tasks = taskset.load_tasks()
    if config.shuffle:
        random.Random(0).shuffle(tasks)
    if config.num_tasks is not None:
        tasks = tasks[: config.num_tasks]
    if isinstance(config.runtime, vf.SubprocessConfig) and (
        taskset.NEEDS_CONTAINER or any(t.image for t in tasks)
    ):
        raise SystemExit(
            "taskset needs a container runtime to validate - pass --runtime.type docker (or prime)"
        )

    results_path = out / "results.jsonl"
    prior: list[dict] = []
    if config.resume and results_path.exists():
        prior, done = _read_prior(results_path)
        tasks = [t for t in tasks if t.idx not in done]
        logger.info(
            "resuming %s - %d already validated, %d to go",
            results_path,
            len(done),
            len(tasks),
        )
    else:
        results_path.write_text("")
    logger.info(
        "validating %d task(s) from %s on the %s runtime",
        len(tasks),
        config.name,
        config.runtime.type,
    )

    sem = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None

    async def _one(task) -> dict:
        async with sem or contextlib.nullcontext():
            row = await _validate_task(taskset, task, config)
        # Sync single-line append: race-free across rollouts in the one event loop.
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    results = list(prior)
    bar = (
        tqdm(total=len(tasks), desc="validate", dynamic_ncols=True)
        if tqdm is not None
        else None
    )
    for fut in asyncio.as_completed([_one(t) for t in tasks]):
        row = await fut
        results.append(row)
        if bar is not None:
            bar.update(1)
            bar.set_postfix(valid=sum(r["valid"] for r in results))
        else:
            detail = f" - {row['error']}" if row["error"] else ""
            logger.info(
                "idx=%s valid=%s reason=%s (%.1fs)%s",
                row["index"],
                row["valid"],
                row["reason"],
                row["elapsed"],
                detail,
            )
    if bar is not None:
        bar.close()
    return results


def _summarize(results: list[dict], out: Path) -> dict:
    total = len(results)
    valid = sum(1 for r in results if r["valid"])
    by_reason = dict(Counter(r["reason"] for r in results))
    summary = {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "pass_rate": round(valid / total, 4) if total else 0.0,
        "by_reason": by_reason,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "validated %d task(s) - %d valid, %d invalid (%.1f%%) - reasons: %s",
        total,
        valid,
        total - valid,
        100 * summary["pass_rate"],
        by_reason,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        local = local_examples("examples/tasksets")
        if local:
            print("example tasksets:", ", ".join(local))
        sys.argv = [sys.argv[0], "--help"]
        cli(_narrow(argv))  # full option help, narrowed to the given taskset
        return
    if not extract_id(argv, "taskset") and not references_config_file(argv):
        raise SystemExit(
            USAGE
        )  # need a taskset (positional / --taskset.id) or a @ file.toml

    config_type = _narrow(argv)
    sys.argv = [sys.argv[0], *argv]  # let prime-pydantic-config render help/errors
    config = cli(config_type)
    # Load + check the taskset before any side effects (no output dir for a bad run): an
    # un-overridden `validate` would report every task valid without checking anything, so
    # refuse it (see Taskset.validate).
    taskset = vf.load_taskset(config.taskset)
    if type(taskset).validate is Taskset.validate:
        raise SystemExit(
            f"taskset {config.name!r} does not implement a `validate` hook - nothing to "
            "validate (override `Taskset.validate` to make it validatable)"
        )
    out = _output_path(config)
    setup_logging(
        "DEBUG" if config.verbose else "INFO",
        log_file=str(out / "validate.log"),
        console=True,
    )
    if config.dry_run:  # resolved + validated; write it to the output dir and exit
        _write_config(config, out)
        logger.info("wrote config to %s", out / "config.toml")
        return
    # Make SIGTERM behave like Ctrl-C so a killed run still runs each task's `finally`
    # (tears down containers/sandboxes) — and the atexit backstop catches the rest.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    _write_config(config, out)
    logger.info("results: %s", out)
    results = asyncio.run(run_validate(taskset, config, out))
    _summarize(results, out)


if __name__ == "__main__":
    main()
