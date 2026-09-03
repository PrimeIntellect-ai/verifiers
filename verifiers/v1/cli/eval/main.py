"""Eval CLI entrypoint."""

import asyncio
import json
import logging
import shutil
import sys

from pydantic_config import cli

from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.cli.output import (
    TRACES_FILE,
    create_attempt_log_dir,
    output_path,
    saved_config_path,
    write_config,
)
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    plugin_errors,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.utils.interrupt import install_interrupt
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run eval [<taskset-id>] [--env.id <id>] [options] [@ file.toml]\n"
    "       uv run eval @ <run-dir>/configs/resolved/eval.json --resume   (re-run the run's missing/errored rollouts)"
)


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        with plugin_errors():
            cli(
                narrow_config(EvalConfig, argv)
            )  # full option help, narrowed to the given ids
        return
    # An env-block flag skips the usage gate so the typed parse renders its
    # did-you-mean instead of a bare usage line.
    typed_axis = any(a.startswith(("--env.", "--serve.")) for a in argv)
    if (
        not extract_id(argv, "env.taskset")
        and not references_config_file(argv)
        and not typed_axis
    ):
        raise SystemExit(
            USAGE
        )  # need a taskset (positional / --env.taskset.id) or a @ file.toml

    with plugin_errors():
        config_type = narrow_config(EvalConfig, argv)
        sys.argv = [
            sys.argv[0],
            *argv,
        ]  # let prime-pydantic-config render help/errors
        config = cli(config_type)
    # A named run directory is re-entered only by `--resume` or wiped by `--clean`: any
    # other write into it — the dry-run config included, which would clobber the
    # config a resume typically re-runs — would overwrite the previous run.
    run_path = output_path(config)
    if config.clean and not config.resume and run_path.exists():
        output_dir = config.output_dir.resolve()
        resolved_run_path = run_path.resolve()
        if resolved_run_path == output_dir or not resolved_run_path.is_relative_to(
            output_dir
        ):
            raise SystemExit("--clean requires run.dir to name a child of output_dir")
        shutil.rmtree(run_path)
    traces_file = run_path / TRACES_FILE
    if not config.resume and traces_file.exists() and traces_file.stat().st_size > 0:
        raise SystemExit(
            f"run directory {run_path} already contains results - append --resume to "
            "re-run its missing/errored rollouts, overwrite it with --clean, or pick "
            "another --run.name"
        )
    if config.resume:
        # A resumed eval is only trustworthy under the exact config that produced the
        # existing rollouts — anything else silently mixes incomparable results. Saved
        # configs are full JSON dumps (nulls included), so the comparison is exact.
        saved_path = saved_config_path(run_path)
        if saved_path is None:
            raise SystemExit(
                f"--resume: no saved config under {run_path} - not a run dir"
            )
        with plugin_errors():
            saved = config_type.model_validate_json(saved_path.read_text())
        saved_dump = saved.model_dump(mode="json")
        current_dump = config.model_dump(mode="json")
        saved_json = json.dumps(saved_dump, sort_keys=True, separators=(",", ":"))
        current_json = json.dumps(current_dump, sort_keys=True, separators=(",", ":"))
        if saved_json != current_json:
            changed = sorted(
                key
                for key in set(saved_dump) | set(current_dump)
                if saved_dump.get(key) != current_dump.get(key)
            )
            raise SystemExit(
                f"--resume requires the exact config the run was started with - it "
                f"differs in [{', '.join(changed)}]. Resumed rollouts would not be "
                f"comparable; re-run with `uv run eval @ {saved_path} --resume`, or "
                "start a fresh run"
            )
    if config.dry_run:  # resolved + validated; write it to the output dir and exit
        setup_logging("DEBUG" if config.verbose else "INFO")
        logger.info("wrote config to %s", write_config(config, run_path))
        return
    # Always tee this attempt's logs to `logs/attempt_<n>/eval.log` (`logs/latest`
    # points there) — in server mode (the default) the workers write there too, and
    # `--rich.show-logs` tails it live.
    log_file = str(create_attempt_log_dir(run_path) / "eval.log")
    level = "DEBUG" if config.verbose else "INFO"
    setup_logging(level, log_file=log_file, console=config.rich is None)
    # First Ctrl-C / SIGTERM warns and raises KeyboardInterrupt so a killed/timed-out eval still
    # runs each rollout's `finally` (tears down containers/sandboxes) and any worker pool it
    # spawned; further signals during that cleanup are swallowed so an impatient second Ctrl-C
    # can't orphan those resources.
    install_interrupt()

    try:
        # Through the env-server worker pool by default; in-process with --no-serve.
        episodes = asyncio.run(run_eval(config))
    except KeyboardInterrupt:
        # Graceful cleanup has already run (each rollout's `finally`); partial results are on
        # disk. Exit on the conventional Ctrl-C code without a traceback.
        raise SystemExit(130)
    if config.push and config.rich is None:
        from verifiers.v1.utils.platform import push_traces

        push_traces(episodes, config)
    if (
        config.rich is None
    ):  # --rich is the whole output; otherwise dump each trace as JSON
        for episode in episodes:
            for trace in episode.traces:
                print(trace.model_dump_json(indent=2, exclude_none=True))
