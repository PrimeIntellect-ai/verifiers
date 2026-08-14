"""Eval CLI entrypoint."""

import asyncio
import logging
import shutil
import sys

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.cli.output import TRACES_FILE, output_path, write_config
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
    "       uv run eval @ <run-dir>/config.toml --resume   (re-run the run's missing/errored rollouts)"
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
    # An env-block flag (or a since-moved flat axis) skips the usage gate so the
    # typed parse renders its did-you-mean instead of a bare usage line.
    typed_axis = any(
        a.startswith(("--env.", "--taskset.", "--harness.", "--serve.")) for a in argv
    )
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
    # other write into it — the dry-run config.toml included, which would clobber the
    # config a resume typically re-runs — would overwrite the previous run.
    run_path = output_path(config)
    if config.clean and not config.resume and run_path.exists():
        if not run_path.resolve().is_relative_to(config.output_dir.resolve()):
            raise SystemExit("--clean requires run.dir to remain under output_dir")
        shutil.rmtree(run_path)
    traces_file = run_path / TRACES_FILE
    if not config.resume and traces_file.exists() and traces_file.stat().st_size > 0:
        raise SystemExit(
            f"run directory {run_path} already contains results - append --resume to "
            "re-run its missing/errored rollouts, overwrite it with --clean, or pick "
            "another --run.name"
        )
    if config.dry_run:  # resolved + validated; write it to the output dir and exit
        setup_logging("DEBUG" if config.verbose else "INFO")
        logger.info("wrote config to %s", write_config(config, output_path(config)))
        return
    # Execution path: in-process by default; `--server` opts into the env-server worker pool
    # (the path prime-rl trains through). The `--rich` dashboard reads live in-process run
    # slots, so it's in-process only (`server + rich` is rejected at config validation).
    # Always tee the run's logs to a file under the output dir (in-process and server mode).
    log_file = str(output_path(config) / "eval.log")
    level = "DEBUG" if config.verbose else "INFO"
    if config.rich:
        setup_logging(level, log_file=log_file, console=False)
        # drop stray stdlib records that bypass loguru (else they print over the UI)
        logging.lastResort = None
    else:
        setup_logging(level, log_file=log_file, console=True)
    # First Ctrl-C / SIGTERM warns and raises KeyboardInterrupt so a killed/timed-out eval still
    # runs each rollout's `finally` (tears down containers/sandboxes) and any worker pool it
    # spawned; further signals during that cleanup are swallowed so an impatient second Ctrl-C
    # can't orphan those resources.
    install_interrupt()

    try:
        if config.server:  # opt-in: drive rollouts through the env-server worker pool
            from verifiers.v1.cli.eval.runner import run_eval_server

            episodes = asyncio.run(run_eval_server(config))
        else:  # in-process (default), with or without the live dashboard
            env = vf.load_environment(config.env)
            episodes = asyncio.run(run_eval(env, config))
    except KeyboardInterrupt:
        # Graceful cleanup has already run (each rollout's `finally`); partial results are on
        # disk. Exit on the conventional Ctrl-C code without a traceback.
        raise SystemExit(130)
    if config.push and not config.rich:
        from verifiers.v1.utils.platform import push_traces

        push_traces(episodes, config)
    if not config.rich:  # --rich is the whole output; otherwise dump each trace as JSON
        for episode in episodes:
            for trace in episode.traces:
                print(trace.model_dump_json(indent=2, exclude_none=True))
