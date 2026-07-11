"""Eval CLI entrypoint."""

import asyncio
import logging
import sys

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.utils.interrupt import install as install_interrupt
from verifiers.v1.utils.logging import setup_logging
from verifiers.v1.cli.output import output_path, write_config
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.cli.eval.resume import load_resume_config, split_resume
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.configs.eval import EvalConfig

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run eval [<taskset-id>] [--harness.id <id>] [--id <env-id> (legacy)] [options] [@ file.toml]\n"
    "       uv run eval --resume <output-dir>   (re-run a previous run's missing/errored rollouts)"
)


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        cli(
            narrow_config(EvalConfig, argv)
        )  # full option help, narrowed to the given ids
        return
    resume_dir, rest = split_resume(argv)
    # re-run a previous run's missing/errored rollouts, in place
    if resume_dir is not None:
        if rest:
            raise SystemExit(
                f"{USAGE}\n--resume re-runs a saved config verbatim and takes no other arguments"
            )
        config = load_resume_config(resume_dir)
    else:
        legacy_id = any(a == "--id" or a.startswith("--id=") for a in argv)  # v0 env id
        if (
            not extract_id(argv, "taskset")
            and not legacy_id
            and not references_config_file(argv)
        ):
            raise SystemExit(
                USAGE
            )  # need a taskset (positional / --taskset.id), a legacy --id, or a @ file.toml

        config_type = narrow_config(EvalConfig, argv)
        sys.argv = [sys.argv[0], *argv]  # let prime-pydantic-config render help/errors
        config = cli(config_type)
        if config.dry_run:  # resolved + validated; write it to the output dir and exit
            setup_logging("DEBUG" if config.verbose else "INFO")
            logger.info("wrote config to %s", write_config(config, output_path(config)))
            return
    if config.is_legacy and config.resume is not None:
        raise SystemExit("--resume is not supported for legacy (v0) evals")
    # Execution path: in-process by default; `--server` opts into the env-server worker pool
    # (the path prime-rl trains through). The `--rich` dashboard reads live in-process Rollout
    # state, so it's in-process only (`server + rich` is rejected at config validation). Legacy
    # always runs in-process via the bridge.
    rich = config.rich and not config.is_legacy
    # Always tee the run's logs to a file under the output dir (in-process and server mode).
    log_file = str(output_path(config) / "eval.log")
    level = "DEBUG" if config.verbose else "INFO"
    if rich:
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
        if (
            config.is_legacy
        ):  # v0 backwards-compat: run the classic env, bridged to Traces
            from verifiers.v1.legacy import run_legacy_eval

            traces = asyncio.run(run_legacy_eval(config))
        elif config.server:  # opt-in: drive rollouts through the env-server worker pool
            from verifiers.v1.cli.eval.runner import run_eval_server

            traces = asyncio.run(run_eval_server(config))
        else:  # in-process (default), with or without the live dashboard
            env = vf.Environment(config)
            traces = asyncio.run(run_eval(env, config))
    except KeyboardInterrupt:
        # Graceful cleanup has already run (each rollout's `finally`); partial results are on
        # disk. Exit on the conventional Ctrl-C code without a traceback.
        raise SystemExit(130)
    if config.push and not rich:
        from verifiers.v1.push import push_traces

        push_traces(traces, config)
    if not rich:  # --rich is the whole output; otherwise dump each trace as JSON
        for trace in traces:
            print(trace.model_dump_json(indent=2, exclude_none=True))
