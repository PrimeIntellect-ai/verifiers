"""Eval CLI entrypoint."""

import asyncio
import logging
import shutil
import sys
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig, cli

import verifiers.v1 as vf
from verifiers.v1.cli.eval.resume import load_resume_config
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.cli.output import TRACES_FILE, output_path, write_config
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    plugin_errors,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.cli.eval import EvalConfig, RunConfig
from verifiers.v1.utils.interrupt import install_interrupt
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run eval [<taskset-id>] [--env.id <id>] [options] [@ file.toml]\n"
    "       uv run eval --resume --run.name <name> [-o <output-dir>]   (re-run a previous run's missing/errored rollouts)"
)


class ResumeArgs(BaseConfig):
    """The arguments `--resume` takes: the run to resume, located as
    `output_dir / (run.dir or run.name)`. The run's saved config is then loaded
    verbatim, so nothing else may be passed."""

    run: RunConfig = Field(default_factory=RunConfig)
    output_dir: Path = Field(
        Path("outputs"), validation_alias=AliasChoices("output_dir", "o")
    )


def resume_command(config: EvalConfig) -> str:
    """The `--resume` invocation that targets this config's run dir."""
    parts = ["uv run eval --resume"]
    if config.run.dir != config.run.name:
        parts.append(f"--run.dir {config.run.dir}")
    else:
        parts.append(f"--run.name {config.run.name}")
    if config.output_dir != Path("outputs"):
        parts.append(f"-o {config.output_dir}")
    return " ".join(parts)


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
    # re-run a previous run's missing/errored rollouts, in place
    if "--resume" in argv:
        rest = [arg for arg in argv if arg != "--resume"]
        if rest and not rest[0].startswith("-"):
            raise SystemExit(f"{USAGE}\n--resume locates the run by name, not by path")
        with plugin_errors():
            sys.argv = [sys.argv[0], *rest]
            args = cli(ResumeArgs)
        leaf = args.run.dir or args.run.name
        if leaf is None:
            raise SystemExit(
                f"{USAGE}\n--resume needs --run.name <name> (or --run.dir <dir>)"
            )
        config = load_resume_config(args.output_dir / leaf)
    else:
        # An env-block flag (or a since-moved flat axis) skips the usage gate so the
        # typed parse renders its did-you-mean instead of a bare usage line.
        typed_axis = any(
            a.startswith(("--env.", "--taskset.", "--harness.", "--serve."))
            for a in argv
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
        if config.dry_run:  # resolved + validated; write it to the output dir and exit
            setup_logging("DEBUG" if config.verbose else "INFO")
            logger.info("wrote config to %s", write_config(config, output_path(config)))
            return
    # A named run directory is reused only by `--resume` or wiped by `--clean`: a second
    # run writing into it would overwrite its results.
    if config.clean and config.resume is None and output_path(config).exists():
        shutil.rmtree(output_path(config))
    traces_file = output_path(config) / TRACES_FILE
    if (
        config.resume is None
        and traces_file.exists()
        and traces_file.stat().st_size > 0
    ):
        raise SystemExit(
            f"run directory {output_path(config)} already contains results - resume it with "
            f"`{resume_command(config)}`, overwrite it with --clean, or pick another --run.name"
        )
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
