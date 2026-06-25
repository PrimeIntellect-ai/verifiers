"""The eval process entrypoint: run, resolve, capabilities, and direct help."""

import asyncio
import json
import logging
import signal
import sys

from pydantic_config import ConfigFileError, cli

import verifiers.v1 as vf
from verifiers.v1.cli.eval.resolver import resolve_eval
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.cli.output import (
    PROTOCOL_VERSION,
    TRACE_SCHEMA_VERSION,
    output_path,
    write_config,
    write_run_info,
)
from verifiers.v1.cli.resolve import narrow_config, with_positional_taskset
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger("verifiers.v1.cli.eval")

USAGE = (
    "usage: eval run [<taskset-id>] [--harness.id <id>] [--id <env-id> (legacy)] [options] [@ file.toml]\n"
    "       eval run --resume <output-dir>\n"
    "       eval resolve --format json <eval options>\n"
    "       eval --protocol-version"
)
RESOLVE_USAGE = "usage: eval resolve --format json <eval options>"
VERSIONS = {
    "protocol_version": PROTOCOL_VERSION,
    "trace_schema_version": TRACE_SCHEMA_VERSION,
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch the versioned process operations without mutating ``sys.argv``."""
    args = list(sys.argv[1:] if argv is None else argv)

    if args == ["--protocol-version"]:
        print(json.dumps({**VERSIONS, "operations": ["run", "resolve"]}, indent=2))
        return
    if "--protocol-version" in args:
        raise SystemExit(f"{USAGE}\n--protocol-version takes no other arguments")

    if args[:1] == ["resolve"]:
        resolve_args = args[1:]
        if any(arg in ("-h", "--help") for arg in resolve_args):
            print(RESOLVE_USAGE)
            return
        if resolve_args[:2] != ["--format", "json"]:
            raise SystemExit(RESOLVE_USAGE)
        resolve_args = resolve_args[2:]
        try:
            invocation = resolve_eval(resolve_args, prog="eval resolve")
        except (ConfigFileError, ValueError) as exc:
            raise SystemExit(f"{RESOLVE_USAGE}\n{exc}") from exc
        print(
            json.dumps(
                {
                    "operation": "resolve",
                    **VERSIONS,
                    "run_id": invocation.run_id,
                    "output_dir": str(invocation.output_dir),
                    "resume": invocation.resume,
                    "config": invocation.config.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return

    explicit_run = args[:1] == ["run"]
    if explicit_run:
        args = args[1:]
    prog = "eval run" if explicit_run else "eval"
    normalized = with_positional_taskset(args)
    if not normalized or any(arg in ("-h", "--help") for arg in normalized):
        print(USAGE)
        help_args = normalized or ["--help"]
        try:
            # `prog` is supported by cli(), but is missing from its overload declarations.
            cli(  # type: ignore[no-matching-overload]
                narrow_config(EvalConfig, normalized), args=help_args, prog=prog
            )
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        return

    try:
        invocation = resolve_eval(args, prog=prog)
    except (ConfigFileError, ValueError) as exc:
        raise SystemExit(f"{USAGE}\n{exc}") from exc
    config = invocation.config
    if config.is_legacy and config.resume is not None:
        raise SystemExit("--resume is not supported for legacy (v0) evals")
    write_run_info(invocation.output_dir, invocation.run_id)

    level = "DEBUG" if config.verbose else "INFO"
    if config.dry_run:
        setup_logging(level)
        logger.info("wrote config to %s", write_config(config, output_path(config)))
        return

    rich = config.rich and not config.is_legacy
    log_file = str(invocation.output_dir / "eval.log")
    if rich:
        setup_logging(level, log_file=log_file, console=False)
        logging.lastResort = None
    else:
        setup_logging(level, log_file=log_file, console=True)

    # SIGTERM follows the Ctrl-C path so rollout cleanup runs before the process exits.
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    if config.is_legacy:
        from verifiers.v1.legacy import run_legacy_eval

        traces = asyncio.run(run_legacy_eval(config))
    elif config.server:
        from verifiers.v1.cli.eval.runner import run_eval_server

        traces = asyncio.run(run_eval_server(config))
    else:
        traces = asyncio.run(run_eval(vf.Environment(config), config))

    if not rich:
        for trace in traces:
            print(trace.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    main()
