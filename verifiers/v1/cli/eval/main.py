"""The eval entrypoint: `uv run eval --taskset.id <id> [options]`.

Registered as the `eval` console script. Mirrors the `~/prime-rl` pattern (`config =
cli(Config)`). The taskset and harness are selected by their `--taskset.id` / `--harness.id`
(the discriminator fields); `cli/resolve.py` narrows each to its config type, so the single
`prime-pydantic-config` parse keeps their fields typed and overridable via dotted flags
(e.g. `--harness.runtime.type docker`, `--taskset.*`) / `@ eval.toml`.

`-h`/`--help` (or no args) prints the local example tasksets/harnesses plus the full, typed
pydantic-config help — narrowed to whatever `--taskset.id` / `--harness.id` are given.
"""

import asyncio
import json
import logging
import signal
import sys

from pydantic_config import ConfigFileError, cli

import verifiers.v1 as vf
from verifiers.v1.cli.eval.resolver import resolve_eval
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.cli.output import output_path, write_config
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
VERSIONS = {"protocol_version": 1, "trace_schema_version": 1}


def main(argv: list[str] | None = None) -> None:
    """Dispatch the versioned process operations without mutating ``sys.argv``."""
    args = list(sys.argv[1:] if argv is None else argv)

    if args == ["--protocol-version"]:
        print(json.dumps({**VERSIONS, "operations": ["run", "resolve"]}, indent=2))
        return

    if args[:1] == ["resolve"]:
        if any(arg in ("-h", "--help") for arg in args[1:]):
            print(RESOLVE_USAGE)
            return
        if args[1:3] != ["--format", "json"]:
            raise SystemExit(RESOLVE_USAGE)
        try:
            config = resolve_eval(args[3:], prog="eval resolve")
        except (ConfigFileError, ValueError) as exc:
            raise SystemExit(f"{RESOLVE_USAGE}\n{exc}") from exc
        print(
            json.dumps(
                {
                    "operation": "resolve",
                    **VERSIONS,
                    "run_id": config.uuid,
                    "output_dir": str(output_path(config)),
                    "resume": config.resume is not None,
                    "config": config.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return

    explicit_run = args[:1] == ["run"]
    if explicit_run:
        args = args[1:]
    prog = "eval run" if explicit_run else "eval"
    args = with_positional_taskset(args)
    if not args or any(arg in ("-h", "--help") for arg in args):
        print(USAGE)
        # `prog` is supported by cli(), but is missing from its overload declarations.
        cli(  # type: ignore[no-matching-overload]
            narrow_config(EvalConfig, args), args=args or ["--help"], prog=prog
        )
        return

    try:
        config = resolve_eval(args, prog=prog)
    except (ConfigFileError, ValueError) as exc:
        raise SystemExit(f"{USAGE}\n{exc}") from exc

    if config.dry_run:
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
    # Make SIGTERM behave like Ctrl-C (SIGINT) so a killed/timed-out eval still runs each
    # rollout's `finally` (tears down containers/sandboxes) and any worker pool it spawned.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    if config.is_legacy:  # v0 backwards-compat: run the classic env, bridged to Traces
        from verifiers.v1.legacy import run_legacy_eval

        traces = asyncio.run(run_legacy_eval(config))
    elif config.server:  # opt-in: drive rollouts through the env-server worker pool
        from verifiers.v1.cli.eval.runner import run_eval_server

        traces = asyncio.run(run_eval_server(config))
    else:  # in-process (default), with or without the live dashboard
        env = vf.Environment(config)
        traces = asyncio.run(run_eval(env, config))
    if not rich:  # --rich is the whole output; otherwise dump each trace as JSON
        for trace in traces:
            print(trace.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    main()
