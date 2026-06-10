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
import signal
import sys

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.log import setup_logging
from verifiers.v1.cli.resolve import (
    extract_id,
    local_examples,
    narrow_config,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.cli.runner import run_eval
from verifiers.v1.configs.eval import EvalConfig

USAGE = "usage: uv run eval [<taskset-id>] [--harness.id <id>] [--id <env-id> (legacy)] [options] [@ file.toml]"


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        for kind in ("tasksets", "harnesses"):
            local = local_examples(f"examples/{kind}")
            if local:
                print(f"example {kind}:", ", ".join(local))
        sys.argv = [sys.argv[0], "--help"]
        cli(
            narrow_config(EvalConfig, argv)
        )  # full option help, narrowed to the given ids
        return
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
    if config.dry_run:  # resolved + validated; dump it and skip the run
        print(config.model_dump_json(indent=2, exclude_none=True))
        return
    # The --rich dashboard reads live v1 Rollout state, so it's off for a legacy run.
    rich = config.rich and not config.is_legacy
    # --rich owns the screen, so quiet the per-rollout INFO logs it would replace.
    setup_logging("DEBUG" if config.verbose else "WARNING" if rich else "INFO")

    if config.is_legacy:  # v0 backwards-compat: run the classic env, bridged to Traces
        from verifiers.v1.legacy import run_legacy_eval

        traces = asyncio.run(run_legacy_eval(config))
    else:
        env = vf.Environment(config)
        # Make SIGTERM behave like Ctrl-C (SIGINT) so a killed/timed-out eval still runs
        # each rollout's `finally` — i.e. tears down its docker container / prime sandbox.
        signal.signal(
            signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
        )
        traces = asyncio.run(run_eval(env, config))
    if not rich:  # --rich is the whole output; otherwise dump each trace as JSON
        for trace in traces:
            print(trace.model_dump_json(indent=2, exclude_none=True))
