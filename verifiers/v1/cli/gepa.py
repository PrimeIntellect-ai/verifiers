"""The GEPA entrypoint: `uv run gepa [<taskset-id>] --model <model> [options]`.

Registered as the `gepa` console script. Optimizes a v1 taskset's `Task.system_prompt` via
GEPA (Genetic-Pareto): alternating rollouts with a teacher LM reflecting on results — see
`verifiers.v1.gepa`. CLI resolution mirrors `eval`/`serve` (`verifiers.v1.cli.resolve`): a
leading bare token is the taskset id, the taskset/harness subconfigs are narrowed from their
`id`s so `--taskset.*` / `--harness.*` stay typed and `-h` renders them, `@ file.toml` loads,
and the actual parse is `pydantic_config.cli`. v1-native tasksets only — a legacy (v0) env is
rejected; run those through the existing `vf-gepa` command instead.
"""

import asyncio
import logging
import sys

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.output import output_path, write_config
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.gepa import GEPAConfig, run_gepa
from verifiers.v1.utils.interrupt import install_interrupt
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = "usage: uv run gepa [<taskset-id>] [--harness.id <id>] --model <model> [options] [@ file.toml]"


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        cli(
            narrow_config(GEPAConfig, argv)
        )  # full option help, narrowed to the given ids
        return
    if not extract_id(argv, "taskset") and not references_config_file(argv):
        raise SystemExit(
            USAGE
        )  # need a taskset (positional / --taskset.id) or a @ file.toml

    config_type = narrow_config(GEPAConfig, argv)
    sys.argv = [sys.argv[0], *argv]  # let prime-pydantic-config render help/errors
    config = cli(config_type)
    if config.is_legacy:
        raise SystemExit(
            "gepa optimizes native v1 tasksets; run a legacy (v0) environment through "
            "`vf-gepa` instead of `gepa`."
        )
    setup_logging("DEBUG" if config.verbose else "INFO")
    if config.dry_run:  # resolved + validated; write it to the output dir and exit
        logger.info("wrote config to %s", write_config(config, output_path(config)))
        return

    # First Ctrl-C / SIGTERM warns and raises KeyboardInterrupt so a killed/timed-out run still
    # runs the runner's `serving()` teardown (interception pool / tool-server runtimes); further
    # signals during that cleanup are swallowed so an impatient second Ctrl-C can't orphan them.
    install_interrupt()

    env = vf.Environment(config)
    result = asyncio.run(run_gepa(env, config))
    print(f"best system prompt:\n{result.best_candidate.get('system_prompt', '')}")
