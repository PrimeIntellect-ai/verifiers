"""The GEPA entrypoint: `uv run gepa --taskset.id <id> --model <model> [options]`.

Registered as the `gepa` console script. Optimizes a v1 taskset's `Task.system_prompt` via
GEPA (Genetic-Pareto): alternating rollouts with a teacher LM reflecting on results — see
`verifiers.v1.gepa`. Parsing is entirely `pydantic_config.cli(GEPAConfig)`: it resolves the
taskset/harness subconfigs from their `id`s (so `--taskset.*` / `--harness.*` stay typed),
loads `@ file.toml`, and renders `-h`. v1-native tasksets only — a legacy (v0) env is rejected;
run those through the existing `vf-gepa` command instead.
"""

import asyncio
import logging
import signal

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.output import output_path, write_config
from verifiers.v1.gepa import GEPAConfig, run_gepa
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    config = cli(GEPAConfig, args=argv, prog="gepa")
    if config.is_legacy:
        raise SystemExit(
            "gepa optimizes native v1 tasksets; run a legacy (v0) environment through "
            "`vf-gepa` instead of `gepa`."
        )
    setup_logging("DEBUG" if config.verbose else "INFO")
    if config.dry_run:  # resolved + validated; write it to the output dir and exit
        logger.info("wrote config to %s", write_config(config, output_path(config)))
        return

    # Make SIGTERM behave like Ctrl-C (SIGINT) so a killed/timed-out run still runs the
    # runner's `serving()` teardown (tears down interception pool / tool-server runtimes).
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    env = vf.Environment(config)
    result = asyncio.run(run_gepa(env, config))
    print(f"best system prompt:\n{result.best_candidate.get('system_prompt', '')}")
