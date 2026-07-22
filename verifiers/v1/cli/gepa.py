"""The GEPA entrypoint: `uv run gepa [<taskset-id>] --model <model> [options]`.

Registered as the `gepa` console script. Optimizes a v1 taskset's `Task.system_prompt` via
GEPA (Genetic-Pareto): alternating rollouts with a teacher LM reflecting on results — see
`verifiers.v1.gepa`. CLI resolution mirrors `eval`/`serve` (`verifiers.v1.cli.resolve`): a
leading bare token is the taskset id, the `env` subconfig is narrowed from the ids so
`--env.taskset.*` / `--env.<role>.*` stay typed and `-h` renders them, `@ file.toml` loads,
and the actual parse is `pydantic_config.cli`. v1-native tasksets only — a legacy (v0) env is
rejected; run those through the existing `vf-gepa` command instead.
"""

import logging
import sys

from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.cli.output import output_path, write_config
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    plugin_errors,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.gepa import GEPAConfig, run_gepa
from verifiers.v1.utils.interrupt import install_interrupt
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = "usage: uv run gepa [<taskset-id>] [--env.id <id>] --model <model> [options] [@ file.toml]"


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        with plugin_errors():
            cli(
                narrow_config(GEPAConfig, argv)
            )  # full option help, narrowed to the given ids
        return
    if any(a == "--id" or a.startswith("--id=") for a in argv):  # v0 env id
        raise SystemExit(
            "gepa optimizes native v1 tasksets; run a legacy (v0) environment through "
            "`vf-gepa` instead of `gepa`."
        )
    typed_axis = any(a.startswith(("--env.", "--taskset.", "--harness.")) for a in argv)
    if (
        not extract_id(argv, "env.taskset")
        and not references_config_file(argv)
        and not typed_axis
    ):
        raise SystemExit(
            USAGE
        )  # need a taskset (positional / --env.taskset.id) or a @ file.toml

    with plugin_errors():
        config_type = narrow_config(GEPAConfig, argv)
        sys.argv = [sys.argv[0], *argv]  # let prime-pydantic-config render help/errors
        config = cli(config_type)
    setup_logging("DEBUG" if config.verbose else "INFO")
    # Refuse multi-agent before the dry-run return, so --dry-run can't write a
    # config the real invocation would reject.
    env_cls = vf.environment_class(
        config.env.taskset.id if config.env.taskset is not None else "",
        config.env.id,
    )
    if not issubclass(env_cls, vf.SingleAgentEnv):
        raise SystemExit(
            f"gepa: {config.env.env_id!r} runs {env_cls.__name__}, a multi-agent env; "
            "gepa optimizes one agent's prompt against per-trace rewards and can't "
            "drive a multi-agent interaction — only eval runs those"
        )
    if config.dry_run:  # resolved + validated; write it to the output dir and exit
        logger.info("wrote config to %s", write_config(config, output_path(config)))
        return

    # First Ctrl-C / SIGTERM warns and raises KeyboardInterrupt so a killed/timed-out run still
    # runs the runner's `serving()` teardown (interception / tool-server runtimes); further
    # signals during that cleanup are swallowed so an impatient second Ctrl-C can't orphan them.
    install_interrupt()

    env = vf.load_environment(config.env)
    try:
        result = run_gepa(env, config)
    except KeyboardInterrupt:
        # Graceful cleanup already ran (run_gepa's finally tore down serving); exit on the
        # conventional Ctrl-C code without dumping a traceback.
        raise SystemExit(130)
    print(f"best system prompt:\n{result.best_candidate.get('system_prompt', '')}")
