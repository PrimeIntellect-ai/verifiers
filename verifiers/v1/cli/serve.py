"""Environment-server CLI entrypoint."""

import sys
from functools import partial

from pydantic_config import cli

from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    plugin_errors,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.cli.serve import ServeConfig
from verifiers.v1.configs.serve import pool_serve_kwargs
from verifiers.v1.serve import serve_env
from verifiers.v1.utils.logging import setup_logging

USAGE = "usage: uv run serve [<taskset-id>] [--env.id <id>] [--legacy.id <env-id> (v0)] [options] [@ file.toml]"


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        with plugin_errors():
            cli(narrow_config(ServeConfig, argv))
        return
    legacy_id = any(a == "--legacy.id" or a.startswith("--legacy.id=") for a in argv)
    # An env-block flag (or a retired flat axis) skips the usage gate so the typed
    # parse renders its did-you-mean / pointer to the new flags instead of a bare
    # usage line.
    typed_axis = any(
        a.startswith(("--env.", "--taskset.", "--harness.", "--serve.")) for a in argv
    )
    if (
        not extract_id(argv, "env.taskset")
        and not legacy_id
        and not references_config_file(argv)
        and not typed_axis
    ):
        raise SystemExit(
            USAGE
        )  # need a taskset (positional / --env.taskset.id), a v0 --legacy.id, or @ file.toml

    with plugin_errors():
        config_type = narrow_config(ServeConfig, argv)
        sys.argv = [sys.argv[0], *argv]
        config = cli(config_type)
    if config.dry_run:
        print(config.model_dump_json(indent=2, exclude_none=True))
        return
    level = "DEBUG" if config.verbose else "INFO"
    setup_logging(level)

    # The pool config decides in-process vs router + worker pool (static or elastic); the
    # frontend speaks the same protocol either way. serve_env owns the SIGTERM teardown.
    # Pool workers are spawned with no logging, so hand serve_env the same setup to apply
    # in each one.
    server_kwargs = (
        {
            "env_id": config.legacy.id,
            "env_args": config.legacy.args,
            "extra_env_kwargs": config.legacy.extra_env_kwargs,
        }
        if config.is_legacy
        # `--serve.max-concurrent` is each v1 worker's episode bound; the legacy
        # bridge has never had one.
        else {"config": config.env, "max_concurrent": config.serve.max_concurrent}
    )
    serve_env(
        **pool_serve_kwargs(config.serve.pool),
        legacy=config.is_legacy,
        address=config.serve.address,
        log_setup=partial(setup_logging, level),
        **server_kwargs,
    )
