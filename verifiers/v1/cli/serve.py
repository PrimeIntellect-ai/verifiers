"""`serve --taskset.id <id> [options]` — serve a taskset's environment over ZMQ.

Mirrors the eval entrypoint (`cli/eval.py`): the taskset and harness are selected by their
`--taskset.id` / `--harness.id` and narrowed to their config types (via `cli/resolve.py`), so
the taskset/harness flags stay typed (`--taskset.*`, `--harness.*`). The server then runs
rollouts on request by task idx.
"""

import sys
from functools import partial

from pydantic_config import cli

from verifiers.v1.utils.logging import setup_logging
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.serve import ServeConfig
from verifiers.v1.env import pool_serve_kwargs
from verifiers.v1.serve import serve_env

USAGE = "usage: serve [<taskset-id>] [--harness.id <id>] [--id <env-id> (legacy)] [options] [@ file.toml]"


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        cli(narrow_config(ServeConfig, argv), args=argv or ["--help"], prog="serve")
        return
    legacy_id = any(a == "--id" or a.startswith("--id=") for a in argv)  # v0 env id
    if (
        not extract_id(argv, "taskset")
        and not legacy_id
        and not references_config_file(argv)
    ):
        raise SystemExit(
            USAGE
        )  # need a --taskset.id (v1), a legacy --id (v0), or @ file.toml

    config_type = narrow_config(ServeConfig, argv)
    config = cli(config_type, args=argv, prog="serve")
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
            "env_id": config.id,
            "env_args": config.args,
            "extra_env_kwargs": config.extra_env_kwargs,
        }
        if config.is_legacy
        else {"config": config}
    )
    serve_env(
        **pool_serve_kwargs(config.pool),
        legacy=config.is_legacy,
        address=config.address,
        log_setup=partial(setup_logging, level),
        **server_kwargs,
    )


if __name__ == "__main__":
    main()
