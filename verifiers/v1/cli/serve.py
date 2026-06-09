"""`serve --taskset.id <id> [options]` — serve a taskset's environment over ZMQ.

Mirrors the eval entrypoint (`cli/eval.py`): the taskset and harness are selected by their
`--taskset.id` / `--harness.id` and narrowed to their config types (via `cli/resolve.py`), so
the taskset/harness flags stay typed (`--taskset.*`, `--harness.*`). The server then runs
rollouts on request by task idx.
"""

import asyncio
import signal
import sys

from pydantic_config import cli

from verifiers.v1.cli.log import setup_logging
from verifiers.v1.cli.resolve import (
    extract_id,
    local_examples,
    narrow_config,
    with_positional_taskset,
)
from verifiers.v1.configs.serve import EnvServerConfig
from verifiers.v1.serve.server import EnvServer

USAGE = "usage: uv run serve <taskset-id> [--harness.id <id>] [options] [@ file.toml]"


def main(argv: list[str] | None = None) -> None:
    argv = with_positional_taskset(list(sys.argv[1:]) if argv is None else list(argv))

    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        for kind in ("tasksets", "harnesses"):
            local = local_examples(f"examples/{kind}")
            if local:
                print(f"example {kind}:", ", ".join(local))
        sys.argv = [sys.argv[0], "--help"]
        cli(narrow_config(EnvServerConfig, argv))
        return
    if not extract_id(argv, "taskset"):
        raise SystemExit(USAGE)  # --taskset.id is required

    config_type = narrow_config(EnvServerConfig, argv)
    sys.argv = [sys.argv[0], *argv]
    config = cli(config_type)
    if config.dry_run:
        print(config.model_dump_json(indent=2, exclude_none=True))
        return
    setup_logging("DEBUG" if config.verbose else "INFO")

    server = EnvServer(config, address=config.address)
    # SIGTERM behaves like Ctrl-C so a killed server runs its teardown (closes clients).
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    asyncio.run(server.run())
