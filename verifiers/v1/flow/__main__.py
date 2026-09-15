"""`python -m verifiers.v1.flow [-v] <module:flow> <rows.jsonl> <run_dir> [@ config.toml] [--<field> <value> ...]`

The flow is an async function `flow(ctx, row)`; its config is the `FlowConfig`
subclass its module exports via `__all__` (as a harness or taskset module exports its
class), else `FlowConfig`, and is parsed like every v1 CLI: `@ file.toml`
loads a file, `--author.model x` sets a field, `-h` after the positionals lists them.
Ctrl-C once drains (in-flight steps finish and record), twice cancels. The same
command against the same run directory resumes. `-v` streams every progress event
(the `events.jsonl` line) to stderr.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path
from types import ModuleType

from pydantic_config import cli

from verifiers.v1.flow.config import FlowConfig
from verifiers.v1.flow.run import Run, drain_on_interrupt

USAGE = (
    "usage: python -m verifiers.v1.flow [-v] <module:flow> <rows.jsonl> <run_dir> "
    "[@ config.toml] [--<field> <value> ...]"
)


def config_type(module: ModuleType) -> type[FlowConfig]:
    """The `FlowConfig` subclass `module` exports via `__all__`, else `FlowConfig`."""
    for name in getattr(module, "__all__", ()):
        obj = getattr(module, name, None)
        if (
            isinstance(obj, type)
            and issubclass(obj, FlowConfig)
            and obj is not FlowConfig
        ):
            return obj
    return FlowConfig


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    verbose = any(arg in ("-v", "--verbose") for arg in args)
    args = [arg for arg in args if arg not in ("-v", "--verbose")]
    if len(args) < 3 or args[0] in ("-h", "--help"):
        raise SystemExit(USAGE)
    target, rows_file, run_dir, *rest = args

    logger = logging.getLogger("verifiers.flow")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    if verbose and not logger.handlers:
        logger.addHandler(logging.StreamHandler())  # stderr, bare messages

    module_name, _, name = target.partition(":")
    module = importlib.import_module(module_name)
    flow = getattr(module, name)
    config = cli(config_type(module), args=rest, prog="flow")
    rows = [
        json.loads(line)
        for line in Path(rows_file).read_text().splitlines()
        if line.strip()
    ]
    run = Run(Path(run_dir), config)

    async def main_async() -> bool:
        drain_on_interrupt(run)
        ok = True
        async for result in run.stream(flow, rows):
            print(
                f"{result.row}: {result.state}"
                + (f" — {result.error}" if result.error else "")
            )
            ok &= result.state == "ok"
        return ok

    sys.exit(0 if asyncio.run(main_async()) else 1)


if __name__ == "__main__":
    main()
