"""`python -m verifiers.v1.flow <module:flow> <rows.jsonl> <run_dir> [config.json]`

The flow is an async function `flow(ctx, row)`; its config class is read off its
`config` attribute, else `FlowConfig`. Ctrl-C once drains (in-flight steps finish and
record), twice cancels. The same command against the same run directory resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import signal
import sys
from pathlib import Path

from verifiers.v1.flow.config import FlowConfig
from verifiers.v1.flow.run import Run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="flow")
    parser.add_argument("flow", help="module:function")
    parser.add_argument("rows", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("config", type=Path, nargs="?")
    args = parser.parse_args(argv)

    module, _, name = args.flow.partition(":")
    flow = getattr(importlib.import_module(module), name)
    config_cls: type[FlowConfig] = getattr(flow, "config", FlowConfig)
    config = (
        config_cls.model_validate_json(args.config.read_text())
        if args.config
        else config_cls()
    )
    rows = [
        json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()
    ]
    run = Run(args.run_dir, config)

    async def main_async() -> bool:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        assert task is not None

        def interrupt() -> None:
            if run.status()["draining"]:
                task.cancel()
            else:
                print(
                    "draining: in-flight steps finish; Ctrl-C again cancels",
                    file=sys.stderr,
                )
                run.drain()

        loop.add_signal_handler(signal.SIGINT, interrupt)
        loop.add_signal_handler(signal.SIGTERM, interrupt)
        ok = True
        async for result in run.stream(flow, rows):
            state = "ok" if result.ok else "stopped" if result.stopped else "failed"
            print(
                f"{result.row}: {state}"
                + (f" — {result.error}" if result.error else "")
            )
            ok &= result.ok
        return ok

    sys.exit(0 if asyncio.run(main_async()) else 1)


if __name__ == "__main__":
    main()
