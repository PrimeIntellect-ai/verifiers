"""`python -m verifiers.v1.flow check <module:Flow>`, `preflight <module:Flow> [config.json]`,
`run <module:Flow> <rows.jsonl> <run_dir> [config.json]`."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

from verifiers.v1.flow.engine import Engine
from verifiers.v1.flow.flow import Flow
from verifiers.v1.flow.preflight import preflight


def load_flow(spec: str) -> type[Flow]:
    module, _, name = spec.partition(":")
    cls = getattr(importlib.import_module(module), name)
    if not (isinstance(cls, type) and issubclass(cls, Flow)):
        raise SystemExit(f"{spec} is not a Flow subclass")
    return cls


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="flow")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="compile the flow and print its graph")
    check.add_argument("flow")
    pre = sub.add_parser("preflight", help="check credentials and model endpoints")
    pre.add_argument("flow")
    pre.add_argument("config", type=Path, nargs="?")
    pre.add_argument("--no-contact", action="store_true", help="skip GET /models")
    run = sub.add_parser("run", help="run every row of a JSONL file")
    run.add_argument("flow")
    run.add_argument("rows", type=Path)
    run.add_argument("run_dir", type=Path)
    run.add_argument("config", type=Path, nargs="?")
    args = parser.parse_args(argv)

    cls = load_flow(args.flow)  # compiles; a FlowError lists every failed check
    if args.command == "check":
        print(json.dumps(cls.graph.to_json(), indent=1))
        return
    config = (
        cls.config_type().model_validate_json(args.config.read_text())
        if args.config
        else cls.config_type()()
    )
    if args.command == "preflight":
        checks = asyncio.run(preflight(config, contact=not args.no_contact))
        for check in checks:
            print(("ok   " if check.ok else "FAIL ") + str(check))
        sys.exit(0 if all(c.ok for c in checks) else 1)
    rows = [
        json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()
    ]
    results = asyncio.run(Engine(cls(config), args.run_dir).run(rows))
    for result in results:
        print(
            f"{result.row}: {'ok' if result.ok else 'failed'}"
            + (f" — {result.error}" if result.error else "")
        )
    sys.exit(0 if all(r.ok for r in results) else 1)


if __name__ == "__main__":
    main()
