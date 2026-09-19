"""Inspect and steer runs: `python -m verifiers.v1.flow {inspect,steer,drain} --help`."""

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from verifiers.v1.flow.flow import DRAIN_FILE, TRANSITIONS, UNITS, unit_path
from verifiers.v1.flow.unit import Unit, UnitInspection


class Inspection(BaseModel):
    units: list[UnitInspection[Any]]
    events: Path
    traces: Path
    calls: Path
    draining: bool


def inspect(root: Path, name: str | None = None) -> Inspection:
    root = root.resolve()
    paths = [unit_path(root, name)] if name else sorted((root / UNITS).iterdir())
    return Inspection(
        units=[Unit(path).inspect() for path in paths if (path / ".git").exists()],
        events=root / TRANSITIONS,
        traces=root / "traces.jsonl",
        calls=root / "calls",
        draining=(root / DRAIN_FILE).exists(),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser(
        "inspect", help="Committed states and executing stages as JSON"
    )
    view.add_argument("root", type=Path)
    view.add_argument("unit", nargs="?")
    steer = commands.add_parser(
        "steer", help="Publish boundary controls or settled data updates"
    )
    steer.add_argument("root", type=Path)
    steer.add_argument("unit")
    steer.add_argument("--stage")
    steer.add_argument("--status", choices=("ready", "held", "waiting", "terminal"))
    steer.add_argument("--reason")
    steer.add_argument("--note")
    steer.add_argument(
        "--data", type=Path, help="JSON object of pipeline data fields to update"
    )
    steer.add_argument(
        "--expected", help="Workflow revision; required for data updates"
    )
    drain = commands.add_parser("drain", help="Finish running calls and stop admission")
    drain.add_argument("root", type=Path)
    options = parser.parse_args(argv)
    if options.command == "inspect":
        print(inspect(options.root, options.unit).model_dump_json(indent=2))
    elif options.command == "drain":
        (options.root / DRAIN_FILE).touch()
    else:
        unit = Unit(unit_path(options.root, options.unit))
        revision = unit.steer(
            stage=options.stage,
            status=options.status,
            reason=options.reason,
            note=options.note,
            expected=options.expected,
            data=json.loads(options.data.read_text()) if options.data else None,
        )
        print(json.dumps({"revision": revision}))


if __name__ == "__main__":
    main()
