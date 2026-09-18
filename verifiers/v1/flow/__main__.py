"""`python -m verifiers.v1.flow <command> ...`

    run <module:pipeline> <root> [@ config.toml] [--<field> <value> ...]   run until nothing is runnable
    status <root> [unit] [--json]                                          committed state, active execution, calls
    release <root> <unit> [--note TEXT]                                    a held or waiting unit back to ready
    hold <root> <unit> [--note TEXT]                                       park a unit
    route <root> <unit> <stage> [--note TEXT]                              send a unit to a stage, ready
    note <root> <unit> TEXT                                                leave a note for the unit's next stage
    update <root> <unit> <file.json> --expected SHA [--stage STAGE] [--status STATUS]
    drain <root>                                                          finish in-flight calls and stop

The pipeline is a `Pipeline` the module exports by name. `run` against the same root resumes:
units keep their state, calls keep their records. Ctrl-C once drains (calls in flight finish,
units stay ready), twice cancels; so does a `drain` file in the root. Operator commands are
commits on the unit, so they show in its log beside the pipeline's own.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic_config import cli

from verifiers.v1.flow.calls import Record
from verifiers.v1.flow.flow import (
    CAMPAIGN,
    TASKS,
    Flow,
    Pipeline,
    drain_on_interrupt,
    succeeded,
    task_path,
)
from verifiers.v1.flow.unit import Unit

USAGE = __doc__ or ""


def _unit(root: Path, name: str) -> Unit:
    path = root / CAMPAIGN if name == CAMPAIGN else task_path(root, name)
    if not (path / ".git").exists():
        raise SystemExit(f"no unit {name!r} under {root}")
    return Unit(path)


def inspect(root: Path, name: str | None = None) -> dict:
    units = (
        [_unit(root, name)]
        if name
        else [
            _unit(root, CAMPAIGN),
            *(
                Unit(p)
                for p in sorted((root / TASKS).iterdir())
                if (p / ".git").exists()
            ),
        ]
    )
    ids = {unit.id for unit in units}
    calls = [
        Record.model_validate_json(file.read_text()).model_dump(mode="json")
        for unit in sorted(ids)
        for file in sorted((root / "calls" / unit).glob("*.json"))
    ]
    return {
        "units": [unit.inspect() for unit in units],
        "calls": calls,
        "draining": (root / "drain").exists(),
    }


def status(root: Path, *, name: str | None = None, as_json: bool = False) -> int:
    snapshot = inspect(root, name)
    if as_json:
        print(json.dumps(snapshot, indent=2))
    else:
        for unit in snapshot["units"]:
            state = unit["state"]
            activity = (
                f"running {unit['active']['stage']}" if unit["active"] else "settled"
            )
            print(
                f"{unit['unit']}  {state['stage']}  {state['status']}  {activity}"
                f"{'  DIRTY' if unit['dirty'] else ''}  {state['reason']}"
            )
    return (
        0
        if succeeded(
            _unit(root, CAMPAIGN),
            [Unit(p) for p in (root / TASKS).iterdir() if (p / ".git").exists()],
        )
        else 1
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    verbose = "-v" in args
    args = [a for a in args if a != "-v"]
    if not args or args[0] in ("-h", "--help"):
        raise SystemExit(USAGE)
    command, *rest = args
    note = None
    if "--note" in rest:
        i = rest.index("--note")
        note = rest[i + 1]
        del rest[i : i + 2]

    if command == "run":
        target, root, *config_args = rest
        module_name, _, name = target.partition(":")
        pipeline: Pipeline = getattr(importlib.import_module(module_name), name)
        config = cli(pipeline.config, args=config_args)
        logger = logging.getLogger("verifiers.flow")
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        if verbose and not logger.handlers:
            logger.addHandler(logging.StreamHandler())

        async def go() -> int:
            async with Flow(Path(root), config, pipeline) as flow:
                drain_on_interrupt(flow)
                await flow.sweep()
                counts = await flow.run()
                success = not flow.draining and succeeded(flow.campaign, flow.tasks())
            print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no tasks")
            return 0 if success else 1

        sys.exit(asyncio.run(go()))
    parser = argparse.ArgumentParser(prog=f"flow {command}")
    parser.add_argument("root", type=Path)
    if command == "status":
        parser.add_argument("unit", nargs="?")
        parser.add_argument("--json", action="store_true")
        options = parser.parse_args(rest)
        sys.exit(status(options.root, name=options.unit, as_json=options.json))
    if command == "drain":
        options = parser.parse_args(rest)
        (options.root / "drain").touch()
        return
    parser.add_argument("unit")
    if command == "route":
        parser.add_argument("stage")
    elif command == "note":
        parser.add_argument("text")
    elif command == "update":
        parser.add_argument(
            "file", type=Path, help="JSON object of pipeline data fields to update"
        )
        parser.add_argument(
            "--expected", required=True, help="workflow revision from status --json"
        )
        parser.add_argument("--stage")
        parser.add_argument(
            "--status", choices=("ready", "held", "waiting", "terminal")
        )
    elif command not in ("hold", "release"):
        raise SystemExit(USAGE)
    options = parser.parse_args(rest)
    changes: dict[str, Any] = {"note": note}
    if command == "hold":
        changes.update(status="held", reason=note or "held by operator")
    elif command == "release":
        changes.update(status="ready")
    elif command == "route":
        changes.update(stage=options.stage, status="ready")
    elif command == "note":
        changes.update(note=options.text)
    else:
        changes.update(
            data=json.loads(options.file.read_text()),
            expected=options.expected,
            stage=options.stage,
            status=options.status,
        )
    sha = _unit(options.root, options.unit).steer(**changes)
    print(f"{options.unit}: {command} ({sha})")


if __name__ == "__main__":
    main()
