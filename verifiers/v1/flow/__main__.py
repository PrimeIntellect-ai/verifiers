"""`python -m verifiers.v1.flow <command> ...`

    run <module:pipeline> <root> [@ config.toml] [--<field> <value> ...]   run until nothing is runnable
    status <root>                                                          every unit: stage, status, reason
    release <root> <unit> [--note TEXT]                                    a held or waiting unit back to ready
    hold <root> <unit> [--note TEXT]                                       park a unit
    route <root> <unit> <stage> [--note TEXT]                              send a unit to a stage, ready
    note <root> <unit> TEXT                                                leave a note for the unit's next stage

The pipeline is a `Pipeline` the module exports by name. `run` against the same root resumes:
units keep their state, calls keep their records. Ctrl-C once drains (calls in flight finish,
units stay ready), twice cancels; so does a `drain` file in the root. Operator commands are
commits on the unit, so they show in its log beside the pipeline's own.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from pathlib import Path

from pydantic_config import cli

from verifiers.v1.flow.flow import CAMPAIGN, TASKS, Flow, Pipeline, drain_on_interrupt
from verifiers.v1.flow.unit import Unit

USAGE = __doc__ or ""


def _unit(root: Path, name: str) -> Unit:
    unit = Unit(root / CAMPAIGN) if name == CAMPAIGN else Unit(root / TASKS / name)
    if not (unit.path / ".git").exists():
        raise SystemExit(f"no unit {name!r} under {root}")
    return unit


def _steer(root: Path, name: str, message: str, state: dict, note: str | None) -> None:
    unit = _unit(root, name)
    if note:
        state["notes"] = [*unit.state().get("notes", []), note]
    sha = unit.commit(message + (f": {note}" if note else ""), state=state)
    print(f"{name}: {message} ({sha[:8]})")


def status(root: Path) -> int:
    units = [
        Unit(root / CAMPAIGN),
        *(Unit(p) for p in sorted((root / TASKS).glob("*")) if (p / ".git").exists()),
    ]
    width = max(len(u.id) for u in units)
    for unit in units:
        st = unit.state()
        line = f"{unit.id:{width}}  {st.get('stage', '?'):12} {st.get('status', '?'):9} {st.get('reason', '')}"
        print(line[:200])
    live = (
        sorted(p.stem for p in (root / "live").glob("*.json"))
        if (root / "live").exists()
        else []
    )
    if live:
        print("live:", ", ".join(live))
    return 0 if all(u.state().get("status") == "terminal" for u in units[1:]) else 1


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
        config = cli(pipeline.config, args=config_args, prog="flow")
        logger = logging.getLogger("verifiers.flow")
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        if verbose and not logger.handlers:
            logger.addHandler(logging.StreamHandler())

        async def go() -> int:
            async with Flow(Path(root), config, pipeline) as flow:
                drain_on_interrupt(flow)
                await flow.sweep()
                counts = await flow.run()
            print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no tasks")
            return 0 if set(counts) <= {"terminal"} else 1

        sys.exit(asyncio.run(go()))
    root = Path(rest[0])
    if command == "status":
        sys.exit(status(root))
    if command == "release":
        _steer(root, rest[1], "release", {"status": "ready"}, note)
    elif command == "hold":
        _steer(
            root,
            rest[1],
            "hold",
            {"status": "held", "reason": note or "held by operator"},
            note,
        )
    elif command == "route":
        _steer(
            root,
            rest[1],
            f"route: {rest[2]}",
            {"stage": rest[2], "status": "ready"},
            note,
        )
    elif command == "note":
        _steer(root, rest[1], "note", {}, " ".join(rest[2:]))
    else:
        raise SystemExit(USAGE)


if __name__ == "__main__":
    main()
