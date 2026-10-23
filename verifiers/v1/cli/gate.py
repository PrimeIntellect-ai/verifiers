"""`vf-gate`: show the host's request gates (`verifiers.v1.request_gate`), or resize them live.

    uv run vf-gate                          # each gate: cap, limit, in flight, ...
    uv run vf-gate --resize 48 --model M    # move the cap of M's live gates to 48

A raise admits waiting requests at once; a lower sheds nothing and admits no one until
fewer than the new cap are in flight. A resize holds until no process uses the gate: the
next run starts from its own `max_concurrent`.
"""

import json
import time
from typing import Any

from pydantic_config import cli

from verifiers.v1.configs.cli.gate import GateConfig

COLUMNS = {
    "model": "model",
    "max_concurrent": "cap",
    "limit": "limit",
    "rejections": "429s",
    "in_flight": "in_flight",
    "waiting": "waiting",
    "admitted": "admitted",
    "wait_seconds": "wait_s",
}


def render(states: list[dict[str, Any]], now: float | None = None) -> str:
    """The gates as a table, with how long ago each was last updated."""
    now = time.time() if now is None else now
    rows = [[*COLUMNS.values(), "updated"]]
    for state in states:
        ago = max(0.0, now - state.get("updated", now))
        row = [state.get(key) for key in COLUMNS]
        rows.append(["-" if v is None else str(v) for v in row] + [f"{ago:.0f}s ago"])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(
            cell.ljust(w) if i == 0 else cell.rjust(w)
            for i, (cell, w) in enumerate(zip(row, widths))
        )
        for row in rows
    )


def main(argv: list[str] | None = None) -> None:
    config = cli(GateConfig, args=argv, prog="vf-gate", description=__doc__)
    from verifiers.v1 import request_gate as gate

    directory = config.dir or gate.default_gate_dir()
    if config.resize is not None:
        states = gate.resize(directory, config.resize, config.model)
        if not states:
            which = f" for {config.model}" if config.model else ""
            raise SystemExit(f"vf-gate: no live request gate{which} under {directory}")
    else:
        states = gate.gate_states(directory, config.model)
        if not states and config.output == "table":
            print(f"no request gates under {directory}")
            return
    print(json.dumps(states, indent=2) if config.output == "json" else render(states))


if __name__ == "__main__":
    main()
