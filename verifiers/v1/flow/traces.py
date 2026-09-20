"""Where a flow's traces go: `traces.jsonl` in verifiers' results format, one rollout per
line, read back by id for a stage that attaches to a finished call."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from verifiers.v1.cli.output import TRACES_FILE, append_trace, type_adapter
from verifiers.v1.trace import Trace, WireTrace
from verifiers.v1.types import Usage


def trim_torn_tail(file: Path) -> None:
    """Drop the partial last line a kill mid-append left, so the next append does not fuse
    with it into a line no reader can skip. Creates the file."""
    file.touch()
    data = file.read_bytes()
    if data and not data.endswith(b"\n"):
        file.write_bytes(data[: data.rfind(b"\n") + 1])


class Traces:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.file = root / TRACES_FILE
        self.lock = asyncio.Lock()
        self._index: dict[str, tuple[int, int]] = {}  # id -> (offset, length)
        self._offset = 0
        self.usage: dict[str, Usage | None] = {}

    async def append(self, trace: Trace) -> None:
        await append_trace(self.root, trace, self.lock, env="flow")

    def index(self) -> dict[str, tuple[int, int]]:
        """Index new complete records, retaining native usage alongside their offsets."""
        if self.file.exists():
            with self.file.open("rb") as file:
                offset = self._offset
                file.seek(offset)
                for line in file:
                    if not line.endswith(b"\n"):
                        break
                    for t in json.loads(line)["traces"]:
                        self._index[t["id"]] = (offset, len(line))
                        self.usage[t["id"]] = Usage.aggregate(
                            Usage.model_validate(u)
                            for u in [
                                *(
                                    c["usage"]
                                    for c in t["calls"]
                                    if c.get("usage") is not None
                                ),
                                *t.get("extra_usage", []),
                            ]
                        )
                    offset += len(line)
                self._offset = offset
        return self._index

    def get(self, trace_id: str) -> WireTrace | None:
        if (where := self.index().get(trace_id)) is None:
            return None
        with self.file.open("rb") as file:
            file.seek(where[0])
            episode = json.loads(file.read(where[1]))
        adapter = type_adapter(WireTrace)
        return next(
            (
                adapter.validate_python(t)
                for t in episode["traces"]
                if t["id"] == trace_id
            ),
            None,
        )
