"""Where a flow's traces go: `traces.jsonl` in verifiers' results format, one rollout per
line, read back by id for a stage that attaches to a finished call."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from verifiers.v1.cli.output import TRACES_FILE, append_trace, type_adapter
from verifiers.v1.trace import Trace, WireTrace


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
        trim_torn_tail(self.file)
        self.lock = asyncio.Lock()
        self._index: dict[str, tuple[int, int]] | None = None  # id -> (offset, length)

    async def append(self, trace: Trace) -> None:
        async with self.lock:
            start = self.file.stat().st_size
            await append_trace(self.root, trace, asyncio.Lock(), env="flow")
            self.index()[trace.id] = (start, self.file.stat().st_size - start)

    def index(self) -> dict[str, tuple[int, int]]:
        """Trace id to its line, built by one scan of the file on first use."""
        if self._index is None:
            self._index, offset = {}, 0
            with self.file.open("rb") as file:
                for line in file:
                    try:
                        for t in json.loads(line)["traces"]:
                            self._index[t["id"]] = (offset, len(line))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
                    offset += len(line)
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
