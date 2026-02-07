"""Event utilities for unified callback system."""

from pathlib import Path
from typing import TextIO

from verifiers.types import EvalEvent


class LogStreamFileWriter:
    """Handler that writes log_stream events to files for tailing (#753)."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.handles: dict[str, TextIO] = {}

    async def __call__(self, event: EvalEvent) -> None:
        if event["type"] != "log_stream":
            return

        stream_id = event["stream_id"]
        if stream_id not in self.handles:
            path = event.get("file_path") or self.base_dir / f"{stream_id}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handles[stream_id] = open(path, "w")

        self.handles[stream_id].write(event["data"])
        self.handles[stream_id].flush()  # enable tailing

    def close_all(self):
        """Close all open file handles."""
        for fh in self.handles.values():
            fh.close()
        self.handles.clear()
