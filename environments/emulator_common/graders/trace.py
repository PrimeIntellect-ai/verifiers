import json
import zlib
from collections.abc import Iterable
from typing import TypeAlias

TraceValue: TypeAlias = str | int | float | bool | None
TraceEvent: TypeAlias = dict[str, TraceValue]


def trace_crc(events: Iterable[TraceEvent]) -> str:
    crc = 0
    for event in events:
        payload = json.dumps(dict(event), sort_keys=True, separators=(",", ":"))
        crc = zlib.crc32(payload.encode("utf-8"), crc)
    return f"{crc & 0xFFFFFFFF:08x}"
