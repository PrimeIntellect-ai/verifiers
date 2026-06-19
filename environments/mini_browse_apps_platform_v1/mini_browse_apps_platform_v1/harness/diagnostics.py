"""Small helpers for surfacing Mini Browse sandbox diagnostics."""

from __future__ import annotations

import json
from collections import deque
from typing import Any


async def read_jsonl_tail(
    runtime: Any,
    path: str,
    *,
    max_lines: int = 80,
    max_chars: int = 20_000,
) -> dict[str, Any]:
    """Read a bounded JSONL tail from a sandbox artifact."""

    try:
        raw = await runtime.read(path)
    except Exception as exc:
        return {"path": path, "is_error": True, "error": str(exc)}

    text = raw.decode("utf-8", errors="replace")
    original_chars = len(text)
    if max_chars > 0 and original_chars > max_chars:
        text = text[-max_chars:]
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]

    events: deque[Any] = deque(maxlen=max(0, max_lines))
    parse_errors = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            parse_errors += 1
            events.append(line[:1000])

    return {
        "path": path,
        "is_error": False,
        "events": list(events),
        "event_count": len(events),
        "parse_errors": parse_errors,
        "truncated": original_chars > len(text),
    }
