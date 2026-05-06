from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import Any, cast


def timing_record(start_time: float | None = None) -> dict[str, float]:
    return {
        "generation_ms": 0.0,
        "scoring_ms": 0.0,
        "total_ms": 0.0,
        "start_time": time.time() if start_time is None else start_time,
    }


def ensure_timing(state: MutableMapping[str, Any]) -> dict[str, float]:
    timing = state.setdefault("timing", timing_record())
    if not isinstance(timing, dict):
        raise TypeError("state.timing must be a mapping.")
    return cast(dict[str, float], timing)


def record_generation_timing(state: MutableMapping[str, Any]) -> None:
    timing = ensure_timing(state)
    start_time = float(timing.get("start_time", time.time()))
    elapsed_ms = (time.time() - start_time) * 1000
    timing["generation_ms"] = elapsed_ms
    timing["total_ms"] = elapsed_ms


def record_scoring_timing(state: MutableMapping[str, Any], start_time: float) -> None:
    timing = ensure_timing(state)
    scoring_ms = (time.time() - start_time) * 1000
    timing["scoring_ms"] = float(timing.get("scoring_ms", 0.0)) + scoring_ms
    timing["total_ms"] = float(timing.get("total_ms", 0.0)) + scoring_ms
