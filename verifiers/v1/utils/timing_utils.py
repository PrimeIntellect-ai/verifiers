from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import cast


def span_record(start: float = 0.0, end: float = 0.0) -> dict[str, float]:
    return {
        "start": start,
        "end": end,
        "duration": max(0.0, end - start) if end > 0.0 else 0.0,
    }


def spans_record() -> dict[str, object]:
    return {"spans": [], "duration": 0.0}


def timing_record(start_time: float | None = None) -> dict[str, object]:
    start = time.time() if start_time is None else start_time
    return {
        "start_time": start,
        "setup": span_record(),
        "generation": span_record(start=start),
        "scoring": span_record(),
        "model": spans_record(),
        "env": spans_record(),
        "total": 0.0,
        "overhead": 0.0,
    }


def ensure_timing(state: MutableMapping[str, object]) -> dict[str, object]:
    timing = state.setdefault("timing", timing_record())
    if not isinstance(timing, dict):
        raise TypeError("state.timing must be a mapping.")
    timing = cast(dict[str, object], timing)
    if "generation_ms" in timing or "total_ms" in timing:
        start = _float_value(timing.get("start_time"), time.time())
        elapsed = (
            _float_value(timing.get("total_ms", timing.get("generation_ms", 0.0)))
            / 1000
        )
        timing.clear()
        timing.update(timing_record(start))
        if elapsed > 0.0:
            _set_span(timing, "generation", start, start + elapsed)
            _set_total(timing, start + elapsed)
    return timing


def record_generation_timing(state: MutableMapping[str, object]) -> None:
    timing = ensure_timing(state)
    start_time = _float_value(timing.get("start_time"), time.time())
    end_time = time.time()
    _set_span(timing, "generation", start_time, end_time)
    _set_total(timing, end_time)


def record_scoring_timing(
    state: MutableMapping[str, object], start_time: float
) -> None:
    timing = ensure_timing(state)
    end_time = time.time()
    _set_span(timing, "scoring", start_time, end_time)
    _set_total(timing, end_time)


def record_model_timing(
    state: MutableMapping[str, object], start_time: float, end_time: float
) -> None:
    timing = ensure_timing(state)
    spans = timing.setdefault("model", spans_record())
    if not isinstance(spans, dict):
        raise TypeError("state.timing.model must be a mapping.")
    spans = cast(dict[str, object], spans)
    span_list = spans.setdefault("spans", [])
    if not isinstance(span_list, list):
        raise TypeError("state.timing.model.spans must be a list.")
    span_list = cast(list[object], span_list)
    span_list.append(span_record(start_time, end_time))
    spans["duration"] = _duration(spans) + max(0.0, end_time - start_time)


def _set_span(
    timing: MutableMapping[str, object], key: str, start_time: float, end_time: float
) -> None:
    timing[key] = span_record(start_time, end_time)


def _set_total(timing: MutableMapping[str, object], end_time: float) -> None:
    start_time = _float_value(timing.get("start_time"), end_time)
    total = max(0.0, end_time - start_time)
    timing["total"] = total
    model = timing.get("model", {})
    env = timing.get("env", {})
    setup = timing.get("setup", {})
    scoring = timing.get("scoring", {})
    timing["overhead"] = max(
        0.0,
        total
        - _duration(setup)
        - _duration(model)
        - _duration(env)
        - _duration(scoring),
    )


def _duration(value: object) -> float:
    if not isinstance(value, dict):
        return 0.0
    mapping = cast(dict[str, object], value)
    duration = mapping.get("duration", 0.0)
    return _float_value(duration, 0.0)


def _float_value(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return default
    return float(value or 0.0)
