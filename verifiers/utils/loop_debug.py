"""Gated per-call-site event-loop timing (VF_LOOP_DEBUG, on by default).

Wrap each blocking call in the env-worker rollout/model path with
``with looptime("category"):`` to attribute where event-loop wall time goes
(render / engine / serialize / offload / materialize / sandbox / gc / ...).
Logs only segments >= VF_LOOPTIME_MIN_S (default 0.5s) to ``vf.looptime``.
NOTE: around an ``await`` the measured time includes legitimate suspension; for
on-loop *blocking* attribution compare against the loop-lag / faulthandler dumps.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

_log = logging.getLogger("vf.looptime")

_ENABLED = os.getenv("VF_LOOP_DEBUG", "1").strip().lower() not in (
    "0", "off", "false", "no", "",
)
try:
    _MIN_S = float(os.getenv("VF_LOOPTIME_MIN_S", "0.5") or 0.5)
except ValueError:
    _MIN_S = 0.5

# Crash breadcrumbs (VF_LOOP_BREADCRUMB): emit a START line at entry for the
# heavy NATIVE regions so a worker SIGSEGV can be localized — the last START
# with no matching completion before a worker dies is the crashing call. Uses
# the worker logger (comes through full; router relays don't). Allowlisted to
# the native suspects to limit flood.
_BREADCRUMB = os.getenv("VF_LOOP_BREADCRUMB", "0").strip().lower() not in (
    "0", "off", "false", "no", "",
)
_NATIVE = {
    "render_bridge", "rdr_build_mm_features", "engine_generate",
    "engine_generate_repair", "mm_cleanup", "parse_response", "render",
}


@contextmanager
def looptime(category: str):
    if not _ENABLED:
        yield
        return
    if _BREADCRUMB and category in _NATIVE:
        _log.warning("BREADCRUMB START %s", category)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if _BREADCRUMB and category in _NATIVE:
            _log.warning("BREADCRUMB END   %s %.3fs", category, dt)
        if dt >= _MIN_S:
            _log.warning("looptime %s %.3fs", category, dt)
