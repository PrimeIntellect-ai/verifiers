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


@contextmanager
def looptime(category: str):
    if not _ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if dt >= _MIN_S:
            _log.warning("looptime %s %.3fs", category, dt)
