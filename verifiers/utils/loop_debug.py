"""Gated per-call-site event-loop timing (VF_LOOP_DEBUG, on by default).

Wrap each blocking call in the env-worker rollout/model path with
``with looptime("category"):`` to attribute where event-loop wall time goes
(render / engine / serialize / offload / materialize / sandbox / gc / ...).

Two views:
  * Per-segment   : logs any single segment >= VF_LOOPTIME_MIN_S (default 0.5s).
  * Per-request   : open a scope with ``looptime_scope_begin/end`` (or the
    ``looptime_scope`` cm) around one model request — on close it emits ONE
    aggregated ``LOOPSUM`` line with the FULL per-category breakdown for that
    request (the "ping cycle"), keyed by id. The scope is an asyncio
    contextvar, so every nested ``looptime`` (including work hopped to
    ``asyncio.to_thread``, which copies the context) rolls up into it.
  * Per-process   : a rolling total across ALL requests in this worker process,
    dumped as ``LOOPAGG`` every VF_LOOPAGG_EVERY requests (default 25).

NOTE: around an ``await`` the measured time includes legitimate suspension; for
on-loop *blocking* attribution compare against the loop-lag / faulthandler dumps.
"""
from __future__ import annotations

import contextvars
import itertools
import logging
import os
import threading
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
try:
    _AGG_EVERY = int(os.getenv("VF_LOOPAGG_EVERY", "25") or 25)
except ValueError:
    _AGG_EVERY = 25

# Crash breadcrumbs (VF_LOOP_BREADCRUMB): emit a START line at entry for the
# heavy NATIVE regions so a worker SIGSEGV can be localized — the last START
# with no matching completion before a worker dies is the crashing call.
_BREADCRUMB = os.getenv("VF_LOOP_BREADCRUMB", "0").strip().lower() not in (
    "0", "off", "false", "no", "",
)
_NATIVE = {
    "render_bridge", "rdr_build_mm_features", "engine_generate",
    "engine_generate_repair", "mm_cleanup", "parse_response", "render",
    "image_processor",
}

# ── Per-request scope (contextvar) + per-process rolling aggregate ──────────
# Each accumulator maps category -> [total_seconds, count].
_REQ_ACC: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "vf_loop_req_acc", default=None
)
_proc_lock = threading.Lock()
_PROC_ACC: dict[str, list] = {}
_PROC_REQS = 0
_counter = itertools.count(1)


def _accumulate(category: str, dt: float) -> None:
    """Add a segment to the current request scope (if any) + the process total."""
    acc = _REQ_ACC.get()
    if acc is not None:
        e = acc.get(category)
        if e is None:
            acc[category] = [dt, 1]
        else:
            e[0] += dt
            e[1] += 1
    with _proc_lock:
        e = _PROC_ACC.get(category)
        if e is None:
            _PROC_ACC[category] = [dt, 1]
        else:
            e[0] += dt
            e[1] += 1


class _ScopeToken:
    __slots__ = ("acc", "tok", "t0", "label", "rid")


def looptime_scope_begin(label: str, rid=None):
    """Open a per-request timing scope. Returns a token for ``..._end``."""
    if not _ENABLED:
        return None
    t = _ScopeToken()
    t.acc = {}
    t.tok = _REQ_ACC.set(t.acc)
    t.t0 = time.perf_counter()
    t.label = label
    t.rid = str(rid) if rid is not None else f"{os.getpid()}.{next(_counter)}"
    return t


def looptime_scope_end(t) -> None:
    """Close a scope: emit the full-path LOOPSUM (+ periodic LOOPAGG)."""
    global _PROC_REQS
    if t is None:
        return
    wall = time.perf_counter() - t.t0
    try:
        _REQ_ACC.reset(t.tok)
    except Exception:
        pass
    acc = t.acc
    if acc:
        # full per-category breakdown, biggest first: cat=secs/count
        parts = " ".join(
            f"{c}={v[0]:.3f}/{v[1]}"
            for c, v in sorted(acc.items(), key=lambda kv: -kv[1][0])
        )
        acct = sum(v[0] for v in acc.values())
    else:
        parts = "(no segments)"
        acct = 0.0
    _log.warning(
        "LOOPSUM pid=%s rid=%s %s wall=%.3fs acct=%.3fs | %s",
        os.getpid(), t.rid, t.label, wall, acct, parts,
    )
    snap = None
    with _proc_lock:
        _PROC_REQS += 1
        n = _PROC_REQS
        if _AGG_EVERY > 0 and n % _AGG_EVERY == 0:
            snap = sorted(
                ((c, v[0], v[1]) for c, v in _PROC_ACC.items()),
                key=lambda x: -x[1],
            )
    if snap is not None:
        agg = " ".join(f"{c}={tt:.1f}/{nn}" for c, tt, nn in snap)
        _log.warning("LOOPAGG pid=%s reqs=%d | %s", os.getpid(), n, agg)


@contextmanager
def looptime_scope(label: str, rid=None):
    t = looptime_scope_begin(label, rid)
    try:
        yield (t.rid if t else None)
    finally:
        looptime_scope_end(t)


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
        _accumulate(category, dt)
        if dt >= _MIN_S:
            _log.warning("looptime %s %.3fs", category, dt)
