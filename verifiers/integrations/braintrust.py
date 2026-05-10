"""Braintrust tracing for verifiers.

Patches harness.run, harness.setup_state, runtime.submit_model_request,
and runtime.call_tool to emit the spans that wrap_openai() cannot
capture — rollout boundaries, setup timing, turn grouping, tool timing.

Activated automatically when VF_BRAINTRUST_PROJECT is set, or manually::

    from verifiers.integrations.braintrust import instrument
    instrument(env, project="my-project")
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

_bt: Any = None


def _bt_or_raise() -> Any:
    global _bt
    if _bt is not None:
        return _bt
    try:
        import braintrust

        _bt = braintrust
        return _bt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "pip install braintrust to use this integration"
        ) from e


# -- Turn tracking ---------------------------------------------------------


class _Turns:
    """One per rollout.  model_request closes the current turn;
    tool_call after a closed turn opens the next one."""

    __slots__ = ("_parent", "_span", "_n")

    def __init__(self, parent: Any) -> None:
        self._parent = parent
        self._span: Any = None
        self._n = 0

    def current(self) -> Any:
        if self._span is None:
            self._span = self._parent.start_span(name=f"turn_{self._n}")
        return self._span

    def close(self) -> None:
        if self._span is not None:
            self._span.end()
            self._span = None
            self._n += 1

    def finish(self) -> None:
        self.close()


_ctx: dict[int, _Turns] = {}


# -- Public API -------------------------------------------------------------


def instrument(env: Any, project: str, api_key: str | None = None) -> None:
    """Patch *env* so every rollout emits a Braintrust trace."""
    bt = _bt_or_raise()
    bt.init_logger(project=project, api_key=api_key)
    _patch_run(env.harness, bt)
    _patch_setup(env.harness, bt)
    _patch_submit(env.harness.runtime, bt)
    _patch_tool(env.harness.runtime, bt)


@contextmanager
def traced_group(project: str, name: str = "group", api_key: str | None = None):
    """Wrap a batch of rollouts in a group span."""
    bt = _bt_or_raise()
    bt.init_logger(project=project, api_key=api_key)
    with bt.start_span(name=name, type="task") as span:
        yield span
    bt.flush()


def flush() -> None:
    _bt_or_raise().flush()


# -- Patches ----------------------------------------------------------------


def _patch_run(harness: Any, bt: Any) -> None:
    orig = harness.run

    async def traced(task, state=None):
        with bt.start_span(name="rollout") as span:
            result = await orig(task, state)
            turns = _ctx.pop(id(result), None)
            if turns:
                turns.finish()
            span.log(
                scores=_scores(result),
                metadata=_meta(result),
            )
        return result

    harness.run = traced


def _patch_setup(harness: Any, bt: Any) -> None:
    orig = harness.setup_state

    async def traced(task, state):
        with bt.start_span(name="setup_state"):
            result = await orig(task, state)
        try:
            _ctx[id(result)] = _Turns(bt.current_span())
        except Exception:
            pass
        return result

    harness.setup_state = traced


def _patch_submit(runtime: Any, bt: Any) -> None:
    orig = runtime.submit_model_request

    async def traced(prompt, task, state, tool_defs=None, extras=None):
        turns = _ctx.get(id(state))
        if turns is None:
            return await orig(prompt, task, state, tool_defs=tool_defs, extras=extras)

        turn = turns.current()
        span = turn.start_span(name="model_request")
        n_before = len(state.get("trajectory", []))
        t0 = time.monotonic()

        try:
            resp = await orig(prompt, task, state, tool_defs=tool_defs, extras=extras)
        except BaseException:
            span.end()
            turns.close()
            raise

        traj = state.get("trajectory", [])
        step = traj[-1] if len(traj) > n_before else {}
        span.log(
            input=step.get("prompt"),
            output=step.get("completion") if isinstance(step, dict) else None,
            metrics=_tokens(state, n_before),
            metadata={"elapsed_s": round(time.monotonic() - t0, 3)},
        )
        span.end()
        turns.close()
        return resp

    runtime.submit_model_request = traced


def _patch_tool(runtime: Any, bt: Any) -> None:
    orig = runtime.call_tool

    async def traced(tool_name, task, state, **kw):
        turns = _ctx.get(id(state))
        if turns is None:
            return await orig(tool_name, task, state, **kw)

        span = turns.current().start_span(name=f"tool_call:{tool_name}")
        t0 = time.monotonic()

        try:
            result = await orig(tool_name, task, state, **kw)
        except BaseException:
            span.end()
            raise

        span.log(metadata={"elapsed_s": round(time.monotonic() - t0, 3)})
        span.end()
        return result

    runtime.call_tool = traced


# -- Helpers ----------------------------------------------------------------


def _tokens(state: Mapping, n_before: int) -> dict:
    traj = state.get("trajectory", [])
    if len(traj) <= n_before:
        return {}
    tok = traj[-1].get("tokens") if isinstance(traj[-1], Mapping) else None
    if not isinstance(tok, Mapping):
        return {}
    p = _int(tok.get("prompt_tokens"))
    c = _int(tok.get("completion_tokens"))
    out: dict[str, int] = {}
    if p:
        out["prompt_tokens"] = p
    if c:
        out["completion_tokens"] = c
    if p + c:
        out["tokens"] = p + c
    cache = _int(tok.get("cache_read_input_tokens"))
    if cache or p:
        out["cache_hit_pct"] = round(100 * cache / max(p, 1))
    return out


def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _scores(state: Mapping) -> dict[str, float]:
    s: dict[str, float] = {}
    r = state.get("reward")
    if r is not None:
        try:
            s["reward"] = float(r)
        except (TypeError, ValueError):
            pass
    m = state.get("metrics")
    if isinstance(m, Mapping):
        for k, v in m.items():
            try:
                s[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
    return s


def _meta(state: Mapping) -> dict[str, Any]:
    m: dict[str, Any] = {"is_completed": bool(state.get("is_completed", False))}
    stop = state.get("stop_condition")
    if stop is not None:
        m["stop_condition"] = stop
    traj = state.get("trajectory", [])
    m["num_turns"] = len(traj)
    total = sum(
        _int((step.get("tokens") or {}).get("prompt_tokens"))
        + _int((step.get("tokens") or {}).get("completion_tokens"))
        for step in traj
        if isinstance(step, Mapping)
    )
    if total:
        m["total_tokens"] = total
    return m
