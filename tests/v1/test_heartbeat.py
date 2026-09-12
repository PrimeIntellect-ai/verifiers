"""The loop's heartbeat (a stall noted and logged, a request that straddled one), the
outage classifiers, and the rule that makes an outage of client-side timeouts (three
distinct sessions within the window, none across a stall)."""

import asyncio
import time

import httpx
import pytest

from verifiers.v1 import heartbeat as beats
from verifiers.v1.errors import HarnessError, ProviderError
from verifiers.v1.heartbeat import Heartbeat, Timeouts, outage, timed_out


async def test_the_heartbeat_notes_a_stall_of_the_loop_and_a_request_that_straddled_it(
    monkeypatch,
):
    """The heartbeat wakes every `HEARTBEAT_S`; a wakeup over `LAG_S` late is a stall
    `(from, to)`, and `stalled_since(began)` says whether a request begun at `began`
    straddled one, also while the stall it woke from is not yet noted; a heartbeat that
    is not running never says so."""
    monkeypatch.setattr(beats, "HEARTBEAT_S", 0.02)
    monkeypatch.setattr(beats, "LAG_S", 0.1)
    idle = Heartbeat()
    assert not idle.stalled_since(time.monotonic() - 3600)
    lines: list[str] = []
    beat = Heartbeat(lines.append)
    beat.start()
    beat.start()  # idempotent
    try:
        assert beat._task is not None and not beat.stalled_since(
            time.monotonic() - 3600
        )
        await asyncio.sleep(0.05)
        before = time.monotonic()
        time.sleep(0.25)  # noqa: ASYNC251 - the loop blocked on purpose, as a long synchronous scan blocks it
        assert (
            beat.stalled_since(before) and beat.stalls.maxlen == 64
        )  # not yet noted: the beat itself is late
        await asyncio.sleep(0.05)
        ((began, ended),) = beat.stalls
        assert (
            began <= before < ended
            and beat.stalled_since(before)
            and beat.stalled_since(began - 1)
        )
        assert not beat.stalled_since(time.monotonic())
        stalled = [line for line in lines if line.startswith("[loop] stalled ")]
        assert (
            len(stalled) == 1 and float(stalled[0].split()[2]) >= 0.25
        )  # one line per stall
    finally:
        await beat.close()
    assert beat._task is None
    await beat.close()  # idempotent


def test_an_outage_is_a_typed_provider_error_at_a_gateway_status_and_a_timeout_is_the_clients_own():
    for status in (502, 503, 504):
        assert outage(ProviderError("down", status_code=status))
    for status in (400, 401, 429, 500):
        assert not outage(ProviderError("no", status_code=status))
    assert not outage(HarnessError("exited 1"))
    request = httpx.Request("POST", "http://provider.invalid/v1/chat/completions")
    timeout = ProviderError("timed out", status_code=504)
    timeout.__cause__ = httpx.ConnectTimeout("timed out", request=request)
    refused = ProviderError("refused", status_code=503)
    refused.__cause__ = httpx.ConnectError("refused", request=request)
    assert timed_out(timeout) and outage(timeout)
    assert not timed_out(refused) and outage(refused)
    assert not timed_out(ProviderError("upstream 504", status_code=504))


async def test_three_distinct_sessions_timing_out_within_the_window_are_the_outage_and_a_stall_does_not_count():
    """One or two sessions' timeouts within the window surface to their callers (a
    saturated server answers slowly, not never); the third distinct session's is the
    outage; timeouts older than the window do not count; a timeout whose request
    straddled a stall of the loop is neither noted nor counted."""
    beat = Heartbeat()
    rule = Timeouts(beat)
    now = time.monotonic()
    assert (
        not rule.note("a", now) and not rule.note("a", now) and not rule.note("b", now)
    )
    assert [session for _, session in rule.seen] == ["a", "a", "b"]
    for n, (at, session) in enumerate(
        rule.seen
    ):  # the window passes: the three no longer count
        rule.seen[n] = (at - beats.TIMEOUT_WINDOW_S - 1, session)
    assert not rule.note("c", now) and len(rule.seen) == 1
    assert not rule.note("a", now)
    beat.start()
    try:
        beat.stalls.append((now - 1, now + 60))
        assert (
            not rule.note("b", now) and len(rule.seen) == 2
        )  # across a stall: the stall's, not the endpoint's
        beat.stalls.clear()
        assert rule.note("b", now)  # the third distinct session in the window
    finally:
        await beat.close()
    assert (
        Timeouts(beat, count=2).note("x", now) is False
        and Timeouts(beat, count=1).note("x", now) is True
    )


def test_timeouts_with_a_short_window(monkeypatch):
    rule = Timeouts(Heartbeat(), count=2, window=0.0)
    now = time.monotonic()
    assert not rule.note("a", now)
    time.sleep(0.001)
    assert not rule.note("b", now) and [session for _, session in rule.seen] == [
        "b"
    ]  # `a` fell out of the window


@pytest.mark.parametrize("count", [1, 2])
def test_the_rule_counts_distinct_sessions_not_timeouts(count):
    rule = Timeouts(Heartbeat(), count=count)
    now = time.monotonic()
    assert rule.note(None, now) is (count == 1)
    assert rule.note(None, now) is (
        count == 1
    )  # the same (unknown) session again: still one
