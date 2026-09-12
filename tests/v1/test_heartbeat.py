"""The loop's heartbeat: a stall noted and logged, a request that straddled one."""

import asyncio
import logging
import time

from verifiers.v1 import heartbeat as beats
from verifiers.v1.heartbeat import Heartbeat


async def test_the_heartbeat_notes_a_stall_of_the_loop_and_a_request_that_straddled_it(
    monkeypatch, caplog
):
    """The heartbeat wakes every `HEARTBEAT_S`; a wakeup over `LAG_S` late is a stall
    `(from, to)`, and `stalled_since(began)` says whether a request begun at `began`
    straddled one, also while the stall it woke from is not yet noted; a heartbeat that
    is not running never says so."""
    monkeypatch.setattr(beats, "HEARTBEAT_S", 0.02)
    monkeypatch.setattr(beats, "LAG_S", 0.1)
    idle = Heartbeat()
    assert not idle.stalled_since(time.monotonic() - 3600)
    beat = Heartbeat()
    beat.start()
    beat.start()  # idempotent
    try:
        assert beat._task is not None and not beat.stalled_since(
            time.monotonic() - 3600
        )
        await asyncio.sleep(0.05)
        before = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="verifiers.v1.heartbeat"):
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
        stalled = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("[loop] stalled ")
        ]
        assert (
            len(stalled) == 1 and float(stalled[0].split()[2]) >= 0.25
        )  # one line per stall
    finally:
        await beat.close()
    assert beat._task is None
    await beat.close()  # idempotent
