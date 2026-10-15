"""An agent process killed by a signal while its box stays up (a box's memory limit) is a
typed `HarnessKilled`, retried as infrastructure, and the run that continues it is told
why it died."""

from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.errors import killed_by
from verifiers.v1.harnesses.null import NullHarness, NullHarnessConfig
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.trace import Error
from verifiers.v1.utils.retries import (
    KILLED_NOTE,
    RETRY_NOTE,
    retry_note,
    retryable_error,
)


class AliveBox:
    async def alive(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("exit_code", "signal"), [(137, 9), (143, 15), (1, None), (0, None), (255, None)]
)
def test_an_exit_status_names_the_signal_that_killed_the_process(exit_code, signal):
    assert killed_by(exit_code) == signal


async def test_a_killed_agent_process_is_a_typed_retryable_harness_failure():
    harness = NullHarness(NullHarnessConfig(id="null"))
    trace = SimpleNamespace(stop_condition=None)
    with pytest.raises(vf.HarnessKilled) as raised:
        await harness._check_result(
            trace, AliveBox(), ProgramResult(exit_code=137, stdout="", stderr="")
        )
    assert raised.value.signal == 9
    assert isinstance(raised.value, vf.HarnessError)
    assert retryable_error(raised.value)
    with pytest.raises(vf.HarnessError) as plain:
        await harness._check_result(
            trace, AliveBox(), ProgramResult(exit_code=1, stdout="", stderr="")
        )
    assert not isinstance(plain.value, vf.HarnessKilled)


def test_a_run_killed_by_a_signal_continues_with_the_killed_note():
    killed = Error.of(vf.HarnessKilled("harness 'bash' was killed", signal=9))
    assert killed.type == "HarnessKilled"
    assert retry_note(killed) == KILLED_NOTE
    assert retry_note(Error.of(vf.HarnessError("exited 1"))) == RETRY_NOTE
    assert retry_note(None) == RETRY_NOTE
