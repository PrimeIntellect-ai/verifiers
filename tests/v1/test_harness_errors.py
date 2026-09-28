"""Harness exit-code classification: a relayed provider fault is a ProviderError, not a HarnessError."""

import types

import pytest

from verifiers.v1.errors import (
    PROVIDER_ERROR_EXIT_CODE,
    HarnessError,
    ProviderError,
    SandboxError,
)
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils import core
from verifiers.v1.runtimes import ProgramResult


def test_provider_exit_code_constants_agree():
    # core runs standalone in the sandbox and mirrors the constant as a literal.
    assert core.PROVIDER_ERROR_EXIT_CODE == PROVIDER_ERROR_EXIT_CODE


def _runtime(alive: bool):
    async def _alive():
        return alive

    return types.SimpleNamespace(alive=_alive)


async def _check(exit_code: int, *, alive: bool = True, stop_condition=None):
    harness = types.SimpleNamespace(config=types.SimpleNamespace(id="bash"))
    trace = types.SimpleNamespace(stop_condition=stop_condition)
    result = ProgramResult(exit_code=exit_code, stdout="", stderr="boom")
    await Harness._check_result(harness, trace, _runtime(alive), result)


@pytest.mark.asyncio
async def test_provider_exit_code_maps_to_provider_error():
    with pytest.raises(ProviderError) as excinfo:
        await _check(PROVIDER_ERROR_EXIT_CODE)
    assert "model call failed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_other_nonzero_exit_stays_harness_error():
    with pytest.raises(HarnessError):
        await _check(1)


@pytest.mark.asyncio
async def test_dead_runtime_wins_over_provider_exit_code():
    # A provider exit code from a program whose sandbox also died is still a SandboxError.
    with pytest.raises(SandboxError):
        await _check(PROVIDER_ERROR_EXIT_CODE, alive=False)


@pytest.mark.asyncio
async def test_zero_exit_is_clean():
    await _check(0)


@pytest.mark.asyncio
async def test_stopped_turn_ignores_exit_code():
    await _check(PROVIDER_ERROR_EXIT_CODE, stop_condition="tool_stop")
