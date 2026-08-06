"""Deterministic guard tests for the Prime Agent live capability fixtures."""

from types import SimpleNamespace

import pytest
from prime_agent_failed_turn_v1 import has_raised_provider_failure
from prime_agent_ipython_cell_v1 import (
    CELL,
    has_ipython_cell_acp_shape,
    has_ipython_cell_call,
)
from prime_agent_persistence_v1 import (
    FIRST_CELL,
    SECOND_CELL,
    PrimeAgentPersistenceTask,
)

import verifiers.v1 as vf


def test_ipython_cell_guard_requires_the_exact_ipython_raw_code():
    trace = SimpleNamespace(
        info={
            "prime_agent_segments": [
                {
                    "last_reply": "DONE",
                    "terminated": False,
                    "tool_calls": [
                        {"name": "ipython", "arguments": f'{{"code": {CELL!r}}}'},
                    ],
                }
            ]
        }
    )
    trace.info["prime_agent_tool_calls"] = [
        {"title": "IPython cell", "rawInput": {"code": CELL}}
    ]
    assert has_ipython_cell_call(trace)
    assert has_ipython_cell_acp_shape(trace)
    trace.info["prime_agent_segments"][0]["tool_calls"][0]["arguments"] = (
        '{"code":"print(\'wrong\')"}'
    )
    assert not has_ipython_cell_call(trace)
    trace.info["prime_agent_tool_calls"][0]["title"] = "Python cell"
    assert not has_ipython_cell_acp_shape(trace)


def test_failed_turn_guard_rejects_a_clean_stop_reason():
    failed = SimpleNamespace(
        ok=False,
        last_error=SimpleNamespace(type="HarnessError"),
        stop_condition=None,
        calls=[SimpleNamespace(error=SimpleNamespace(type="ProviderError"))],
    )
    assert has_raised_provider_failure(failed)
    failed.stop_condition = "agent_completed"
    assert not has_raised_provider_failure(failed)
    failed.stop_condition = None
    failed.calls[-1].error = None
    assert not has_raised_provider_failure(failed)


@pytest.mark.asyncio
async def test_kernel_persistence_guard_rejects_a_reimported_secret():
    token = "a" * 64
    trace = SimpleNamespace(
        info={
            "prime_agent_segments": [
                {
                    "last_reply": "READY",
                    "tool_outputs": [token],
                    "tool_calls": [f'{{"code": {FIRST_CELL!r}}}'],
                    "terminated": False,
                },
                {
                    "last_reply": token,
                    "tool_outputs": [token],
                    "tool_calls": [f'{{"code": {SECOND_CELL!r}}}'],
                    "terminated": False,
                },
            ]
        }
    )
    task = PrimeAgentPersistenceTask(vf.TaskData(idx=0, prompt=None))
    assert await task.persisted(trace) == 1.0
    reimported_cell = "import secrets\n" + SECOND_CELL
    trace.info["prime_agent_segments"][1]["tool_calls"] = [
        f'{{"code": {reimported_cell!r}}}'
    ]
    assert await task.persisted(trace) == 0.0
