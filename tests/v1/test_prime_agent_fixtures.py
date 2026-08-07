"""Deterministic guard tests for the Prime Agent live capability fixtures."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from prime_agent_failed_turn_v1 import has_raised_provider_failure
from prime_agent_ipython_cell_v1 import CELL, SENTINEL, has_ipython_cell_call
from prime_agent_persistence_v1 import (
    FIRST_CELL,
    SECOND_CELL,
    PrimeAgentPersistenceTask,
)

import verifiers.v1 as vf


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


def test_ipython_cell_guard_requires_verbatim_code_and_real_execution():
    """A fabricated call plus the right reply must not score.

    The reward needs both halves: the cell submitted verbatim AND the sentinel in
    a tool RESULT, which only a real kernel execution produces.
    """

    def trace_with(code: str, outputs: list[str]) -> SimpleNamespace:
        return SimpleNamespace(
            info={
                "prime_agent_segments": [
                    {
                        "last_reply": "DONE",
                        "terminated": False,
                        "tool_calls": [
                            {"name": "ipython", "arguments": json.dumps({"code": code})}
                        ],
                        "tool_outputs": outputs,
                    }
                ]
            }
        )

    assert has_ipython_cell_call(trace_with(CELL, [SENTINEL]))
    # Claimed but never executed: no tool result carries the sentinel.
    assert not has_ipython_cell_call(trace_with(CELL, []))
    assert not has_ipython_cell_call(trace_with(CELL, ["something else"]))
    # Executed something else, even if it printed the sentinel itself.
    assert not has_ipython_cell_call(trace_with(f"{CELL}\nprint('extra')", [SENTINEL]))


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _PersistenceRuntime:
    def __init__(self, existing_paths: set[str]):
        self.existing_paths = existing_paths
        self.calls: list[tuple[list[str], dict]] = []

    async def run(self, command: list[str], environment: dict):
        self.calls.append((command, environment))
        return SimpleNamespace(exit_code=int(command[-1] in self.existing_paths))


class _PersistenceAgent:
    def __init__(self, runtime, interaction):
        self.runtime = runtime
        self._interaction = interaction

    def provision(self, task):
        return _AsyncContext(self.runtime)

    def interaction(self, task, *, runtime):
        assert runtime is self.runtime
        return _AsyncContext(self._interaction)


class _PersistenceInteraction:
    def __init__(self, trace):
        self.trace = trace
        self._segments = [
            SimpleNamespace(last_reply="READY", messages=[], terminated=False),
            SimpleNamespace(last_reply="marker", messages=[], terminated=False),
        ]

    async def turn(self, prompt):
        return self._segments.pop(0)


def _prime_agent_trace_root(trace_id: str) -> str:
    return (
        "/tmp/vf-prime-agent-state/"
        f"{hashlib.sha256(trace_id.encode()).hexdigest()[:32]}"
    )


@pytest.mark.asyncio
async def test_persistence_fixture_checks_the_harness_trace_root_after_cleanup():
    from prime_agent_persistence_v1 import PrimeAgentPersistenceEnv

    trace = SimpleNamespace(id="trace/with untrusted input", info={})
    runtime = _PersistenceRuntime(existing_paths=set())
    interaction = _PersistenceInteraction(trace)
    agents = SimpleNamespace(agent=_PersistenceAgent(runtime, interaction))

    await PrimeAgentPersistenceEnv.run(
        object.__new__(PrimeAgentPersistenceEnv), None, agents
    )

    assert runtime.calls == [
        (["test", "!", "-e", _prime_agent_trace_root(trace.id)], {})
    ]
    assert trace.info["prime_agent_state_cleaned"] is True


@pytest.mark.asyncio
async def test_persistence_fixture_rejects_an_uncleaned_harness_trace_root():
    from prime_agent_persistence_v1 import PrimeAgentPersistenceEnv

    trace = SimpleNamespace(id="trace-that-remains", info={})
    root = _prime_agent_trace_root(trace.id)
    runtime = _PersistenceRuntime(existing_paths={root})
    interaction = _PersistenceInteraction(trace)
    agents = SimpleNamespace(agent=_PersistenceAgent(runtime, interaction))

    await PrimeAgentPersistenceEnv.run(
        object.__new__(PrimeAgentPersistenceEnv), None, agents
    )

    assert runtime.calls == [(["test", "!", "-e", root], {})]
    assert trace.info["prime_agent_state_cleaned"] is False
