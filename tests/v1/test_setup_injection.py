"""`Taskset.setup` declares its inputs by name and the framework injects them.

Mirrors the dispatch the rollout / validate entrypoints use (`invoke(taskset.setup, ...)`):
an override may take any subset of `task`, `trace`, `runtime` — the legacy `(task, runtime)`
signature and a `trace`-taking signature both work, and the trace already exists at setup time.
"""

import verifiers.v1 as vf
from verifiers.v1.decorators import invoke
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace


def _available(taskset: vf.Taskset) -> dict:
    task = Task(idx=0, prompt=None)
    trace = Trace(task=task, state=state_cls(type(taskset))())
    return {"task": task, "trace": trace, "runtime": object()}


async def test_legacy_task_runtime_signature():
    class T(vf.Taskset):
        async def setup(self, task, runtime):
            self.seen = {"task": task, "runtime": runtime}

    ts = T(vf.TasksetConfig())
    available = _available(ts)
    await invoke(ts.setup, available)
    assert ts.seen["task"] is available["task"]
    assert ts.seen["runtime"] is available["runtime"]


async def test_trace_is_injected():
    class T(vf.Taskset):
        async def setup(self, task, trace, runtime):
            trace.info["setup_ran"] = True
            self.seen = {"trace": trace}

    ts = T(vf.TasksetConfig())
    available = _available(ts)
    await invoke(ts.setup, available)
    assert ts.seen["trace"] is available["trace"]
    assert available["trace"].info["setup_ran"] is True


async def test_partial_signature_only_gets_declared_params():
    class T(vf.Taskset):
        async def setup(self, trace):
            self.seen = {"trace": trace}

    ts = T(vf.TasksetConfig())
    available = _available(ts)
    await invoke(ts.setup, available)
    assert ts.seen == {"trace": available["trace"]}


async def test_base_setup_is_noop_under_injection():
    ts = vf.Taskset(vf.TasksetConfig())
    # The default no-op accepts injection without error.
    assert await invoke(ts.setup, _available(ts)) is None
