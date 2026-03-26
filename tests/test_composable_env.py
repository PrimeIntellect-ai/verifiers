"""Tests for the composable architecture: Task, TaskSet, ComposableEnv."""

from unittest.mock import AsyncMock

import pytest

from verifiers.envs.experimental.task import TaskSpec, TaskSet


# ── Fixtures ────────────────────────────────────────────────────────────


class MockTask:
    """Minimal Task implementation for testing."""

    needs_sandbox = False

    def get_dataset(self):
        return [{"question": "fix the bug", "info": {"id": 1}, "answer": ""}]

    def get_prompt(self, info):
        return [{"role": "user", "content": f"Fix bug #{info.get('id', 0)}"}]

    def get_image(self, info):
        return "python:3.11-slim"

    def get_workdir(self, info):
        return "/testbed"

    def get_env_vars(self):
        return {"FOO": "bar"}

    async def setup(self, sandbox_client, sandbox_id, state):
        state["task_setup_done"] = True

    async def evaluate(self, sandbox_client, sandbox_id, state):
        return 1.0

    def get_extra_tools(self):
        return []

    async def apply_gold_patch(self, sandbox_client, sandbox_id, state):
        pass


# ── Task Protocol ───────────────────────────────────────────────────────


def test_mock_task_implements_protocol():
    task = MockTask()
    assert isinstance(task, TaskSpec)


def test_task_get_prompt():
    task = MockTask()
    prompt = task.get_prompt({"id": 42})
    assert len(prompt) == 1
    assert "42" in prompt[0]["content"]


def test_task_needs_sandbox():
    task = MockTask()
    assert task.needs_sandbox is False


# ── TaskSet ─────────────────────────────────────────────────────────────


def _make_dataset(n=3):
    from datasets import Dataset
    return Dataset.from_dict({
        "question": [f"q{i}" for i in range(n)],
        "info": [{"id": i} for i in range(n)],
        "answer": ["" for _ in range(n)],
    })


def test_taskset_len():
    ds = _make_dataset(5)
    ts = TaskSet(spec=MockTask(), dataset=ds, name="test")
    assert len(ts) == 5


def test_taskset_get_dataset():
    ds = _make_dataset()
    ts = TaskSet(spec=MockTask(), dataset=ds, name="test")
    assert ts.get_dataset() is ds


def test_taskset_spec():
    spec = MockTask()
    ts = TaskSet(spec=spec, dataset=_make_dataset(), name="test")
    assert ts.spec is spec


def test_taskset_delegates_to_spec():
    ts = TaskSet(spec=MockTask(), dataset=_make_dataset(), name="test")
    assert ts.get_image({"id": 1}) == "python:3.11-slim"
    assert ts.get_workdir({"id": 1}) == "/testbed"
    assert ts.get_env_vars() == {"FOO": "bar"}
    assert ts.get_extra_tools() == []
    assert ts.needs_sandbox is False


def test_taskset_filter():
    ds = _make_dataset(5)
    ts = TaskSet(spec=MockTask(), dataset=ds, name="test")
    filtered = ts.filter(lambda ex: ex["info"]["id"] < 3)
    assert len(filtered) == 3


def test_taskset_take():
    ds = _make_dataset(5)
    ts = TaskSet(spec=MockTask(), dataset=ds, name="test")
    taken = ts.take(2)
    assert len(taken) == 2


def test_taskset_repr():
    ts = TaskSet(spec=MockTask(), dataset=_make_dataset(), name="mytest")
    assert "mytest" in repr(ts)
    assert "3" in repr(ts)
