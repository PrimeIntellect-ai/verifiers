"""Tests for the composable architecture: Task, TaskSet, SandboxTaskSet, SandboxSpec."""

import pytest

import verifiers as vf
from verifiers.envs.experimental.composable import (
    ComposableEnv,
    Harness,
    SandboxSpec,
    SandboxTaskSet,
    Task,
    TaskSet,
)


# ── Mock Rubrics ──────────────────────────────────────────────────────


class MockSandboxRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_reward_func(self.solved)

    async def solved(self, state, **kwargs) -> float:
        return 1.0 if state.get("test_output") == "PASS" else 0.0


class MockMathRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_reward_func(self.correct)

    async def correct(self, state, **kwargs) -> float:
        return 1.0 if state.get("info", {}).get("id") == 0 else 0.0


# ── Mock TaskSets ───────────────────────────────────────────────────────


class MockSandboxTaskSet(SandboxTaskSet):
    """SandboxTaskSet for testing."""

    def get_instruction(self, info):
        return f"Fix bug #{info.get('id', 0)}"

    def get_sandbox_spec(self, info):
        return SandboxSpec(image="python:3.11-slim", cpu_cores=2, memory_gb=2)

    def get_rubric(self):
        return MockSandboxRubric()

    def get_workdir(self, info):
        return "/testbed"

    def get_env_vars(self):
        return {"FOO": "bar"}


class MockTaskSet(TaskSet):
    """Plain TaskSet (no sandbox) for testing."""

    def get_instruction(self, info):
        return info.get("question", "")

    def get_rubric(self):
        return MockMathRubric()


def _make_dataset(n=3):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "info": [{"id": i, "question": f"q{i}"} for i in range(n)],
            "answer": ["" for _ in range(n)],
        }
    )


# ── SandboxSpec ─────────────────────────────────────────────────────────


def test_sandbox_spec_defaults():
    spec = SandboxSpec()
    assert spec.image == "python:3.11-slim"
    assert spec.cpu_cores == 4


def test_sandbox_spec_custom():
    spec = SandboxSpec(image="lean-tactic:v4.27", gpu_count=1)
    assert spec.image == "lean-tactic:v4.27"
    assert spec.gpu_count == 1


# ── Task from SandboxTaskSet ───────────────────────────────────────────


def test_task_sandbox_spec():
    ts = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    task = ts[0]
    assert isinstance(task, Task)
    assert task.sandbox_spec is not None
    assert task.sandbox_spec.image == "python:3.11-slim"
    assert task.sandbox_spec.cpu_cores == 2


def test_task_image():
    ts = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    task = ts[0]
    assert task.image == "python:3.11-slim"


def test_task_workdir():
    ts = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    task = ts[0]
    assert task.workdir == "/testbed"


def test_task_repr_sandbox():
    ts = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    task = ts[0]
    assert "python:3.11-slim" in repr(task)


# ── Task from plain TaskSet ────────────────────────────────────────────


def test_task_no_sandbox():
    ts = MockTaskSet(dataset=_make_dataset(), name="math")
    task = ts[0]
    assert task.sandbox_spec is None
    assert task.image is None


def test_task_repr_no_sandbox():
    ts = MockTaskSet(dataset=_make_dataset(), name="math")
    task = ts[0]
    assert "no sandbox" in repr(task)


# ── TaskSet ─────────────────────────────────────────────────────────────


def test_taskset_isinstance():
    ts = MockTaskSet(dataset=_make_dataset(), name="math")
    assert not isinstance(ts, SandboxTaskSet)

    ts2 = MockSandboxTaskSet(dataset=_make_dataset(), name="swe")
    assert isinstance(ts2, SandboxTaskSet)


def test_taskset_len():
    ts = MockTaskSet(dataset=_make_dataset(5), name="test")
    assert len(ts) == 5


def test_taskset_iter():
    ts = MockTaskSet(dataset=_make_dataset(3), name="test")
    tasks = list(ts)
    assert len(tasks) == 3
    assert all(isinstance(t, Task) for t in tasks)


def test_taskset_filter():
    ts = MockSandboxTaskSet(dataset=_make_dataset(5), name="test")
    filtered = ts.filter(lambda ex: ex["info"]["id"] < 3)
    assert len(filtered) == 3
    assert isinstance(filtered, MockSandboxTaskSet)


def test_taskset_take():
    ts = MockSandboxTaskSet(dataset=_make_dataset(5), name="test")
    taken = ts.take(2)
    assert len(taken) == 2
    assert isinstance(taken, MockSandboxTaskSet)


def test_taskset_repr():
    ts = MockTaskSet(dataset=_make_dataset(), name="mytest")
    assert "mytest" in repr(ts)
    assert "3" in repr(ts)


@pytest.mark.asyncio
async def test_composable_env_exports_task_workdir():
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(run_command="true"),
    )

    env_vars = await env.build_env_vars(
        {
            "info": {"id": 0},
            "interception_base_url": "https://test.trycloudflare.com/v1",
        }
    )

    assert env_vars["AGENT_WORKDIR"] == "/testbed"
    assert env_vars["FOO"] == "bar"
