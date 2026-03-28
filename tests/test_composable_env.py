"""Tests for the composable architecture: Task, TaskSet, ComposableEnv."""

from verifiers.envs.experimental.composable import Task, TaskSet


# ── Mock TaskSet ────────────────────────────────────────────────────────


class MockTaskSet(TaskSet):
    """Minimal TaskSet for testing."""

    needs_sandbox = False

    def _get_prompt(self, info):
        return [{"role": "user", "content": f"Fix bug #{info.get('id', 0)}"}]

    def _get_image(self, info):
        return "python:3.11-slim"

    def _get_workdir(self, info):
        return "/testbed"

    def _get_env_vars(self):
        return {"FOO": "bar"}


def _make_dataset(n=3):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "question": [f"q{i}" for i in range(n)],
            "info": [{"id": i} for i in range(n)],
            "answer": ["" for _ in range(n)],
        }
    )


def _make_taskset(n=3, name="test"):
    return MockTaskSet(dataset=_make_dataset(n), name=name)


# ── Task ────────────────────────────────────────────────────────────────


def test_task_from_taskset():
    ts = _make_taskset()
    task = ts[0]
    assert isinstance(task, Task)
    assert task.info == {"id": 0}


def test_task_prompt():
    ts = _make_taskset()
    task = ts[0]
    assert "0" in task.prompt[0]["content"]


def test_task_image():
    ts = _make_taskset()
    task = ts[0]
    assert task.image == "python:3.11-slim"


def test_task_workdir():
    ts = _make_taskset()
    task = ts[0]
    assert task.workdir == "/testbed"


def test_task_repr():
    ts = _make_taskset()
    task = ts[0]
    assert "test" in repr(task)


# ── TaskSet ─────────────────────────────────────────────────────────────


def test_taskset_len():
    ts = _make_taskset(5)
    assert len(ts) == 5


def test_taskset_get_dataset():
    ds = _make_dataset()
    ts = MockTaskSet(dataset=ds, name="test")
    assert ts.get_dataset() is ds


def test_taskset_needs_sandbox():
    ts = _make_taskset()
    assert ts.needs_sandbox is False


def test_taskset_delegates():
    ts = _make_taskset()
    assert ts.get_image({"id": 1}) == "python:3.11-slim"
    assert ts.get_workdir({"id": 1}) == "/testbed"
    assert ts.get_env_vars() == {"FOO": "bar"}
    assert ts.get_extra_tools() == []


def test_taskset_iter():
    ts = _make_taskset(3)
    tasks = list(ts)
    assert len(tasks) == 3
    assert all(isinstance(t, Task) for t in tasks)


def test_taskset_filter():
    ts = _make_taskset(5)
    filtered = ts.filter(lambda ex: ex["info"]["id"] < 3)
    assert len(filtered) == 3
    assert isinstance(filtered, MockTaskSet)


def test_taskset_take():
    ts = _make_taskset(5)
    taken = ts.take(2)
    assert len(taken) == 2
    assert isinstance(taken, MockTaskSet)


def test_taskset_repr():
    ts = _make_taskset(name="mytest")
    assert "mytest" in repr(ts)
    assert "3" in repr(ts)
