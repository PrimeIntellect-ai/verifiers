from __future__ import annotations

import importlib
import sys
from pathlib import Path

import verifiers.v1 as vf


def load_module(monkeypatch):
    env_dir = Path(__file__).parents[1] / "environments" / "paperbench"
    monkeypatch.syspath_prepend(str(env_dir))
    sys.modules.pop("paperbench", None)
    return importlib.import_module("paperbench")


class Result:
    def __init__(self, exit_code: int):
        self.exit_code = exit_code
        self.stdout = ""
        self.stderr = ""


class RecordingSandbox:
    def __init__(self, results: list[Result]):
        self.results = list(results)
        self.calls = []

    async def execute(self, command: str, **kwargs):
        self.calls.append({"command": command, **kwargs})
        return self.results.pop(0)


def test_paperbench_loads_dev_split(monkeypatch):
    module = load_module(monkeypatch)

    env = module.load_environment(limit=2, image="local-paperbench")
    rows = list(env.taskset.source())

    assert isinstance(env, vf.Env)
    assert isinstance(env.harness, vf.OpenCode)
    assert env.taskset.taskset_id == "paperbench"
    assert [row["task_id"] for row in rows] == module.SPLITS["dev"][:2]
    assert rows[0]["sandbox"]["image"] == "local-paperbench"
    assert rows[0]["sandbox"]["workdir"] == "/home"
    assert rows[0]["program"]["env"]["AGENT_WORKDIR"] == "/home"
    assert "/home/submission" in rows[0]["prompt"][0]["content"]


def test_paperbench_full_mode_mentions_reproduce_script(monkeypatch):
    module = load_module(monkeypatch)

    row = module.make_record(
        "rice",
        "debug",
        "pb-env:latest",
        "/home",
        "/home/paper",
        "/home/submission",
        code_only=False,
    )

    assert "reproduce.sh" in row["prompt"][0]["content"]
    assert row["info"]["code_only"] is False


def test_paperbench_direct_submission_layout(monkeypatch):
    module = load_module(monkeypatch)
    row = module.make_record(
        "rice",
        "debug",
        "pb-env:latest",
        "/home",
        "/home/paper",
        "/home/submission",
        code_only=True,
    )

    assert module.direct_submission_layout(row) == {
        "paper_id": "rice",
        "submission_dir": "/home/submission",
        "code_only": "true",
    }


def test_paperbench_rejects_unknown_split(monkeypatch):
    module = load_module(monkeypatch)

    try:
        module.split_ids("missing")
    except ValueError as exc:
        assert "Unknown PaperBench split" in str(exc)
    else:
        raise AssertionError("expected ValueError")


async def test_paperbench_valid_code_only_layout(monkeypatch):
    module = load_module(monkeypatch)
    taskset = module.load_taskset(paper_ids=["rice"])
    task = vf.Task(list(taskset.source())[0])
    state = vf.State.for_task(task)
    sandbox = RecordingSandbox([Result(0), Result(0), Result(0)])
    state["_paperbench_sandbox"] = sandbox

    assert await module.valid_submission_layout(task, state) == 1.0
    assert [call["command"] for call in sandbox.calls] == [
        "test -d /home/submission",
        "test -d /home/submission/.git",
        "test -s /home/submission/README.md",
    ]


async def test_paperbench_full_layout_requires_reproduce_script(monkeypatch):
    module = load_module(monkeypatch)
    taskset = module.load_taskset(paper_ids=["rice"], code_only=False)
    task = vf.Task(list(taskset.source())[0])
    state = vf.State.for_task(task)
    state["_paperbench_sandbox"] = RecordingSandbox(
        [Result(0), Result(0), Result(0), Result(1)]
    )

    assert await module.valid_submission_layout(task, state) == 0.0
