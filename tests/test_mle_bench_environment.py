from __future__ import annotations

import importlib
import sys
from pathlib import Path

import verifiers.v1 as vf


def load_module(monkeypatch):
    env_dir = Path(__file__).parents[1] / "environments" / "mle_bench"
    monkeypatch.syspath_prepend(str(env_dir))
    sys.modules.pop("mle_bench", None)
    return importlib.import_module("mle_bench")


class Result:
    def __init__(self, exit_code: int, stdout: str = "", stderr: str = ""):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class RecordingSandbox:
    def __init__(self, results: list[Result]):
        self.results = list(results)
        self.calls = []

    async def execute(self, command: str, **kwargs):
        self.calls.append({"command": command, **kwargs})
        return self.results.pop(0)


def test_mle_bench_loads_low_split_without_registry(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module, "load_registry_competition", lambda _id: None)

    env = module.load_environment(limit=2, image="local-mlebench")
    rows = list(env.taskset.source())

    assert isinstance(env, vf.Env)
    assert isinstance(env.harness, vf.OpenCode)
    assert env.taskset.taskset_id == "mle-bench"
    assert [row["task_id"] for row in rows] == module.LOW_COMPETITIONS[:2]
    assert rows[0]["sandbox"]["image"] == "local-mlebench"
    assert rows[0]["sandbox"]["workdir"] == "/home"
    assert rows[0]["program"]["env"]["AGENT_WORKDIR"] == "/home"
    assert "/home/submission/submission.csv" in rows[0]["prompt"][0]["content"]


def test_mle_bench_prompt_uses_configured_paths(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module, "load_registry_competition", lambda _id: None)

    row = module.make_record(
        "custom-competition",
        "dev",
        "mlebench-env",
        "/workspace",
        "/workspace/out/final.csv",
        "/workspace/tools/check.sh",
    )
    prompt = row["prompt"][0]["content"]

    assert "/workspace/data/description.md" in prompt
    assert "/workspace/data" in prompt
    assert "/workspace/out/final.csv" in prompt
    assert "/workspace/tools/check.sh /workspace/out/final.csv" in prompt
    assert "/home/submission/submission.csv" not in prompt
    assert "/home/validate_submission.sh" not in prompt


def test_mle_bench_uses_registry_metadata_when_available(monkeypatch):
    module = load_module(monkeypatch)

    def fake_registry(competition_id):
        return {
            "id": competition_id,
            "name": "Example Competition",
            "description": "Predict the label.",
            "competition_type": "simple",
            "sample_submission": "/data/sample_submission.csv",
            "answers": "/data/test.csv",
        }

    monkeypatch.setattr(module, "load_registry_competition", fake_registry)
    row = module.make_record(
        "example-competition",
        "dev",
        "mlebench-env",
        "/home",
        "/home/submission/submission.csv",
        "/home/validate_submission.sh",
    )

    assert row["info"]["competition_type"] == "simple"
    assert row["info"]["sample_submission"] == "/data/sample_submission.csv"
    assert "Predict the label." in row["prompt"][0]["content"]


def test_mle_bench_grading_submission_row(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module, "load_registry_competition", lambda _id: None)
    row = module.make_record(
        "spaceship-titanic",
        "dev",
        "mlebench-env",
        "/home",
        "/home/submission/submission.csv",
        "/home/validate_submission.sh",
    )

    assert module.grading_submission_row(row) == {
        "competition_id": "spaceship-titanic",
        "submission_path": "/home/submission/submission.csv",
    }
    assert module.grading_submission_jsonl(row) == (
        '{"competition_id": "spaceship-titanic", '
        '"submission_path": "/home/submission/submission.csv"}\n'
    )


def test_mle_bench_split_ids_reject_unknown_split(monkeypatch):
    module = load_module(monkeypatch)

    try:
        module.split_ids("missing")
    except ValueError as exc:
        assert "Unknown MLE-Bench split" in str(exc)
    else:
        raise AssertionError("expected ValueError")


async def test_mle_bench_valid_submission_records_validator_output(monkeypatch):
    module = load_module(monkeypatch)
    taskset = module.load_taskset(competition_ids=["spaceship-titanic"])
    task = vf.Task(list(taskset.source())[0])
    state = vf.State.for_task(task)
    sandbox = RecordingSandbox([Result(0, "Submission is valid.")])
    state["_mle_bench_sandbox"] = sandbox

    reward = await module.valid_submission(task, state)

    assert reward == 1.0
    assert state["validation_exit_code"] == 0
    assert state["validation_stdout"] == "Submission is valid."
    assert sandbox.calls[0]["command"] == (
        "/home/validate_submission.sh /home/submission/submission.csv"
    )
    assert sandbox.calls[0]["working_dir"] == "/home"


async def test_mle_bench_invalid_submission_gets_zero_reward(monkeypatch):
    module = load_module(monkeypatch)
    taskset = module.load_taskset(competition_ids=["spaceship-titanic"])
    task = vf.Task(list(taskset.source())[0])
    state = vf.State.for_task(task)
    state["_mle_bench_sandbox"] = RecordingSandbox([Result(1, "bad csv")])

    assert await module.valid_submission(task, state) == 0.0
    assert state["validation_exit_code"] == 1


async def test_mle_bench_invalid_stdout_does_not_match_valid(monkeypatch):
    module = load_module(monkeypatch)
    taskset = module.load_taskset(competition_ids=["spaceship-titanic"])
    task = vf.Task(list(taskset.source())[0])
    state = vf.State.for_task(task)
    state["_mle_bench_sandbox"] = RecordingSandbox(
        [Result(0, "Submission invalid! bad csv")]
    )

    assert await module.valid_submission(task, state) == 0.0


def test_mle_bench_validator_accepts_exact_success_line(monkeypatch):
    module = load_module(monkeypatch)

    assert module.validator_accepts("Submission is valid.\n")
    assert not module.validator_accepts("Submission invalid! bad csv")
    assert not module.validator_accepts("not valid")


async def test_mle_bench_submission_exists_metric(monkeypatch):
    module = load_module(monkeypatch)
    taskset = module.load_taskset(competition_ids=["spaceship-titanic"])
    task = vf.Task(list(taskset.source())[0])
    state = vf.State.for_task(task)
    sandbox = RecordingSandbox([Result(0)])
    state["_mle_bench_sandbox"] = sandbox

    assert await module.submission_exists(task, state) == 1.0
    assert sandbox.calls[0]["command"] == "test -f /home/submission/submission.csv"


async def test_mle_bench_sandbox_metrics(monkeypatch):
    module = load_module(monkeypatch)
    taskset = module.load_taskset(competition_ids=["spaceship-titanic"])
    task = vf.Task(list(taskset.source())[0])
    state = vf.State.for_task(task)
    sandbox = RecordingSandbox([Result(0), Result(1)])
    state["_mle_bench_sandbox"] = sandbox

    assert await module.submission_nonempty(task, state) == 1.0
    assert await module.validator_available(task, state) == 0.0
    assert sandbox.calls[0]["command"] == "test -s /home/submission/submission.csv"
    assert sandbox.calls[1]["command"] == "test -x /home/validate_submission.sh"
