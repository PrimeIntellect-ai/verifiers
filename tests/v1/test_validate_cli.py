import asyncio
import json
import tomllib
from types import SimpleNamespace

import pytest

from verifiers.v1.cli import validate
from verifiers.v1.cli.validate_output import (
    CONFIG_FILE,
    RESULTS_FILE,
    SUMMARY_FILE,
    append_result,
    identity,
    load_results,
    load_resume_config,
    output_path,
    save_run,
    summarize,
    validation_mode,
)
from verifiers.v1.configs.cli.validate import ValidateConfig


def result_row(
    position: int,
    key: str,
    reason: str,
    *,
    mode: str = "gold",
) -> dict:
    return {
        "task_position": position,
        "task_key": key,
        "index": 100 + position,
        "name": f"task-{position}",
        "mode": mode,
        "valid": reason == "valid",
        "reason": reason,
        "elapsed": 1.25,
        "error": "failed" if reason in {"error", "timeout"} else None,
        "error_type": "TimeoutError" if reason == "timeout" else None,
    }


def test_validate_output_is_fresh_and_replayable(tmp_path):
    data = {
        "taskset": {"id": "alphabet-sort-v1"},
        "only_setup": True,
        "num_tasks": 7,
        "rich": False,
    }
    first = ValidateConfig.model_validate(data)
    second = ValidateConfig.model_validate(data)
    assert output_path(first) != output_path(second)

    run_dir = tmp_path / "run"
    first.output_dir = run_dir
    save_run(first, run_dir, total=7)

    saved = tomllib.loads((run_dir / CONFIG_FILE).read_text())
    assert saved["only_setup"] is True
    assert saved["num_tasks"] == 7
    assert "uuid" not in saved
    assert (run_dir / RESULTS_FILE).read_text() == ""
    assert json.loads((run_dir / SUMMARY_FILE).read_text())["outcomes"]["missing"] == 7

    resumed = load_resume_config(run_dir)
    assert resumed.taskset.id == "alphabet-sort-v1"
    assert resumed.only_setup is True
    assert resumed.num_tasks == 7
    assert resumed.resume == run_dir
    assert resumed.output_dir == run_dir


def test_resume_keeps_valid_and_invalid_but_retries_the_rest(tmp_path):
    selected = [identity(i, {"idx": i, "prompt": f"p{i}"}) for i in range(5)]
    rows = [
        result_row(0, selected[0][1], "valid"),
        result_row(1, selected[1][1], "invalid"),
        result_row(2, selected[2][1], "error"),
        result_row(3, selected[3][1], "timeout"),
        # A duplicate final result must not survive canonicalization.
        result_row(0, selected[0][1], "valid"),
    ]
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / RESULTS_FILE).write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + '{"task_position":4'
    )

    kept, owed = load_results(tmp_path, selected, "gold")

    assert [row["task_position"] for row in kept] == [0, 1]
    assert owed == [2, 3, 4]
    canonical = [
        json.loads(line) for line in (tmp_path / RESULTS_FILE).read_text().splitlines()
    ]
    assert canonical == kept


def test_summary_reports_all_checks_and_resume_debt():
    rows = [
        {
            **result_row(0, "a", "valid", mode="all"),
            "gold": result_row(0, "a", "valid"),
            "setup": result_row(0, "a", "valid", mode="setup"),
        },
        {
            **result_row(1, "b", "error", mode="all"),
            "gold": result_row(1, "b", "invalid"),
            "setup": result_row(1, "b", "error", mode="setup"),
        },
    ]

    summary = summarize(rows, total=3, mode="all")

    assert summary["outcomes"] == {
        "valid": 1,
        "invalid": 0,
        "error": 1,
        "timeout": 0,
        "missing": 1,
    }
    assert summary["terminal"] == 1
    assert summary["owed"] == 2
    assert summary["checks"]["gold"]["invalid"] == 1
    assert summary["checks"]["setup"]["error"] == 1


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (ValidateConfig(only_gold=True), "gold"),
        (ValidateConfig(only_setup=True), "setup"),
    ],
)
def test_only_modes_dispatch_symmetrically(monkeypatch, config, expected):
    calls = []

    async def gold(task, config):
        calls.append("gold")
        return {"mode": "gold"}

    async def setup(task, config):
        calls.append("setup")
        return {"mode": "setup"}

    monkeypatch.setattr(validate, "_run_gold", gold)
    monkeypatch.setattr(validate, "_run_setup", setup)

    row = asyncio.run(validate._validate_task(object(), config))

    assert validation_mode(config) == expected
    assert row["mode"] == expected
    assert calls == [expected]


class FakeTask:
    NEEDS_CONTAINER = False

    def __init__(self, idx: int):
        self.data = SimpleNamespace(
            idx=idx,
            name=f"task-{idx}",
            image=None,
            model_dump=lambda **_: {"idx": idx, "name": f"task-{idx}"},
        )


class FakeTaskset:
    INFINITE = False

    def __init__(self, tasks):
        self._tasks = tasks

    def __iter__(self):
        return iter(self._tasks)

    def head(self, n):
        return FakeTaskset(self._tasks[:n])

    def shuffle(self):
        return self


@pytest.mark.parametrize("mode", ["gold", "setup"])
def test_run_resume_schedules_only_owed_tasks(monkeypatch, tmp_path, mode):
    tasks = [FakeTask(i) for i in range(5)]
    selected = [identity(i, task.data.model_dump()) for i, task in enumerate(tasks)]
    config = ValidateConfig(
        only_gold=mode == "gold",
        only_setup=mode == "setup",
        output_dir=tmp_path,
        rich=False,
    )
    save_run(config, tmp_path, total=len(tasks))
    for position, reason in enumerate(("valid", "invalid", "error", "timeout")):
        append_result(
            tmp_path,
            result_row(position, selected[position][1], reason, mode=mode),
        )
    config.resume = tmp_path
    called = []

    async def run_task(task, config):
        called.append(task.data.idx)
        return result_row(
            task.data.idx,
            selected[task.data.idx][1],
            "valid",
            mode=mode,
        )

    monkeypatch.setattr(validate.vf, "load_taskset", lambda _: FakeTaskset(tasks))
    monkeypatch.setattr(validate, "_validate_task", run_task)

    rows = asyncio.run(validate.run_validate(config))

    assert sorted(called) == [2, 3, 4]
    assert [row["task_position"] for row in rows] == list(range(5))
    persisted = [
        json.loads(line) for line in (tmp_path / RESULTS_FILE).read_text().splitlines()
    ]
    assert len(persisted) == 5
    assert {row["task_position"] for row in persisted} == set(range(5))
    summary = json.loads((tmp_path / SUMMARY_FILE).read_text())
    assert summary["mode"] == mode
    assert summary["owed"] == 0
