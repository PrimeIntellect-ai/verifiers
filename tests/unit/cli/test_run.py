"""`vf-run` resolves a typed config from the CLI without a model."""

import asyncio
import contextlib
import json

import pytest

from verifiers.v1.cli import run
from verifiers.v1.cli.output import CONFIG_DIR, RESOLVED_DIR
from verifiers.v1.cli.run import USAGE, main
from verifiers.v1.configs.cli.run import RunnerConfig


@pytest.fixture(autouse=True)
def keep_stdlib_logging(monkeypatch):
    """`main` routes logging through loguru for the whole process; keep other tests'
    `caplog` working."""
    monkeypatch.setattr(run, "setup_logging", lambda *args, **kwargs: None)


def test_dry_run_writes_config(tmp_path):
    main(
        [
            "reverse-text",
            "-n",
            "2",
            "-r",
            "3",
            "-o",
            str(tmp_path),
            "--run.dir",
            "run",
            "--dry-run",
        ]
    )
    run_dir = tmp_path / "run"
    config = json.loads((run_dir / CONFIG_DIR / RESOLVED_DIR / "run.json").read_text())
    assert config["env"]["taskset"]["id"] == "reverse-text"
    assert config["select"]["limit"] == 2
    assert config["num_rollouts"] == 3
    assert not (run_dir / "traces.jsonl").exists()


def test_refuses_a_run_dir_with_results(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "traces.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="already contains results"):
        main(["reverse-text", "-o", str(tmp_path), "--run.dir", "run", "--dry-run"])


def test_needs_a_taskset():
    with pytest.raises(SystemExit, match="usage") as exc:
        main(["-n", "1"])
    assert str(exc.value) == USAGE


async def test_failed_rollout_cancels_siblings_before_teardown(tmp_path, monkeypatch):
    events: list[str] = []

    class Taskset(list):
        bounded = True

        def select(self, _select):
            return self

    class FakeEnv:
        taskset = Taskset(["fail", "slow"])

        def slots(self, task, n):
            return [task] * n

        async def run_slot(self, slot, ctx, semaphore, on_complete):
            if slot == "fail":
                await asyncio.sleep(0)
                raise RuntimeError("boom")
            try:
                await asyncio.sleep(60)
            finally:
                events.append("sibling unwound")

        @contextlib.asynccontextmanager
        async def serving(self):
            try:
                yield
            finally:
                events.append("teardown")

    monkeypatch.setattr(run, "load_environment", lambda env: FakeEnv())
    config = RunnerConfig(env={"taskset": {"id": "reverse-text"}}, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        await run.run_episodes(config)
    assert events == ["sibling unwound", "teardown"]
