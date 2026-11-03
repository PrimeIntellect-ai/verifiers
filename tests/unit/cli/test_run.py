"""`vf-run` resolves a typed config from the CLI without a model."""

import json

import pytest

from verifiers.v1.cli import run
from verifiers.v1.cli.output import CONFIG_DIR, RESOLVED_DIR
from verifiers.v1.cli.run import USAGE, main


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
