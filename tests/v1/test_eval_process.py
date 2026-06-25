import json
import sys

from verifiers.v1.cli.eval.main import main
from verifiers.v1.cli.eval.resolver import resolve_eval
from verifiers.v1.cli.output import RunInfo, read_run_info


def test_protocol_version(capsys) -> None:
    main(["--protocol-version"])

    assert json.loads(capsys.readouterr().out) == {
        "protocol_version": 1,
        "trace_schema_version": 1,
        "operations": ["run", "resolve"],
    }


def test_resolve_run_handoff_writes_run_info(tmp_path, capsys) -> None:
    output_dir = tmp_path / "chosen-output"
    args = ["gsm8k-v1", "--dry-run", "--output-dir", str(output_dir)]

    main(["resolve", "--format", "json", *args])
    resolved = json.loads(capsys.readouterr().out)
    main(["run", *args, "--uuid", resolved["run_id"]])

    assert read_run_info(output_dir) == RunInfo(run_id=resolved["run_id"])
    assert (output_dir / "config.toml").exists()
    assert not (output_dir / "results.jsonl").exists()


def test_resume_uses_persisted_run_id(tmp_path) -> None:
    output_dir = tmp_path / "not-the-run-id"
    args = [
        "gsm8k-v1",
        "--dry-run",
        "--output-dir",
        str(output_dir),
        "--uuid",
        "fixed-run-id",
    ]
    main(["run", *args])

    invocation = resolve_eval(["--resume", str(output_dir)])

    assert invocation.run_id == "fixed-run-id"
    assert invocation.config.uuid == "fixed-run-id"


def test_main_does_not_mutate_sys_argv(capsys) -> None:
    before = sys.argv.copy()

    main(["--protocol-version"])
    capsys.readouterr()

    assert sys.argv == before
