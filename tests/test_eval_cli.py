import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from verifiers.v1.cli.eval.main import main
from verifiers.v1.cli.eval.resume import load_resume_config
from verifiers.v1.cli.output import save_config, snapshot_system_prompt
from verifiers.v1.configs.eval import EvalConfig


@pytest.mark.parametrize("path_kind", ["missing", "directory"])
def test_eval_dry_run_rejects_invalid_system_prompt_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_kind: str
):
    prompt_path = tmp_path / "nested" / "prompt.txt"
    if path_kind == "directory":
        prompt_path.mkdir(parents=True)

    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "reverse-text-v1",
            "--system-prompt-path",
            str(prompt_path),
            "--dry-run",
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit, match="does not exist or is not a file"):
        main()

    assert not output_dir.exists()


class PromptConfig(BaseModel):
    system_prompt_path: Path


def test_save_config_snapshots_system_prompt_for_resume(tmp_path: Path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("original prompt")
    output_dir = tmp_path / "output"
    config = PromptConfig(system_prompt_path=prompt_path)

    save_config(config, output_dir)
    prompt_path.write_text("changed prompt")

    snapshot_path = output_dir / "system_prompt.txt"
    assert snapshot_path.read_text() == "original prompt"
    assert config.system_prompt_path == snapshot_path
    assert str(snapshot_path) in (output_dir / "config.toml").read_text()


def test_snapshot_system_prompt_precedes_environment_reads(tmp_path: Path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("original prompt")
    output_dir = tmp_path / "output"
    config = PromptConfig(system_prompt_path=prompt_path)

    snapshot_system_prompt(config, output_dir)
    prompt_path.write_text("changed prompt")

    assert config.system_prompt_path == output_dir / "system_prompt.txt"
    assert config.system_prompt_path.read_text() == "original prompt"



def test_load_resume_config_anchors_prompt_snapshot_to_resume_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    prompt_path = Path("prompt.txt")
    prompt_path.write_text("original prompt")
    output_dir = Path("outputs") / "run"
    save_config(EvalConfig(system_prompt_path=prompt_path), output_dir)

    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    config = load_resume_config((tmp_path / output_dir).resolve())

    snapshot_path = (tmp_path / output_dir / "system_prompt.txt").resolve()
    assert config.system_prompt_path == snapshot_path
    assert config.system_prompt_path.read_text() == "original prompt"
