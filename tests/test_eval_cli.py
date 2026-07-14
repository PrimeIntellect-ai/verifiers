import sys
from pathlib import Path

import pytest

from verifiers.v1.cli.eval.main import main


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
