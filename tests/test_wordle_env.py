from pathlib import Path

import pytest


def test_wordle_load_environment_reads_system_prompt_path(tmp_path, monkeypatch):
    pytest.importorskip("nltk")
    pytest.importorskip("textarena")
    from environments.wordle import wordle

    class CapturingTextArenaEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    prompt = "Optimized Wordle prompt.\n\nPreserve exact text.\n"
    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    monkeypatch.setattr(wordle, "TextArenaEnv", CapturingTextArenaEnv)

    env = wordle.load_environment(path_to_system_prompt=Path(prompt_path))

    assert env.kwargs["system_prompt"] == prompt
