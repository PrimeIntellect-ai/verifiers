"""Tests for CLI sampling argument parsing in vf-eval."""

import json
import sys

import pytest

import verifiers.scripts.eval as vf_eval


@pytest.fixture
def capture_sampling_args(monkeypatch):
    """Fixture to capture sampling args passed to eval_environment."""
    captured: dict[str, dict] = {}

    def fake_eval_environment(**kwargs):
        captured["sampling_args"] = kwargs["sampling_args"]

    monkeypatch.setattr(vf_eval, "eval_environment", fake_eval_environment)
    return captured


class TestCLISamplingArgs:
    """Test CLI sampling argument parsing and merging."""

    def test_sampling_args_from_flags(self, monkeypatch, capture_sampling_args):
        """Test that individual CLI flags are correctly parsed into sampling args."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vf-eval",
                "gsm8k",
                "--temperature",
                "0.7",
                "--max-tokens",
                "256",
                "--top-k",
                "40",
                "--min-p",
                "0.02",
            ],
        )

        vf_eval.main()

        sa = capture_sampling_args["sampling_args"]
        assert sa["temperature"] == 0.7
        assert sa["max_tokens"] == 256
        assert "extra_body" in sa
        assert sa["extra_body"] == {"top_k": 40, "min_p": 0.02}
        assert "_EXTRA_BODY_FIELDS" not in sa

    def test_json_overrides_flags(self, monkeypatch, capture_sampling_args):
        """Test that JSON sampling args override individual CLI flags."""
        overrides = json.dumps(
            {
                "temperature": 0.9,
                "extra_body": {"repetition_penalty": 1.1},
                "top_p": 0.8,
            }
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "vf-eval",
                "gsm8k",
                "--temperature",
                "0.5",
                "--top-k",
                "50",
                "--min-p",
                "0.01",
                "-S",
                overrides,
            ],
        )

        vf_eval.main()

        sa = capture_sampling_args["sampling_args"]
        assert sa["temperature"] == 0.9
        assert sa["top_p"] == 0.8
        assert sa["extra_body"]["repetition_penalty"] == 1.1
        assert sa["extra_body"]["top_k"] == 50
        assert sa["extra_body"]["min_p"] == 0.01
