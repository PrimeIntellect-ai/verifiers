from __future__ import annotations

import pytest

from verifiers.utils.env_config_utils import normalize_env_config_sections


def test_normalize_env_config_sections_keeps_config_first_class() -> None:
    result = normalize_env_config_sections(
        {
            "taskset": {"tasks": "/tmp/tasks"},
            "harness": {"max_turns": 8, "sampling_args": {"max_tokens": 128}},
        },
        global_harness={"max_turns": 4, "sampling_args": {"temperature": 0.2}},
    )

    assert result == {
        "taskset": {"tasks": "/tmp/tasks"},
        "harness": {
            "max_turns": 8,
            "sampling_args": {"temperature": 0.2, "max_tokens": 128},
        },
    }


def test_normalize_env_config_sections_rejects_non_table_aliases() -> None:
    with pytest.raises(ValueError, match="harness"):
        normalize_env_config_sections({"harness": "not-a-table"})
