from __future__ import annotations

import pytest

from verifiers.utils.v1_config_aliases import normalize_v1_config_aliases


def test_normalize_v1_config_aliases_supports_rl_args_key() -> None:
    result = normalize_v1_config_aliases(
        {
            "id": "env1",
            "taskset": {"tasks": "/tmp/tasks"},
            "harness": {"max_turns": 8},
            "args": {"config": {"harness": {"sampling_args": {"max_tokens": 128}}}},
        },
        args_key="args",
        global_harness={"max_turns": 4, "sampling_args": {"temperature": 0.2}},
    )

    assert result == {
        "id": "env1",
        "args": {
            "config": {
                "taskset": {"tasks": "/tmp/tasks"},
                "harness": {
                    "max_turns": 8,
                    "sampling_args": {"temperature": 0.2, "max_tokens": 128},
                },
            }
        },
    }


def test_normalize_v1_config_aliases_rejects_non_table_aliases() -> None:
    with pytest.raises(ValueError, match="harness"):
        normalize_v1_config_aliases(
            {"id": "env1", "harness": "not-a-table"},
            args_key="args",
        )
