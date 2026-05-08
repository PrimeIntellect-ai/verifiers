from __future__ import annotations

import pytest

from verifiers.utils.v1_config_aliases import merge_v1_config_aliases


def test_merge_v1_config_aliases_keeps_config_first_class() -> None:
    result = merge_v1_config_aliases(
        taskset={"tasks": "/tmp/tasks"},
        harness={"max_turns": 8, "sampling_args": {"max_tokens": 128}},
        global_harness={"max_turns": 4, "sampling_args": {"temperature": 0.2}},
    )

    assert result == {
        "taskset": {"tasks": "/tmp/tasks"},
        "harness": {
            "max_turns": 8,
            "sampling_args": {"temperature": 0.2, "max_tokens": 128},
        },
    }


def test_merge_v1_config_aliases_rejects_non_table_aliases() -> None:
    with pytest.raises(ValueError, match="harness"):
        merge_v1_config_aliases(harness="not-a-table")
