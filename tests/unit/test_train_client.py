import pytest

from verifiers.v1.clients.train import _is_valid_incremental_tail


@pytest.mark.parametrize(
    "roles",
    [
        ["tool"],
        ["tool", "tool"],
        ["user"],
        ["tool", "user"],
        ["user", "user"],
        ["user", "tool"],
        ["tool", "user", "tool", "user"],
    ],
)
def test_incremental_tail_accepts_tool_and_user_messages(roles):
    assert _is_valid_incremental_tail([{"role": role} for role in roles])


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant"}],
        [{"role": "system"}],
        [{"role": "tool"}, {"role": "assistant"}],
        [{"role": "user"}, {"content": "no role"}],
    ],
)
def test_incremental_tail_rejects_other_messages(messages):
    assert not _is_valid_incremental_tail(messages)
