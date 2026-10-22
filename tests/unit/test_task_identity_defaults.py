"""Scoring fields added after a task's identity was first hashed keep its key at their defaults."""

from verifiers.v1.task import identity_data, task_key


def test_new_scoring_defaults_keep_the_task_key():
    before = {"name": "t", "scoring": {"timeout": 60}}
    after = {
        "name": "t",
        "scoring": {
            "timeout": 60,
            "network": None,
            "skip_on_timeout": False,
            "skip_on_truncated": False,
        },
    }
    assert identity_data(after) == identity_data(before)
    assert task_key(identity_data(after)) == task_key(identity_data(before))


def test_a_set_scoring_field_changes_the_task_key():
    before = {"name": "t", "scoring": {"timeout": 60}}
    after = {"name": "t", "scoring": {"timeout": 60, "skip_on_timeout": True}}
    assert identity_data(after) != identity_data(before)
