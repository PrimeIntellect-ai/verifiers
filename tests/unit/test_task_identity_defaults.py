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


def _data_part(**data) -> str:
    from verifiers.v1.durable import task_parts
    from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask

    return task_parts(HarborTask(HarborData(idx=0, prompt="p", **data)))["data"]


def test_a_field_added_with_a_default_keeps_the_call_fingerprint():
    """A call's fingerprint hashes the dumped data, nulls included: the shape from before
    `checkpoint` existed has no such key, and a task that sets none hashes as it did."""
    from verifiers.v1.durable import _dump, _hash
    from verifiers.v1.tasksets.harbor.taskset import HarborData

    dumped = _dump(HarborData(idx=0, prompt="p"))
    assert "checkpoint" in dumped  # the key is in the dump, so identity must drop it
    before = {key: value for key, value in dumped.items() if key != "checkpoint"}
    assert _data_part() == _hash(identity_data(before))
    assert task_key(identity_data(dumped)) == task_key(identity_data(before))


def test_a_set_field_changes_the_call_fingerprint():
    assert _data_part(checkpoint="ck-world") != _data_part()
    assert _data_part(checkpoint="ck-a") != _data_part(checkpoint="ck-b")


ALWAYS_HASHED = frozenset(
    {
        # TaskData
        "agent_user", "artifact_max_bytes", "artifacts", "description", "id", "idx",
        "image", "name", "network", "prompt", "resources", "scoring", "skills",
        "system_prompt", "timeout", "workdir",
        # HarborData
        "authors", "category", "collect", "compose_host_image", "criteria",
        "difficulty", "env", "healthcheck", "keywords", "mcp_servers", "tags",
        "task_dir", "tools", "upload_environment", "verifier", "verifier_env",
        "verifier_image", "verifier_user",
    }
)  # fmt: skip
"""Fields task identity has always hashed, so every call fingerprint holds them."""


def test_a_new_task_data_field_is_classified_for_identity():
    """A field added to `TaskData` or `HarborData` changes every call's fingerprint unless
    identity leaves it out at its default: add it to `_IDENTITY_DEFAULTS`, or, for a field
    whose old tasks must change identity, to `ALWAYS_HASHED` knowingly."""
    from verifiers.v1.task import _IDENTITY_DEFAULTS, TaskData
    from verifiers.v1.tasksets.harbor.taskset import HarborData

    fields = set(TaskData.model_fields) | set(HarborData.model_fields)
    unclassified = fields - ALWAYS_HASHED - set(_IDENTITY_DEFAULTS)
    assert not unclassified, f"classify {sorted(unclassified)} for task identity"
