import pytest

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.env import Environment


def test_taskset_required_harness_capability_is_rejected_when_missing() -> None:
    config = EvalConfig.model_validate(
        {
            "taskset": {"id": "capability-required-v1"},
            "harness": {"id": "default"},
        }
    )

    with pytest.raises(ValueError, match="browser-control"):
        Environment(config)


def test_taskset_required_harness_capability_allows_matching_harness() -> None:
    config = EvalConfig.model_validate(
        {
            "taskset": {"id": "capability-required-v1"},
            "harness": {"id": "capability-harness-v1"},
        }
    )

    Environment(config)
