"""The rlm harness's runtime contract and stop-reason mapping."""

from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.runtime import DockerConfig, NetworkPolicyConfig
from verifiers.v1.harnesses.rlm.harness import (
    RLM_RUNTIME_METADATA_KEY,
    RLM_SESSION_METADATA_KEY,
    RLMHarness,
    RLMHarnessConfig,
)


def _trace() -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
    )


@pytest.mark.parametrize(
    "reason,condition",
    [
        ("max_total_tokens", "max_total_tokens"),
        ("max_total_turns", "max_turns"),
        ("token_budget", "max_output_tokens"),
        ("compaction_failed", "compaction_failed"),
        ("done", "agent_completed"),
        (None, "agent_completed"),
    ],
)
def test_rlm_stop_reason_survives_completion_and_serialization(reason, condition):
    trace = _trace()
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))
    metadata = {
        RLM_SESSION_METADATA_KEY: {
            "session_id": trace.id,
            "metrics": {"turns": 1},
            "last_stop_reason": reason,
        }
    }
    harness.acp_turn_result(
        trace, vf.ACPTurn(reply="partial", response_metadata=metadata)
    )
    trace.stop("agent_completed")
    harness.acp_close_result(trace, metadata)

    restored = vf.WireTrace.model_validate_json(trace.model_dump_json())
    assert restored.stop_condition == condition
    assert restored.is_truncated is (condition != "agent_completed")


def _sent_policy(config: RLMHarnessConfig, runtime_config) -> dict:
    runtime = SimpleNamespace(config=runtime_config, env={})
    metadata = RLMHarness(config)._runtime_metadata(
        SimpleNamespace(model="m"), _trace(), runtime, "http://endpoint", "s", None
    )
    return metadata[RLM_RUNTIME_METADATA_KEY]["policy"]


@pytest.mark.parametrize(
    "runtime_config",
    [
        DockerConfig(allow=[]),
        DockerConfig().with_task_network_policy(NetworkPolicyConfig(allow=[])),
    ],
)
def test_rlm_drops_network_guidance_for_boxes_with_no_network(runtime_config):
    policy = _sent_policy(RLMHarnessConfig(id="rlm"), runtime_config)
    assert policy["network_guidance"] is False


@pytest.mark.parametrize(
    "runtime_config",
    [DockerConfig(), DockerConfig(allow=["pypi.org"]), SimpleNamespace(type="local")],
)
def test_rlm_leaves_network_guidance_to_nano_rlm_when_network_reachable(
    runtime_config,
):
    policy = _sent_policy(RLMHarnessConfig(id="rlm"), runtime_config)
    assert "network_guidance" not in policy


@pytest.mark.parametrize("value", [True, False])
def test_rlm_explicit_network_guidance_overrides_runtime_policy(value):
    config = RLMHarnessConfig(id="rlm", network_guidance=value)
    for runtime_config in (DockerConfig(allow=[]), DockerConfig()):
        assert _sent_policy(config, runtime_config)["network_guidance"] is value
