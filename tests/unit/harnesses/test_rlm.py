"""The rlm harness maps its session's stop reason onto the trace's stop condition."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.harnesses.rlm.harness import (
    RLM_SESSION_METADATA_KEY,
    RLMHarness,
    RLMHarnessConfig,
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
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
    )
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
