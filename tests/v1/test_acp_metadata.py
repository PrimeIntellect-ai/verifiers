from types import SimpleNamespace

import verifiers.v1 as vf
from verifiers.v1.acp import ACP_RESPONSE_METADATA_KEY, ACPHarnessSession


def test_acp_host_keeps_arbitrary_response_metadata_runtime_only():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=7, prompt="solve it"),
        ),
    )
    receiver = SimpleNamespace(trace=trace)

    ACPHarnessSession._record_metadata(
        receiver,
        {"metadata": {"prompt": {"agent.private/example": {"token": "secret"}}}},
    )

    assert trace._harness_metadata[ACP_RESPONSE_METADATA_KEY]["prompt"] == {
        "agent.private/example": {"token": "secret"}
    }
    assert "secret" not in trace.model_dump_json()
