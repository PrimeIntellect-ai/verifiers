from types import SimpleNamespace

import verifiers.v1 as vf
from verifiers.v1.acp import ACPHarnessSession


def test_acp_host_keeps_ordered_response_artifacts_runtime_only():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=7, prompt="solve it"),
        ),
    )
    receiver = SimpleNamespace(trace=trace, _metadata_bytes=0)

    ACPHarnessSession._record_artifacts(
        receiver,
        {
            "artifacts": [
                {
                    "operation": "prompt",
                    "metadata": {"agent.private/example": {"token": "first"}},
                },
                {
                    "operation": "prompt",
                    "metadata": {"agent.private/example": {"token": "second"}},
                },
            ]
        },
    )

    artifacts = trace.get_harness_artifacts(protocol="acp", operation="prompt")
    assert [artifact.metadata for artifact in artifacts] == [
        {"agent.private/example": {"token": "first"}},
        {"agent.private/example": {"token": "second"}},
    ]
    assert "first" not in trace.model_dump_json()
    assert "harness_artifacts" not in trace.model_dump()
