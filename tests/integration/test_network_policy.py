"""A task's network policy holds during execution: a scripted agent probes a blocked host,
a permitted host, and the framework's own route from inside the runtime
(network-policy-v1), across the runtimes with a network policy."""

import pytest
from grid import runtimes_grid


@pytest.mark.parametrize(
    "harness_runtime", runtimes_grid(needs=("network",)), indirect=True
)
async def test_network_policy(run_v1, harness_runtime, tmp_path):
    traces = await run_v1(
        "network-policy-v1",
        harness="network-policy-v1",
        runtime=harness_runtime,
        output_dir=tmp_path,
        num_tasks=2,
    )
    by_name = {trace.task.data.name: trace for trace in traces}
    assert set(by_name) == {"block", "allow"}
    for name, trace in by_name.items():
        assert trace.ok, [error.message for error in trace.errors]
        assert trace.info["network_report"] == {
            "blocked": "denied",
            "allowed": "ok",
            "framework": "ok",
        }, name
