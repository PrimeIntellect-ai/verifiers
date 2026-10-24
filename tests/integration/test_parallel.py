"""Overlapping requests on one trace: a harness sends `n` chat programs at a scripted
upstream at once, and the trace graph keeps one branch per copy that shares exactly the
prefix the copies have in common."""

import pytest
from parallel import ParallelHarnessConfig

N = 4


@pytest.mark.parametrize("harness_runtime", ["subprocess"], indirect=True)
@pytest.mark.parametrize("same_prompt", [True, False], ids=["same", "distinct"])
async def test_parallel_requests(run, harness_runtime, scripted_model, same_prompt):
    (episode,) = await run(
        "parallel",
        harness=ParallelHarnessConfig(id="parallel", n=N, same_prompt=same_prompt),
        runtime=harness_runtime,
        client=scripted_model,
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns == trace.num_branches == len(trace.calls) == N
    assert len({call.node for call in trace.calls}) == N

    roles = [node.message.role for node in trace.nodes]
    assert roles.count("system") == 1
    users = [i for i, role in enumerate(roles) if role == "user"]
    replies = [node for node in trace.nodes if node.message.role == "assistant"]
    # The same prompt shares the system and user nodes; distinct prompts share only
    # the system node.
    assert len(users) == (1 if same_prompt else N)
    assert len(trace.nodes) == 1 + len(users) + N
    assert {trace.nodes[i].parent for i in users} == {0}
    assert sorted(reply.parent for reply in replies) == sorted(
        users * N if same_prompt else users
    )
