"""The trivial tasks under different harnesses, and the capability gate that rejects an
unsupported harness/task pairing.

`default` and `compact` both drive task MCP tools, so they run a tool task end to end. `rlm`
does not (`SUPPORTS_TASK_TOOLS = False`), so composing it with a tools taskset is refused up
front — asserted here at build time, without running it (rlm installs a heavy agent binary at
rollout, out of scope for a trivial task).
"""

import pytest

from verifiers.v1.env import Environment, EnvConfig


@pytest.fixture(params=["default", "compact"])
def harness(request) -> str:
    return request.param


@pytest.mark.e2e
async def test_single_turn_across_harnesses(run_v1, harness, tmp_path):
    (trace,) = await run_v1(
        "echo-v1", runtime="subprocess", harness=harness, output_dir=tmp_path, max_turns=2,
    )
    assert trace.errors == []
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_tools_across_harnesses(run_v1, harness, tmp_path):
    (trace,) = await run_v1(
        "glossary-v1", runtime="subprocess", harness=harness, output_dir=tmp_path, max_turns=6,
    )
    assert trace.errors == []
    assert trace.reward == 1.0


def test_rlm_rejects_task_tools():
    """rlm can't drive task MCP tools, so pairing it with a tools taskset (multi-turn + tools)
    raises at build time — no rollout, no model."""
    config = EnvConfig.model_validate(
        {"taskset": {"id": "glossary-v1"}, "harness": {"id": "rlm"}}
    )
    with pytest.raises(ValueError, match="task tools"):
        Environment(config)
