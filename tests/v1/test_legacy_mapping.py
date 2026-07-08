"""Unit-level complement to test_legacy.py's e2e shape tests: the v0 ``RolloutOutput`` ->
v1 ``Trace`` mapping exercised directly on a minimal output dict (no model, no runtime)."""

from verifiers.v1.legacy import rollout_output_to_trace


def test_v0_tool_defs_map_onto_trace():
    # v0 already persists tool defs (`state["tool_defs"]` -> `RolloutOutput.tool_defs`); the
    # bridge carries them onto `Trace.tool_defs` so a bridged run feeds tool-use SFT the same
    # way a native v1 run does. The v0 and v1 Tool shapes are identical, so this is a
    # re-validation; malformed entries are dropped rather than failing the mapping.
    out = {
        "prompt": [{"role": "user", "content": "q"}],
        "reward": 1.0,
        "is_completed": True,
        "tool_defs": [
            {
                "name": "search",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
                "strict": None,
            },
            "not-a-tool",  # malformed entry: dropped, not fatal
        ],
    }
    trace = rollout_output_to_trace(out, task_idx=0)
    assert trace.tool_defs is not None
    (tool,) = trace.tool_defs
    assert tool.name == "search"
    assert tool.parameters["properties"]["q"] == {"type": "string"}

    # absent / empty tool_defs maps to None, matching a native v1 trace with no tools
    assert rollout_output_to_trace({"prompt": []}, task_idx=1).tool_defs is None
