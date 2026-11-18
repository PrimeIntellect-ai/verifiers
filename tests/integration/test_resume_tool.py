"""One task, two segments: the harness preserves context and MCP access across a resumed
exchange (echo-acp-resume), across the harness x runtime grid."""

import re

import pytest
from grid import assert_one_branch, grid
from pydantic_core import to_json

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.errors import ProviderError


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("resume", "mcp")), indirect=True
)
async def test_resume_tool(run, harness, harness_runtime, monkeypatch):
    """Each harness preserves context and MCP access across two segments."""
    if harness.id == "codex":
        send = EvalClient._request

        async def inspect_rejection(self, url, body, headers, **kwargs):
            try:
                return await send(self, url, body, headers, **kwargs)
            except Exception as error:
                match = re.search(r"InputItemWire at line 1 column (\d+)", str(error))
                if match and isinstance(body.get("input"), list):
                    wire = to_json(body, inf_nan_mode="null")
                    prefix = b'"input":['
                    offset = wire.index(prefix) + len(prefix)
                    for index, item in enumerate(body["input"]):
                        end = offset + len(to_json(item, inf_nan_mode="null"))
                        fields = {
                            key: type(value).__name__ for key, value in item.items()
                        }
                        tags = {
                            key: item[key]
                            for key in ("type", "role", "phase", "status")
                            if key in item
                        }
                        children = {
                            key: [p.get("type") for p in value if isinstance(p, dict)][
                                :10
                            ]
                            for key, value in item.items()
                            if isinstance(value, list)
                        }
                        print(
                            "RESPONSES INPUT",
                            index,
                            "bytes",
                            offset + 1,
                            end,
                            tags,
                            fields,
                            children,
                        )
                        offset = end + 1
                    items = body["input"]
                    print(
                        "ACP title request",
                        "title for a conversation" in str(items[-1]),
                    )
                    for label, sample in [
                        ("without catalog", items[1:]),
                        (
                            "catalog alone",
                            [items[0], {"role": "user", "content": "Reply OK."}],
                        ),
                        ("control", [{"role": "user", "content": "Reply OK."}]),
                    ]:
                        try:
                            probe = await send(
                                self, url, {**body, "input": sample}, headers, **kwargs
                            )
                        except ProviderError as probe_error:
                            print("PROVIDER A/B", label, str(probe_error)[:400])
                        else:
                            print("PROVIDER A/B", label, probe.status_code)
                            await probe.aclose()
                raise

        monkeypatch.setattr(EvalClient, "_request", inspect_rejection)
    (episode,) = await run(
        "echo-acp-resume",
        harness=harness,
        runtime=harness_runtime,
        max_turns=8,
        max_tokens=8192,
        rollout_timeout=600,
        # Stirrup replies only through finish.
        taskset_overrides={"finish_tool": harness.id == "stirrup"},
    )
    (trace,) = episode.traces
    assert_one_branch(trace, harness, forks=("openclaw",))
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.stop_condition == "user_closed"
    assert trace.rewards["resumed"].score == 1.0, [
        (
            segment["roles"],
            [value[:240] for value in segment["tool_outputs"]],
            segment["last_reply"][:240],
        )
        for segment in trace.info["acp_segments"]
    ]
    segments = trace.info["acp_segments"]
    assert len(segments) == 2
    assert segments[0]["terminated"] is False
    assert segments[1]["terminated"] is False
    assert trace.last_reply == segments[1]["last_reply"]
    assert "tool" in segments[1]["roles"]
    assert segments[1]["tool_outputs"]
    if harness.id == "pi":
        recall = next(
            tool for tool in trace.tools if tool.name == "mcp__resume__recall"
        )
        assert recall.parameters["properties"]["codeword"]["type"] == "string"
        assert "codeword" in recall.parameters["required"]
    if harness.id == "codex":
        recall = next(tool for tool in trace.tools if tool.name.endswith("recall"))
        assert recall.namespace
        assert "codeword" in recall.parameters["properties"]
        assert any(
            call.namespace == recall.namespace and call.name == recall.name
            for message in trace.assistant_messages
            for call in message.tool_calls or []
        )
    if harness.id == "rlm":
        assert "turns_since_last_compaction" in trace.metrics
        assert all(call.acp is not None for call in trace.calls)
        assert all(
            trace.nodes[parent.node].sampled and node.sampled
            for node in trace.nodes
            for parent in node.semantic_parents
        )
    if harness.id == "prime-agent":
        lifecycle = trace.info["acp_lifecycle"]["ai.primeintellect.prime-agent"]
        assert len(lifecycle) == 2
        for status in lifecycle:
            assert status["infrastructure_status"] == "ok"
            assert status["terminal_quiescence_observed"] is True
            assert (
                status["terminal_quiescence"]["quiescence"]["outstandingSubagents"] == 0
            )
