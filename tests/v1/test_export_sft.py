"""The export-sft reshaping: traces -> SFT rows (unit-level, no model, no files)."""

import json

import verifiers.v1 as vf
from verifiers.v1.cli.export_sft import select, sft_rows
from verifiers.v1.configs.export_sft import ExportSftConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

TOOL = vf.Tool(
    name="echo_back",
    description="Echo a message back, stamped.",
    parameters={
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
)


def _tool_trace(reward: float = 1.0) -> vf.Trace:
    """A linear tool-use rollout: user -> assistant tool call -> tool result -> answer."""
    tr = vf.Trace(
        task=vf.Task(idx=0, prompt="q"),
        tool_defs=[TOOL],
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(
                parent=0,
                message=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="c1", name="echo_back", arguments='{"message": "hi"}'
                        )
                    ]
                ),
                sampled=True,
            ),
            MessageNode(
                parent=1,
                message=ToolMessage(tool_call_id="c1", content="hi [stamped]"),
                sampled=False,
            ),
            MessageNode(
                parent=2, message=AssistantMessage(content="hi [stamped]"), sampled=True
            ),
        ],
    )
    tr.record_reward("reference", reward)
    return tr


def test_sft_rows_shape():
    # One row per branch; messages in OpenAI chat wire shape (tool_calls nested under
    # `function`, tool results carrying `tool_call_id`); tool_defs JSON-encoded.
    (row,) = sft_rows(_tool_trace())
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    call = row["messages"][1]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"] == {"name": "echo_back", "arguments": '{"message": "hi"}'}
    assert row["messages"][2]["tool_call_id"] == "c1"

    (tool,) = json.loads(row["tool_defs"])
    assert tool["name"] == "echo_back"
    assert tool["parameters"] == TOOL.parameters


def test_sft_rows_one_per_branch():
    # Two leaves off one root (a compaction-shaped trace) -> one training row per branch.
    tr = vf.Trace(
        task=vf.Task(idx=0, prompt="q"),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a1"), sampled=True),
            MessageNode(parent=0, message=AssistantMessage(content="a2"), sampled=True),
        ],
    )
    rows = sft_rows(tr)
    assert len(rows) == 2
    assert json.loads(rows[0]["tool_defs"]) == []  # no tools advertised -> empty list


def test_select_filters():
    solved, failed = _tool_trace(reward=1.0), _tool_trace(reward=0.0)
    errored = _tool_trace()
    errored.capture_error(RuntimeError("boom"))
    truncated = _tool_trace()
    truncated.stop("max_turns")

    # Errored traces always drop; the rest follow the knobs.
    assert select([solved, errored], ExportSftConfig()) == [solved]
    assert select([solved, failed], ExportSftConfig(min_reward=1.0)) == [solved]
    assert select([solved, failed], ExportSftConfig()) == [solved, failed]
    assert select([solved, truncated], ExportSftConfig(drop_truncated=True)) == [solved]
