"""Semantic edges between model requests: resolution to message nodes, ACP metadata, validation."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.rlm.harness import (
    RLM_SESSION_METADATA_KEY,
    RLMHarness,
    RLMHarnessConfig,
)
from verifiers.v1.semantic import (
    ACP_EXTENSION_HEADERS,
    ACP_SEMANTIC_EDGES_METADATA_KEY,
    extract_acp_info,
)
from verifiers.v1.types import AssistantMessage, UserMessage


def _semantic_edge_set() -> vf.SemanticEdgeSet:
    return vf.SemanticEdgeSet(
        edges=[
            vf.SemanticEdge(
                source_request_id="root-turn",
                target_request_id="root-compact",
                type="continuation",
            ),
            vf.SemanticEdge(
                source_request_id="root-turn",
                target_request_id="child-turn",
                type="subagent_call",
            ),
            vf.SemanticEdge(
                source_request_id="child-turn",
                target_request_id="root-after",
                type="subagent_return",
            ),
            vf.SemanticEdge(
                source_request_id="root-compact",
                target_request_id="root-after",
                type="compaction",
            ),
            vf.SemanticEdge(
                source_request_id="root-turn",
                target_request_id="root-after",
                type="critic_review",
            ),
        ],
    )


def test_semantic_edges_resolve_to_message_nodes_and_round_trip():
    """Request edges resolve by exact IDs, not call adjacency or graph shape."""
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="root")),
            MessageNode(
                parent=0, message=AssistantMessage(content="root turn"), sampled=True
            ),
            MessageNode(parent=None, message=UserMessage(content="child")),
            MessageNode(
                parent=2, message=AssistantMessage(content="child turn"), sampled=True
            ),
            MessageNode(parent=None, message=UserMessage(content="summarize")),
            MessageNode(
                parent=4, message=AssistantMessage(content="summary"), sampled=True
            ),
            MessageNode(parent=None, message=UserMessage(content="resume")),
            MessageNode(
                parent=6, message=AssistantMessage(content="done"), sampled=True
            ),
        ],
    )
    tr.calls = [
        vf.ModelCall(
            node=1,
            acp=vf.ACPInfo(request_id="root-turn"),
        ),
        vf.ModelCall(
            node=3,
            acp=vf.ACPInfo(request_id="child-turn"),
        ),
        vf.ModelCall(
            node=5,
            acp=vf.ACPInfo(request_id="root-compact"),
        ),
        vf.ModelCall(
            node=7,
            acp=vf.ACPInfo(request_id="root-after"),
        ),
    ]

    edge_set = _semantic_edge_set()
    tr.add_semantic_edges(vf.SemanticEdgeSet(edges=edge_set.edges[:2]))
    first_semantic_parents = tr.nodes[3].semantic_parents
    tr.add_semantic_edges(vf.SemanticEdgeSet.model_validate(edge_set.model_dump()))
    expected_parents = [
        [],
        [],
        [],
        [vf.ParentLink(node=1, type="subagent_call")],
        [],
        [vf.ParentLink(node=1, type="continuation")],
        [],
        [
            vf.ParentLink(node=3, type="subagent_return"),
            vf.ParentLink(node=5, type="compaction"),
            vf.ParentLink(node=1, type="critic_review"),
        ],
    ]
    assert [node.semantic_parents for node in tr.nodes] == expected_parents
    assert tr.nodes[3].semantic_parents is first_semantic_parents

    restored = vf.WireTrace.model_validate_json(tr.model_dump_json())
    assert [node.semantic_parents for node in restored.nodes] == expected_parents
    assert [call.acp for call in restored.calls] == [call.acp for call in tr.calls]

    # The base ACP layer resolves the generic edge set before harness-owned metadata.
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))
    turn_metadata = {
        ACP_SEMANTIC_EDGES_METADATA_KEY: _semantic_edge_set().model_dump(mode="json"),
        RLM_SESSION_METADATA_KEY: {
            "session_id": restored.id,
            "metrics": {"turns": 4},
        },
    }
    harness._consume_protocol_metadata(restored, turn_metadata)
    harness.acp_turn_result(
        restored, vf.ACPTurn(reply="done", response_metadata=turn_metadata)
    )
    assert restored.metrics["turns"] == 4
    assert [node.semantic_parents for node in restored.nodes] == expected_parents

    # session/close may publish the same cumulative edge set again.
    close_metadata = {
        ACP_SEMANTIC_EDGES_METADATA_KEY: _semantic_edge_set().model_dump(mode="json"),
        RLM_SESSION_METADATA_KEY: {
            "session_id": restored.id,
            "metrics": {"turns": 4},
        },
    }
    harness._consume_protocol_metadata(restored, close_metadata)
    harness.acp_close_result(restored, close_metadata)
    assert restored.metrics["turns"] == 4
    assert [node.semantic_parents for node in restored.nodes] == expected_parents

    # A failed provider exchange and its SDK retry share one logical request ID.
    restored.calls.append(
        vf.ModelCall(
            acp=restored.calls[0].acp,
            error=vf.Error(type="E", message="x"),
        )
    )
    restored.add_semantic_edges(_semantic_edge_set())
    assert [node.semantic_parents for node in restored.nodes] == expected_parents


def test_semantic_edge_uses_last_committed_retry_node():
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="root")),
            MessageNode(
                parent=0, message=AssistantMessage(content="attempt 1"), sampled=True
            ),
            MessageNode(
                parent=0, message=AssistantMessage(content="attempt 2"), sampled=True
            ),
            MessageNode(
                parent=None, message=AssistantMessage(content="next"), sampled=True
            ),
        ],
        calls=[
            vf.ModelCall(node=1, acp=vf.ACPInfo(request_id="retried")),
            vf.ModelCall(node=2, acp=vf.ACPInfo(request_id="retried")),
            vf.ModelCall(node=3, acp=vf.ACPInfo(request_id="next")),
        ],
    )

    tr.add_semantic_edges(
        vf.SemanticEdgeSet(
            edges=[
                vf.SemanticEdge(
                    source_request_id="retried",
                    target_request_id="next",
                    type="continuation",
                )
            ]
        )
    )

    assert tr.nodes[3].semantic_parents == [vf.ParentLink(node=2, type="continuation")]


def test_semantic_edge_loops_attach_and_invalid_sets_do_not_partially_mutate():
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="start")),
            MessageNode(
                parent=0, message=AssistantMessage(content="first"), sampled=True
            ),
            MessageNode(parent=1, message=UserMessage(content="continue")),
            MessageNode(
                parent=2, message=AssistantMessage(content="second"), sampled=True
            ),
        ],
        calls=[
            vf.ModelCall(node=1, acp=vf.ACPInfo(request_id="first")),
            vf.ModelCall(node=3, acp=vf.ACPInfo(request_id="second")),
        ],
    )

    loop = vf.SemanticEdge(
        source_request_id="second", target_request_id="first", type="subagent_cancel"
    )
    unknown = vf.SemanticEdge(
        source_request_id="first", target_request_id="missing", type="custom"
    )
    with pytest.raises(ValueError, match="has no committed message node"):
        tr.add_semantic_edges(vf.SemanticEdgeSet(edges=[loop, unknown]))
    assert all(not node.semantic_parents for node in tr.nodes)

    tr.add_semantic_edges(vf.SemanticEdgeSet(edges=[loop]))
    assert tr.nodes[1].semantic_parents == [
        vf.ParentLink(node=3, type="subagent_cancel")
    ]


def test_acp_info_is_validated_and_stripped():
    headers = {
        "Authorization": "Bearer local",
        "Idempotency-Key": "provider-key",
        "X-ACP-Model-Request-ID": "request-1",
        "OpenAI-Beta": "feature",
    }
    acp, forwarded = extract_acp_info(headers)
    assert acp == vf.ACPInfo(request_id="request-1")
    assert not ACP_EXTENSION_HEADERS.intersection(map(str.lower, forwarded))
    assert forwarded["Idempotency-Key"] == "provider-key"
    assert forwarded["OpenAI-Beta"] == "feature"

    absent, unchanged = extract_acp_info({"OpenAI-Beta": "feature"})
    assert absent is None and unchanged == {"OpenAI-Beta": "feature"}

    with pytest.raises(ValueError, match="not a valid ACP request ID"):
        extract_acp_info({"X-ACP-Model-Request-ID": "not/a/valid/id"})


def test_acp_semantic_edge_metadata_is_optional():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
    )
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))

    harness._consume_protocol_metadata(trace, {})

    assert all(not node.semantic_parents for node in trace.nodes)

    harness._consume_protocol_metadata(
        trace, {ACP_SEMANTIC_EDGES_METADATA_KEY: {"edges": []}}
    )

    assert all(not node.semantic_parents for node in trace.nodes)


def test_acp_derives_compaction_attempt_branch_trainability():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="work")),
            MessageNode(
                parent=0,
                message=AssistantMessage(content="working"),
                sampled=True,
                token_ids=[1],
                mask=[True],
                logprobs=[-0.1],
            ),
            MessageNode(parent=1, message=UserMessage(content="summarize")),
            MessageNode(
                parent=2,
                message=AssistantMessage(content="bad tool call"),
                sampled=True,
                token_ids=[2, 3],
                mask=[True, True],
                logprobs=[-0.2, -0.3],
            ),
            MessageNode(
                parent=2,
                message=AssistantMessage(content="accepted summary"),
                sampled=True,
                token_ids=[4, 5],
                mask=[True, True],
                logprobs=[-0.4, -0.5],
            ),
            MessageNode(parent=0, message=UserMessage(content="compacted context")),
            MessageNode(
                parent=5,
                message=AssistantMessage(content="answer"),
                sampled=True,
                token_ids=[6],
                mask=[True],
                logprobs=[-0.6],
            ),
        ],
        calls=[
            vf.ModelCall(node=1, acp=vf.ACPInfo(request_id="work")),
            vf.ModelCall(node=3, acp=vf.ACPInfo(request_id="rejected")),
            vf.ModelCall(node=4, acp=vf.ACPInfo(request_id="accepted")),
            vf.ModelCall(node=6, acp=vf.ACPInfo(request_id="resumed")),
        ],
    )
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))
    harness._consume_protocol_metadata(
        trace,
        {
            ACP_SEMANTIC_EDGES_METADATA_KEY: {
                "edges": [
                    {
                        "source_request_id": "work",
                        "target_request_id": "rejected",
                        "type": "compaction_attempt",
                    },
                    {
                        "source_request_id": "work",
                        "target_request_id": "accepted",
                        "type": "compaction_attempt",
                    },
                ]
            },
        },
    )

    attempts = {branch.nodes[-1].message.content: branch for branch in trace.branches}
    assert attempts["bad tool call"].trainable is False
    assert attempts["accepted summary"].trainable is False

    harness._consume_protocol_metadata(
        trace,
        {
            ACP_SEMANTIC_EDGES_METADATA_KEY: {
                "edges": [
                    {
                        "source_request_id": "work",
                        "target_request_id": "rejected",
                        "type": "compaction_attempt",
                    },
                    {
                        "source_request_id": "work",
                        "target_request_id": "accepted",
                        "type": "compaction_attempt",
                    },
                    {
                        "source_request_id": "accepted",
                        "target_request_id": "resumed",
                        "type": "compaction",
                    },
                ]
            },
        },
    )

    assert trace.nodes[3].sampled is True
    assert list(trace.nodes[3].mask) == [True, True]
    assert list(trace.nodes[4].mask) == [True, True]
    assert list(trace.nodes[6].mask) == [True]
    assert trace.nodes[3].semantic_parents == [
        vf.ParentLink(node=1, type="compaction_attempt")
    ]
    assert trace.nodes[4].semantic_parents == [
        vf.ParentLink(node=1, type="compaction_attempt")
    ]
    assert trace.nodes[6].semantic_parents == [vf.ParentLink(node=4, type="compaction")]
    assert trace.num_branches == 3
    branches = {branch.nodes[-1].message.content: branch for branch in trace.branches}
    assert branches["bad tool call"].trainable is False
    assert branches["accepted summary"].trainable is True
    assert branches["answer"].trainable is True
    assert branches["bad tool call"].nodes[-2] is trace.nodes[2]
    assert branches["accepted summary"].nodes[-2] is trace.nodes[2]

    restored = vf.WireTrace.model_validate_json(trace.model_dump_json())
    assert restored.nodes[3].sampled is True
    assert list(restored.nodes[3].mask) == [True, True]
    assert list(restored.nodes[4].mask) == [True, True]
    restored_branches = {
        branch.nodes[-1].message.content: branch for branch in restored.branches
    }
    assert restored_branches["bad tool call"].trainable is False
    assert restored_branches["accepted summary"].trainable is True


def test_semantic_edge_set_rejects_duplicate_and_self_edges_but_accepts_loops():
    edge_set = _semantic_edge_set().model_dump(mode="json")
    edge_set["edges"].append(edge_set["edges"][0])
    with pytest.raises(ValueError, match="duplicate semantic edge"):
        vf.SemanticEdgeSet.model_validate(edge_set)

    with pytest.raises(ValueError, match="cannot link a request to itself"):
        vf.SemanticEdgeSet.model_validate(
            {
                "edges": [
                    {
                        "source_request_id": "request-1",
                        "target_request_id": "request-1",
                        "type": "custom",
                    }
                ]
            }
        )

    # A parent cancels the child whose report it just read: the cancel lands on that child turn.
    edge_set = _semantic_edge_set().model_dump(mode="json")
    edge_set["edges"].append(
        {
            "source_request_id": "root-after",
            "target_request_id": "child-turn",
            "type": "subagent_cancel",
        }
    )
    assert len(vf.SemanticEdgeSet.model_validate(edge_set).edges) == len(
        edge_set["edges"]
    )


def test_semantic_edge_set_accepts_deep_acyclic_chain():
    edge_set = vf.SemanticEdgeSet(
        edges=[
            vf.SemanticEdge(
                source_request_id=f"request-{index}",
                target_request_id=f"request-{index + 1}",
                type="continuation",
            )
            for index in range(2_000)
        ]
    )

    assert len(edge_set.edges) == 2_000
