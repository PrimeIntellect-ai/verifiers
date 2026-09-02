"""Trace construction + serialization round-trip: a dumped trace re-validates with plain pydantic
(derived values — reward/is_truncated/error/duration — are properties, not serialized, so they just
recompute on load), transient `state` never crosses the wire, and the permissive `WireTrace` loads a
dump without importing the originating taskset."""

import json
from types import SimpleNamespace

import pytest
from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.agent import Interaction
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.rlm.harness import (
    RLM_SESSION_METADATA_KEY,
    RLMHarness,
    RLMHarnessConfig,
)
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.semantic import (
    ACP_EXTENSION_HEADERS,
    ACP_SEMANTIC_EDGES_METADATA_KEY,
    extract_acp_info,
)
from verifiers.v1.types import AssistantMessage, UserMessage


class MyTask(vf.TaskData):
    answer: str = ""  # a task-specific field WireTaskData must absorb


class MyState(vf.State):
    score: int = 0


class EnvTask(vf.TaskData):
    verifier_env: dict[str, str] = Field(default_factory=dict)
    """An environment mapping inside task data, as Harbor's `verifier_env`."""
    verifier: dict = Field(default_factory=dict)
    """A nested block with its own `env`, as Harbor's `verifier.env`."""


class FailingSegmentRollout:
    ok = Rollout.ok
    closed = Rollout.closed
    fail = Rollout.fail
    step = Rollout.step


@pytest.mark.asyncio
async def test_failed_segment_does_not_reuse_prior_root_reply():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=0, prompt=None),
            key="dataset/example-0",
            hash="content-digest",
        ),
    )
    trace.root_reply = "previous reply"

    class FailingSession:
        async def turn(self, messages):
            trace.nodes.append(
                MessageNode(
                    parent=None,
                    message=AssistantMessage(content="current partial reply"),
                    sampled=True,
                )
            )
            raise RuntimeError("segment failed after sampling")

    run = FailingSegmentRollout()
    run.trace = trace
    run._opened = True
    run._closed = False
    run._failed = False
    run._failure = None
    run._borrowed_runtime = None
    run.runtime = None
    run._agent_time_remaining = None
    run._timeouts = RolloutTimeouts()
    run._harness_session = FailingSession()
    run._session = SimpleNamespace(
        request_interceptors=[],
        error=None,
        stopped=False,
    )
    run.deadline_at = None

    segment = await Interaction(run).turn("next")

    assert segment.last_reply == "current partial reply"
    assert trace.root_reply is None
    assert trace.last_reply == "current partial reply"


def test_bare_trace_round_trip():
    # The minimal trace: a base task, no nodes, no extras — dump and back into a plain Trace.
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=3, prompt="hello"),
            key="dataset/example-3",
            hash="content-digest",
        ),
    )
    rt = vf.Trace.model_validate(tr.model_dump())
    assert rt.id == tr.id
    assert rt.task.type == "Task"
    assert rt.task.data.idx == 3 and rt.task.data.prompt == "hello"
    assert rt.task.key == "dataset/example-3" and rt.task.hash == "content-digest"
    assert rt.num_turns == 0 and rt.num_branches == 0
    assert rt.reward == 0.0 and rt.errors == []


def test_custom_task_state_round_trip():
    # Custom data and state round-trip into the same parameterization. Data fields are
    # typed (not just `model_extra`); `state` is runtime-only and never crosses the wire.
    tr = vf.Trace[MyTask, MyState](
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="MyTask", data=MyTask(idx=0, prompt="q", answer="gold")),
        state=MyState(score=7),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a"), sampled=True),
        ],
    )
    tr.record_reward("r", 0.5)
    wire = tr.model_dump()
    assert "state" not in wire  # transient state is excluded from the dump

    rt = vf.Trace[MyTask, MyState].model_validate(wire)
    assert (
        isinstance(rt.task.data, MyTask) and rt.task.data.answer == "gold"
    )  # typed custom field
    assert rt.task.type == "MyTask"  # the producing class's name survives the wire
    assert rt.num_turns == 1 and rt.num_branches == 1
    assert rt.reward == 0.5  # property recomputed from `rewards`


def test_wire_trace_round_trip():
    # Two leaves off one root → 2 branches (a compaction-shaped trace), so the round-trip has to
    # carry node `parent` links for `num_branches` to survive.
    tr = vf.Trace[MyTask, vf.State](
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="MyTask", data=MyTask(idx=0, prompt="q", answer="a")),
        tools=[vf.Tool(name="echo", description="", parameters={"type": "object"})],
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a1"), sampled=True),
            MessageNode(parent=0, message=AssistantMessage(content="a2"), sampled=True),
        ],
    )
    tr.record_reward("r", 1.0)
    tr.rewards.setdefault("solved", None)  # seeded: expected but never scored
    tr.metrics.setdefault("acc", None)
    tr.info = {"build": "ok"}
    tr.root_reply = "root answer"
    tr.stop("done")

    # the dump is plain pydantic — derived values are properties, so they're not serialized
    data = json.loads(tr.model_dump_json(exclude_none=True))
    assert "reward" not in data and "is_truncated" not in data
    # exclude_none drops None FIELDS, not None dict values — unscored seeds survive
    assert data["rewards"]["solved"] is None and data["metrics"]["acc"] is None

    rt = vf.WireTrace.model_validate(data)
    assert rt.num_branches == tr.num_branches == 2  # branch topology survived
    assert rt.num_turns == tr.num_turns == 2
    assert rt.reward == 1.0  # property recomputed from `rewards`, seeds contribute 0
    assert rt.rewards["solved"] is None
    assert rt.stop_condition == "done"
    assert rt.info == {"build": "ok"}
    assert rt.root_reply == "root answer"
    assert rt.last_reply == "root answer"
    rt.root_reply = ""
    assert rt.last_reply == ""
    assert (
        rt.tools == tr.tools
    )  # the advertised tools persist (tool-use SFT reads them)
    assert rt.task.data.model_extra == {
        "answer": "a"
    }  # taskset extras preserved on WireTaskData

    # the env-server wire form (a plain model_dump) loads too
    assert vf.WireTrace.model_validate(tr.model_dump()).num_branches == 2


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


def test_semantic_edge_cycle_is_rejected_without_partial_mutation():
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

    with pytest.raises(ValueError, match="cycle in the message graph"):
        tr.add_semantic_edges(
            vf.SemanticEdgeSet(
                edges=[
                    vf.SemanticEdge(
                        source_request_id="second",
                        target_request_id="first",
                        type="custom",
                    )
                ]
            )
        )

    assert all(not node.semantic_parents for node in tr.nodes)


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


def test_semantic_edge_set_rejects_duplicate_self_and_cyclic_edges():
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

    edge_set = _semantic_edge_set().model_dump(mode="json")
    edge_set["edges"].append(
        {
            "source_request_id": "root-after",
            "target_request_id": "root-turn",
            "type": "custom",
        }
    )
    with pytest.raises(ValueError, match="semantic edge cycle"):
        vf.SemanticEdgeSet.model_validate(edge_set)


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


def test_push_traces_uploads_redacted_projection(monkeypatch):
    """`--push` drops the config fields that carry credentials and replaces every known
    secret the agent echoed — including one the harness printed inside a quoted JSON
    tool result — while the saved record keeps its config and the tokens never touch
    disk. Ordinary text and short or non-credential values stay as they are."""
    import httpx

    from verifiers.v1.clients import EvalClientConfig
    from verifiers.v1.configs.cli.eval import EvalConfig
    from verifiers.v1.configs.harness import HarnessConfig
    from verifiers.v1.episode import EnvInfo, Episode
    from verifiers.v1.utils import platform

    monkeypatch.setenv("PRIME_API_KEY", "prime-platform-key-0001")
    monkeypatch.setenv("MODEL_API_KEY", "sk-model-key-000000000001")
    monkeypatch.setenv("HOST_HF_TOKEN", "hf_host_token_00000001")
    monkeypatch.setenv("KEYCLOAK_REALM", "production-realm")  # KEYCLOAK is not KEY
    monkeypatch.setenv("PGPASSWORD", "pg-pass-000001")  # but PGPASSWORD is a password
    client = EvalClientConfig(
        base_url="https://svc:url-pass-000001@models.example/v1",
        api_key_var="MODEL_API_KEY",
        headers={"X-Auth": 'he said "hi" 0001', "X-Trace": "plain-header"},
    )
    config = EvalConfig(env={"taskset": {"id": "echo-v1"}}, model="m", client=client)
    secrets = {
        "prime-platform-key-0001",
        "sk-model-key-000000000001",
        "hf_host_token_00000001",
        'he said "hi" 0001',
        "hf_harness_token_0001",
        "intercept-token-0001",
        "hooks/abc/def",
        "judge-key-000000001",
        "db%40pass-000001",
        "db@pass-000001",  # the URL password as a client echoes it
        "url-pass-000001",
        "grader-token-0001",
        "retry-token-0001",  # a discarded attempt's token, carried with its errors
        "pg-pass-000001",
    }
    echo = " ".join(sorted(secrets)) + " debug=1 plain-header production-realm"
    # A tool result as another encoder would emit it, `/` escaped and uppercase hex.
    tool_result = (
        '{"env": {"KEY": "he said \\"hi\\" 0001", "url": "hooks\\/abc\\/def", '
        '"ok": "plain-header", "u": "\\u00FCn\\u00EFcode"}}'
    )
    deep = json.dumps({"log": json.dumps({"auth": "hooks/abc/def", "n": 1})})
    trace = vf.Trace[EnvTask, vf.State](
        agent=vf.AgentInfo(
            config=vf.AgentConfig(
                client=client,
                harness=HarnessConfig(
                    id="bash",
                    env={"HF_TOKEN": "hf_harness_token_0001", "DEBUG": "1"},
                    forward_env=["HOME"],
                ),
            )
        ),
        task=vf.TraceTask(
            type="EnvTask",
            data=EnvTask(
                idx=0,
                prompt="q",
                verifier_env={
                    "JUDGE_API_KEY": "judge-key-000000001",
                    "DATABASE_URL": "postgres://app:db%40pass-000001@db/x",
                    "MODE": "fast",
                },
                verifier={"env": {"GRADER_TOKEN": "grader-token-0001", "N": "1"}},
            ),
        ),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content=echo), sampled=True),
            MessageNode(
                parent=1, message=UserMessage(content=tool_result), sampled=False
            ),
            MessageNode(parent=2, message=UserMessage(content=deep), sampled=False),
        ],
        upload_secrets=["intercept-token-0001", "hooks/abc/def"],
    )
    episode = Episode[EnvTask, vf.State](
        env=EnvInfo(id="echo-v1"),
        task=trace.task,
        traces=[trace],
        ok=True,
        upload_secrets=["retry-token-0001"],
    )

    posted: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        posted[request.url.path] = request.content
        if request.url.path.endswith("/environmentshub/resolve"):
            return httpx.Response(200, json={"data": {"id": "env-1"}})
        if request.url.path.endswith("/evaluations/"):
            return httpx.Response(200, json={"evaluation_id": "eval-1"})
        return httpx.Response(200, json={})

    real_client = httpx.Client
    monkeypatch.setattr(
        platform.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    url = platform.push_traces([episode], config)

    assert url is not None and url.endswith("/dashboard/evaluations/eval-1")
    body = posted["/api/v1/evaluations/eval-1/samples"].decode()
    for secret in secrets:
        assert secret not in body and secret.replace('"', '\\"') not in body
    payload = json.loads(body)
    native = payload["samples"][0]["info"]["native_wrapper"]["traces"][0]
    assert "headers" not in native["agent"]["config"]["client"]
    assert (
        native["agent"]["config"]["client"]["base_url"]
        == "https://svc:[REDACTED]@models.example/v1"
    )
    assert "env" not in native["agent"]["config"]["harness"]
    assert native["agent"]["config"]["harness"]["forward_env"] == ["HOME"]
    assert "upload_secrets" not in native
    messages = payload["samples"][0]["completion"]
    assert messages[1]["content"].endswith("debug=1 plain-header production-realm")
    assert json.loads(messages[2]["content"]) == {
        "env": {
            "KEY": "[REDACTED]",
            "url": "[REDACTED]",
            "ok": "plain-header",
            "u": "ünïcode",
        }
    }
    assert json.loads(json.loads(messages[3]["content"])["log"]) == {
        "auth": "[REDACTED]",
        "n": 1,
    }
    # The task's own environment mapping keeps its keys and URL shape; only the
    # credential-named values and the URL password go.
    assert payload["samples"][0]["task"]["verifier_env"] == {
        "JUDGE_API_KEY": "[REDACTED]",
        "DATABASE_URL": "postgres://app:[REDACTED]@db/x",
        "MODE": "fast",
    }
    assert payload["samples"][0]["task"]["verifier"] == {
        "env": {"GRADER_TOKEN": "[REDACTED]", "N": "1"}
    }
    # The saved record keeps the run reproducible; only the tokens stay off disk.
    record = episode.to_record()["traces"][0]
    assert (
        record["agent"]["config"]["harness"]["env"]["HF_TOKEN"]
        == "hf_harness_token_0001"
    )
    assert record["agent"]["config"]["client"]["headers"]["X-Trace"] == "plain-header"
    assert "upload_secrets" not in record
    assert "upload_secrets" not in episode.to_record()
