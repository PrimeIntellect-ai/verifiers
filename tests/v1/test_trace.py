"""Trace construction + serialization round-trip: a dumped trace re-validates with plain pydantic
(derived values — reward/is_truncated/error/duration — are properties, not serialized, so they just
recompute on load), transient `state` never crosses the wire, and the permissive `WireTrace` loads a
dump without importing the originating taskset."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from prime_evals import fingerprint_secret

import verifiers.v1 as vf
from verifiers.v1.agent import Agent, Interaction
from verifiers.v1.cli.output import (
    TRACES_FILE,
    UPLOAD_SECRET_FINGERPRINTS_FILE,
    read_episodes,
    write_episode,
)
from verifiers.v1.configs.client import (
    PRIME_TEAM_ID_HEADER,
    EvalClientConfig,
    resolve_headers,
)
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.episode import Episode
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.rlm.harness import (
    RLM_SESSION_METADATA_KEY,
    RLMHarness,
    RLMHarnessConfig,
)
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.runtimes.docker import DockerRuntimeInfo
from verifiers.v1.semantic import (
    ACP_EXTENSION_HEADERS,
    ACP_SEMANTIC_EDGES_METADATA_KEY,
    extract_acp_info,
)
from verifiers.v1.trace import Error
from verifiers.v1.types import AssistantMessage, UserMessage
from verifiers.v1.utils import platform
from verifiers.v1.utils.platform import PushState, build_samples, push_traces
from verifiers.v1.utils.retries import run_episode_with_retry


class MyTask(vf.TaskData):
    answer: str = ""  # a task-specific field WireTaskData must absorb


class MyState(vf.State):
    score: int = 0


class FailingSegmentRollout:
    ok = Rollout.ok
    closed = Rollout.closed
    fail = Rollout.fail
    step = Rollout.step


class StopInPreflight:
    def __init__(
        self,
        capability: str,
        has_fingerprint: bool = True,
        persisted_secrets: tuple[str, ...] = (),
    ) -> None:
        self.capability = capability
        self.has_fingerprint = has_fingerprint
        self.persisted_secrets = (capability, *persisted_secrets)

    def __call__(
        self, payload, known_secrets, secret_sources, secret_fingerprints
    ) -> None:
        assert payload["samples"]
        assert self.capability in json.dumps(payload)
        assert (
            all(
                fingerprint_secret(secret) in secret_fingerprints
                for secret in self.persisted_secrets
            )
            is self.has_fingerprint
        )
        assert {
            "prime-api-key",
            "agent-api-key-0123456789",
            "run-api-key-0123456789",
        }.issubset(known_secrets)
        sources = {
            name: value for source in secret_sources for name, value in source.items()
        }
        assert sources["RUNTIME_SECRET"] == "runtime-secret-0123456789"
        if "FORWARDED_RUNTIME_SECRET" in sources:
            assert sources["FORWARDED_RUNTIME_SECRET"] == "forwarded-secret-0123456789"
        assert sources["X-Auth"] in {
            "agent-header-secret-0123456789",
            "run-header-secret-0123456789",
        }
        raise RuntimeError("preflight stopped upload")


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


def test_platform_sample_separates_runtime_data_without_reducing_review_data():
    task = vf.WireTaskData.model_validate(
        {
            "idx": 4,
            "prompt": "Solve the task",
            "answer": "reference answer",
            "rubric": "award one point for the reference answer",
            "workdir": "/Users/alice/private-project",
            "network_allow": ["10.0.0.4"],
        }
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(
            config=vf.AgentConfig(
                harness=HarnessConfig(
                    env={"RUNTIME_SETTING": "literal-value"},
                    forward_env=["USER_SECRET"],
                    skills=["/Users/alice/private-skill"],
                ),
                client=EvalClientConfig(
                    base_url="https://example.com/v1",
                    api_key_var="USER_SECRET",
                    headers={
                        "X-Custom": "runtime-header",
                        "X-Prime-Team-ID": "team-private",
                    },
                ),
            ),
            runtime=DockerRuntimeInfo(
                id="container-private",
                image="private.example.com/image",
                workdir="/Users/alice/runtime",
            ),
        ),
        task=vf.TraceTask(
            type="Task",
            data=task,
            key="content-derived-key",
            hash="content-derived-hash",
        ),
        nodes=[
            MessageNode(
                message=AssistantMessage(
                    content="reviewable completion",
                    reasoning_content="reviewable reasoning",
                    provider_state=[
                        {
                            "id": "provider-action-id",
                            "container_id": "container-private",
                            "encrypted_content": "opaque-continuation",
                            "signature": "opaque-signature",
                            "data": "opaque-redacted-thinking",
                            "input": "reviewable provider input",
                            "phase": "commentary",
                        }
                    ],
                ),
                sampled=True,
                token_ids=[1, 2],
                mask=[True, True],
                is_content=[True, True],
                logprobs=[-0.1, -0.2],
            ),
        ],
        info={"raw_credentials": "reviewable trace info"},
        errors=[
            Error(
                type="RuntimeError",
                message="failed in /Users/alice/private-project",
                traceback="reviewable traceback",
            )
        ],
    )
    episode = Episode(task=trace.task, traces=[trace])
    sample = build_samples([episode])[0]

    native_trace = sample["info"]["native_wrapper"]["traces"][0]
    client = native_trace["agent"]["config"]["client"]
    harness = native_trace["agent"]["config"]["harness"]
    assert "headers" not in client and client["api_key_var"] == "USER_SECRET"
    assert {"env", "skills"}.isdisjoint(harness)
    assert harness["forward_env"] == ["USER_SECRET"]
    assert "id" not in native_trace["agent"]["runtime"]
    assert native_trace["info"] == {"raw_credentials": "reviewable trace info"}
    assert native_trace["errors"][0]["traceback"] == "reviewable traceback"
    assert native_trace["task"] == trace.task.model_dump(mode="json", exclude_none=True)

    assistant = native_trace["nodes"][0]
    assert {"token_ids", "mask", "is_content", "logprobs"}.isdisjoint(assistant)
    assert assistant["message"]["reasoning_content"] == "reviewable reasoning"
    provider_state = assistant["message"]["provider_state"][0]
    assert provider_state == {
        "id": "provider-action-id",
        "container_id": "container-private",
        "input": "reviewable provider input",
        "phase": "commentary",
    }

    local_trace = episode.to_record()["traces"][0]
    assert local_trace["agent"]["config"]["client"]["headers"]
    assert local_trace["agent"]["config"]["harness"]["env"]
    assert local_trace["agent"]["runtime"]["id"] == "container-private"
    assert local_trace["nodes"][0]["message"]["provider_state"][0]["encrypted_content"]


def test_prime_team_header_is_resolved_live_without_entering_config(monkeypatch):
    monkeypatch.setenv("PRIME_TEAM_ID", "team-private")
    config = EvalClientConfig(headers={"X-Custom": "keep"})

    assert config.headers == {"X-Custom": "keep"}
    assert PRIME_TEAM_ID_HEADER not in config.model_dump_json()
    assert resolve_headers(config) == {
        "X-Custom": "keep",
        PRIME_TEAM_ID_HEADER: "team-private",
    }


def test_trace_push_runs_preflight_before_opening_the_network(monkeypatch, tmp_path):
    monkeypatch.setenv("FORWARDED_RUNTIME_SECRET", "forwarded-secret-0123456789")
    monkeypatch.setenv("AGENT_API_KEY", "agent-api-key-0123456789")
    monkeypatch.setenv("RUN_API_KEY", "run-api-key-0123456789")
    monkeypatch.setenv("PRIME_API_KEY", "prime-api-key")
    capability = "rollout-capability-0123456789"
    trace = vf.Trace(
        agent=vf.AgentInfo(
            config=vf.AgentConfig(
                client=EvalClientConfig(
                    api_key_var="AGENT_API_KEY",
                    headers={"X-Auth": "agent-header-secret-0123456789"},
                ),
                harness=HarnessConfig(
                    env={"RUNTIME_SECRET": "runtime-secret-0123456789"},
                    forward_env=["FORWARDED_RUNTIME_SECRET"],
                ),
            )
        ),
        task=vf.TraceTask(
            type="Task", data=vf.TaskData(idx=0, prompt=f"echoed {capability}")
        ),
    )
    trace.upload_secrets.append(capability)
    episode = Episode(task=trace.task, traces=[trace])
    episode_capability = "episode-capability-0123456789"
    episode.upload_secrets.append(episode_capability)
    assert "upload_secrets" not in json.dumps(episode.to_record())
    wire_episode = vf.WireEpisode.model_validate(episode.model_dump())
    assert wire_episode.upload_secrets == [episode_capability]
    assert wire_episode.traces[0].upload_secrets == [capability]
    (tmp_path / TRACES_FILE).touch()
    write_episode(tmp_path, episode)
    assert capability in (tmp_path / TRACES_FILE).read_text()
    assert capability not in (tmp_path / UPLOAD_SECRET_FINGERPRINTS_FILE).read_text()
    assert (
        episode_capability
        not in (tmp_path / UPLOAD_SECRET_FINGERPRINTS_FILE).read_text()
    )
    with (tmp_path / UPLOAD_SECRET_FINGERPRINTS_FILE).open("a") as f:
        f.write('{"episode_id":')
    (episode,) = read_episodes(tmp_path, vf.WireTrace)
    assert episode.upload_secrets == []
    assert episode.traces[0].upload_secrets == []
    monkeypatch.delenv("FORWARDED_RUNTIME_SECRET")
    config = SimpleNamespace(
        env=SimpleNamespace(taskset=SimpleNamespace(id="test-env")),
        run=SimpleNamespace(id="run-1", name="test-run"),
        model="test-model",
        num_rollouts=1,
        client=EvalClientConfig(
            api_key_var="RUN_API_KEY",
            headers={"X-Auth": "run-header-secret-0123456789"},
        ),
    )
    state = PushState()

    monkeypatch.setattr(
        platform,
        "prepare_upload",
        StopInPreflight(
            capability,
            persisted_secrets=(
                "forwarded-secret-0123456789",
                episode_capability,
            ),
        ),
    )
    monkeypatch.setattr(
        platform,
        "APIClient",
        lambda **unused_kwargs: pytest.fail("network opened before upload preflight"),
    )

    assert push_traces([episode], config, state, tmp_path) is None
    assert state.error == "RuntimeError: preflight stopped upload"

    state = PushState()
    monkeypatch.setattr(
        platform,
        "prepare_upload",
        StopInPreflight(capability, has_fingerprint=False),
    )
    assert push_traces([episode], config, state) is None
    assert state.error == "RuntimeError: preflight stopped upload"


def test_retry_history_keeps_generated_upload_secrets(monkeypatch):
    task = vf.TraceTask(type="Task", data=vf.TaskData(idx=0))
    agent_info = vf.AgentInfo(config=vf.AgentConfig())
    first_trace = vf.Trace(agent=agent_info, task=task)
    first_trace.errors.append(Error(type="ProviderError", message="retry"))
    first_trace.upload_secrets.append("first-agent-capability-0123456789")
    final_trace = vf.Trace(agent=agent_info, task=task, ok=True)
    final_trace.upload_secrets.append("final-agent-capability-0123456789")
    agent = Agent.__new__(Agent)
    agent._closed = False
    agent.config = SimpleNamespace(retries=RetryConfig(max_retries=1))
    agent._run_once = AsyncMock(side_effect=[first_trace, final_trace])
    monkeypatch.setattr("verifiers.v1.agent.asyncio.sleep", AsyncMock())

    result = asyncio.run(agent.run(SimpleNamespace()))

    assert result.upload_secrets == [
        "first-agent-capability-0123456789",
        "final-agent-capability-0123456789",
    ]

    monkeypatch.setenv("DISCARDED_API_KEY", "discarded-api-key-0123456789")
    discarded_trace = vf.Trace(
        agent=vf.AgentInfo(
            config=vf.AgentConfig(
                client=EvalClientConfig(
                    api_key_var="DISCARDED_API_KEY",
                    headers={"X-Auth": "discarded-header-secret-0123456789"},
                ),
                harness=HarnessConfig(
                    env={"RUNTIME_SECRET": "discarded-runtime-secret-0123456789"}
                ),
            )
        ),
        task=task,
        errors=[Error(type="ProviderError", message="retry")],
        upload_secrets=["discarded-agent-capability-0123456789"],
    )
    first_episode = Episode(
        task=task,
        errors=[Error(type="EnvError", message="retry")],
        upload_secrets=["first-episode-capability-0123456789"],
        traces=[discarded_trace],
    )
    final_episode = Episode(task=task, ok=True, traces=[final_trace])
    attempts = AsyncMock(side_effect=[first_episode, final_episode])

    result = asyncio.run(run_episode_with_retry(attempts, RetryConfig(max_retries=1)))

    assert result.upload_secrets == [
        "first-episode-capability-0123456789",
        "discarded-agent-capability-0123456789",
        "discarded-api-key-0123456789",
        "discarded-runtime-secret-0123456789",
        "discarded-header-secret-0123456789",
    ]


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
