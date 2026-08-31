"""Trace construction + serialization round-trip: a dumped trace re-validates with plain pydantic
(derived values — reward/is_truncated/error/duration — are properties, not serialized, so they just
recompute on load), transient `state` never crosses the wire, and the permissive `WireTrace` loads a
dump without importing the originating taskset."""

import json
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import Interaction
from verifiers.v1.cli.output import TRACES_FILE, write_episode
from verifiers.v1.configs.client import (
    PRIME_TEAM_ID_HEADER,
    EvalClientConfig,
    resolve_headers,
)
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.episode import Episode
from verifiers.v1.graph import MessageNode
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.runtimes.docker import DockerRuntimeInfo
from verifiers.v1.trace import Error
from verifiers.v1.types import AssistantMessage, UserMessage
from verifiers.v1.utils import platform
from verifiers.v1.utils.platform import PushState, build_samples, push_traces


class MyTask(vf.TaskData):
    answer: str = ""  # a task-specific field WireTaskData must absorb


class MyState(vf.State):
    score: int = 0


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


def test_platform_preflight_redacts_credentials_without_reducing_review_data():
    provider_key = "sk-test-0123456789abcdefghijklmnopqrstuv"
    opaque_key = "opaque-judge-key-0123456789"
    payload = {
        "metadata": {"judgeApiKey": opaque_key},
        "completion": f"repeated {provider_key} and {opaque_key}",
        "answer": "reference answer",
        "rubric": "compare against the reference answer",
    }

    reduced, redactions = platform.prepare_upload(payload)

    assert redactions == 3
    assert provider_key not in json.dumps(reduced)
    assert opaque_key not in json.dumps(reduced)
    assert reduced["answer"] == payload["answer"]
    assert reduced["rubric"] == payload["rubric"]
    assert platform.prepare_upload(reduced)[1] == 0


def test_platform_preflight_finds_nested_and_properties_credentials():
    secret = "opaque-nested-secret-0123456789"
    token = "opaque-generic-token-0123456789"
    access_key = "opaque-aws-secret-access-key-0123456789"
    plural_key = "opaque-plural-key-0123456789"
    property_key = "opaque-property-key-0123456789"
    payload = {
        "APIKey": secret,
        "apiKeys": [plural_key],
        "awsSecretAccessKey": access_key,
        "credential": {"value": "pin"},
        "authentication": "opaque-authentication-0123456789",
        "rubric": "keep the pin label",
        "secret": {"value": secret},
        "token": token,
        "properties": {"password": {"type": "string", "value": property_key}},
        "schema": {
            "properties": {
                "password": {"type": ["string", "null"]},
                "apiKey": {"description": "typeless schema"},
            }
        },
    }

    reduced, _ = platform.prepare_upload(payload)

    assert reduced["APIKey"] == "[REDACTED]"
    assert reduced["apiKeys"] == ["[REDACTED]"]
    assert reduced["awsSecretAccessKey"] == "[REDACTED]"
    assert reduced["credential"]["value"] == "[REDACTED]"
    assert reduced["authentication"] == "[REDACTED]"
    assert reduced["rubric"] == payload["rubric"]
    assert reduced["secret"]["value"] == "[REDACTED]"
    assert reduced["token"] == "[REDACTED]"
    assert reduced["properties"]["password"]["value"] == "[REDACTED]"
    assert reduced["schema"] == payload["schema"]


def test_platform_preflight_redacts_schema_enums_and_sensitive_mapping_keys():
    secrets = ["opaque-map-key-0123456789", "second-map-key-0123456789"]
    schema_secret = "opaque-schema-secret-0123456789"
    reduced, _ = platform.prepare_upload(
        {
            "api_keys": {secret: {"owner": "alice"} for secret in secrets},
            "team_id": "team-0123456789",
            "schema": {
                "type": "object",
                "properties": {"api_key": {"type": "string", "enum": [schema_secret]}},
            },
        }
    )

    assert list(reduced["api_keys"]) == ["[REDACTED]", "[REDACTED 2]"]
    assert reduced["team_id"] == "team-0123456789"
    assert reduced["schema"]["properties"]["api_key"]["enum"] == ["[REDACTED]"]


def test_platform_preflight_finds_quoted_and_short_credentials():
    token = "opaque-json-token-0123456789"
    payload = {
        "completion": json.dumps({"Authorization": f"Bearer {token}"}),
        "error": "Authorization: Basic dXNlcjpwYXNz",
        "answer": "Use abc123 and token=version-123 for the example.",
        "password": "s3cr3t",
        "api_key": "abc123",
    }

    reduced, _ = platform.prepare_upload(payload)

    assert token not in reduced["completion"]
    assert "dXNlcjpwYXNz" not in reduced["error"]
    assert reduced["answer"] == payload["answer"]
    assert reduced["password"] == "[REDACTED]"
    assert reduced["api_key"] == "[REDACTED]"

    aws_secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    reduced, _ = platform.prepare_upload(
        {
            "completion": (
                'password="correct horse battery staple" '
                f'{{"aws_secret_access_key":"{aws_secret}"}}'
            )
        }
    )
    assert "correct horse battery staple" not in reduced["completion"]
    assert aws_secret not in reduced["completion"]

    azure_secret = "QWxhZGRpbjpvcGVuIHNlc2FtZQ=="
    reduced, _ = platform.prepare_upload(
        {"completion": f"AccountName=example;AccountKey={azure_secret}"}
    )
    assert azure_secret not in reduced["completion"]

    reduced, _ = platform.prepare_upload(
        {"password": 12345678, "token": 987654321, "secret": None, "auth": False}
    )
    assert reduced == {
        "password": "[REDACTED]",
        "token": "[REDACTED]",
        "secret": None,
        "auth": False,
    }
    usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    assert platform.prepare_upload({"usage": usage})[0]["usage"] == usage

    reduced, _ = platform.prepare_upload(
        {"completion": "runtime-secret-0123456789", "answer": "keep 1 and keep"},
        ["EMPTY"],
        [
            {
                "RUNTIME_SECRET": "runtime-secret-0123456789",
                "DEBUG": "1",
                "X-Custom": "keep",
            }
        ],
    )
    assert reduced == {"completion": "[REDACTED]", "answer": "keep 1 and keep"}


def test_platform_preflight_finds_auth_environment_and_header_tokens(monkeypatch):
    env_secret = "opaque-auth-env-secret-0123456789"
    header_secret = "opaque-auth-header-secret-0123456789"
    monkeypatch.setenv("AUTH", env_secret)

    reduced, _ = platform.prepare_upload(
        {"completion": f"{env_secret} {header_secret}"},
        secret_sources=[{"Authorization": f"Token {header_secret}"}],
    )

    assert reduced["completion"] == "[REDACTED] [REDACTED]"


def test_platform_preflight_redacts_pattern_discovered_secrets_everywhere():
    secret = "opaque-password-value-0123456789"
    reduced, _ = platform.prepare_upload(
        {"prompt": f"model repeated {secret}", "log": f"password={secret}"}
    )

    assert secret not in json.dumps(reduced)
    assert reduced["prompt"] == "model repeated [REDACTED]"


@pytest.mark.parametrize(
    "token",
    [
        "xwfp-0123456789-abcdefghijklmnop",
        "xapp-0123456789-abcdefghijklmnop",
        "rk_" + "live_" + "0" * 24,
        "sk_" + "test_" + "0" * 24,
    ],
)
def test_platform_preflight_finds_additional_provider_credentials(token):
    reduced, _ = platform.prepare_upload({"completion": token})

    assert reduced["completion"] == "[REDACTED]"


def test_platform_preflight_redacts_every_value_in_a_raw_cookie_header():
    session = "opaque-cookie-session-0123456789"
    refresh = "opaque-cookie-refresh-0123456789"
    header = json.dumps({"Cookie": f"session={session}; refresh={refresh}"})
    reduced, _ = platform.prepare_upload({"completion": f"request: {header}"})

    assert session not in reduced["completion"]
    assert refresh not in reduced["completion"]


def test_trace_push_runs_preflight_before_opening_the_network(monkeypatch, tmp_path):
    monkeypatch.setenv("FORWARDED_RUNTIME_SECRET", "forwarded-secret-0123456789")
    monkeypatch.setenv("AGENT_API_KEY", "agent-api-key-0123456789")
    monkeypatch.setenv("RUN_API_KEY", "run-api-key-0123456789")
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
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hello")),
    )
    trace.upload_secrets.append("rollout-capability-0123456789")
    episode = Episode(task=trace.task, traces=[trace])
    assert "rollout-capability" not in json.dumps(episode.to_record())
    assert vf.WireEpisode.model_validate(episode.model_dump()).traces[
        0
    ].upload_secrets == ["rollout-capability-0123456789"]
    (tmp_path / TRACES_FILE).touch()
    write_episode(tmp_path, episode)
    assert "rollout-capability" not in (tmp_path / TRACES_FILE).read_text()
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

    def stop_in_preflight(payload, known_secrets, secret_sources):
        assert payload["samples"]
        assert "rollout-capability" not in json.dumps(payload)
        assert {
            "prime-api-key",
            "agent-api-key-0123456789",
            "run-api-key-0123456789",
            "rollout-capability-0123456789",
        }.issubset(known_secrets)
        sources = {
            name: value for source in secret_sources for name, value in source.items()
        }
        assert sources["RUNTIME_SECRET"] == "runtime-secret-0123456789"
        assert sources["FORWARDED_RUNTIME_SECRET"] == "forwarded-secret-0123456789"
        assert sources["X-Auth"] in {
            "agent-header-secret-0123456789",
            "run-header-secret-0123456789",
        }
        raise RuntimeError("preflight stopped upload")

    monkeypatch.setattr(platform, "prepare_upload", stop_in_preflight)
    monkeypatch.setattr(
        platform,
        "credentials",
        lambda: (
            "prime-api-key",
            "https://api.example",
            "https://app.example",
            None,
        ),
    )
    monkeypatch.setattr(
        platform.httpx,
        "Client",
        lambda **_kwargs: pytest.fail("network opened before upload preflight"),
    )

    assert push_traces([episode], config, state) is None
    assert state.error == "RuntimeError: preflight stopped upload"
