import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError
from tenacity import wait_none
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.skx import program
from verifiers.v1.harnesses.skx.harness import _SUMMARIZER_PREFIX, _mask_summaries
from verifiers.v1.interception.server import _effective_sampling
from verifiers.v1.types import AssistantMessage, SamplingConfig, SystemMessage


def _assistant(text: str, *, parent: int) -> MessageNode:
    return MessageNode(
        parent=parent,
        message=AssistantMessage(content=text),
        sampled=True,
        token_ids=[1, 2, 3],
        mask=[False, True, True],
        logprobs=[-0.2, -0.3],
    )


def test_summary_masking_uses_branch_identity_not_text() -> None:
    summary_root = MessageNode(
        message=SystemMessage(content=_SUMMARIZER_PREFIX + " summarize"),
        token_ids=[1],
        mask=[False],
    )
    summary = _assistant("same text", parent=0)
    policy_root = MessageNode(
        message=SystemMessage(content="ordinary policy"),
        token_ids=[4],
        mask=[False],
    )
    policy = _assistant("same text", parent=2)
    trace = SimpleNamespace(nodes=[summary_root, summary, policy_root, policy])

    nodes, tokens, branches = _mask_summaries(trace)

    assert (nodes, tokens, branches) == (1, 2, 1)
    assert summary.mask == [False, False, False]
    assert summary.logprobs == []
    assert policy.mask == [False, True, True]


def test_multiple_summaries_can_share_one_deduplicated_root() -> None:
    root = MessageNode(
        message=SystemMessage(content=_SUMMARIZER_PREFIX + " summarize"),
        token_ids=[1],
        mask=[False],
    )
    first = _assistant("first", parent=0)
    second = _assistant("second", parent=0)
    trace = SimpleNamespace(nodes=[root, first, second])

    nodes, tokens, branches = _mask_summaries(trace)

    assert (nodes, tokens, branches) == (2, 4, 2)
    assert first.mask == second.mask == [False, False, False]


@pytest.mark.asyncio
async def test_chat_wrapper_has_no_second_retry_layer() -> None:
    calls = 0

    async def create(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("one SDK call")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with pytest.raises(RuntimeError, match="one SDK call"):
        await program._create_with_retry(client, model="model", messages=[])
    assert calls == 1


@pytest.mark.asyncio
async def test_model_retry_attempts_are_counted(monkeypatch) -> None:
    calls = 0

    async def create(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise APIConnectionError(request=httpx.Request("POST", "http://model"))
        return "completion"

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    stats = {"model_call_attempts": 0, "model_call_retries": 0}
    snapshots = []
    monkeypatch.setattr(program, "MODEL_RETRY_WAIT", wait_none())

    result = await program._create_with_retry(
        client,
        model="model",
        messages=[],
        stats=stats,
        stats_updated=lambda: snapshots.append(dict(stats)),
    )

    assert result == "completion"
    assert stats == {"model_call_attempts": 2, "model_call_retries": 1}
    assert snapshots == [
        {"model_call_attempts": 1, "model_call_retries": 0},
        {"model_call_attempts": 2, "model_call_retries": 1},
    ]


def test_auxiliary_sampling_can_only_tighten_the_token_cap() -> None:
    configured = SamplingConfig(max_tokens=8192, temperature=1.0, top_p=0.95)
    body = {"max_completion_tokens": 2048, "temperature": 0}

    ordinary = _effective_sampling(
        ChatDialect(), body, configured, auxiliary=False
    )
    auxiliary = _effective_sampling(
        ChatDialect(), body, configured, auxiliary=True
    )
    attempted_raise = _effective_sampling(
        ChatDialect(), {"max_tokens": 16384}, configured, auxiliary=True
    )

    assert ordinary is configured
    assert auxiliary.max_tokens == 2048
    assert auxiliary.temperature == 0
    assert auxiliary.top_p == 0.95
    assert attempted_raise.max_tokens == 8192


@pytest.mark.asyncio
async def test_compaction_falls_back_after_a_truncated_summary(tmp_path) -> None:
    request = {}

    async def create(**kwargs):
        request.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content="continued task solution"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "read"},
        {
            "role": "tool",
            "content": (
                "/workspace/.skx_artifacts/evaluations/sha256-"
                + "a" * 64
                + ".json\nSHA-256:"
                + "b" * 64
            ),
        },
        {"role": "assistant", "content": "edit"},
        {"role": "tool", "content": "edited"},
        {"role": "assistant", "content": "eval"},
        {"role": "tool", "content": "failed"},
    ]
    tracker = tmp_path / "compaction.jsonl"

    compacted = await program.compact(client, "model", messages, 2, str(tracker))

    assert request["max_completion_tokens"] == program.SUMMARIZER_MAX_TOKENS
    assert request["temperature"] == 0
    assert request["extra_headers"] == {program.AUXILIARY_SAMPLING_HEADER: "1"}
    assert "/workspace/.skx_artifacts/" in request["messages"][0]["content"]
    assert "SHA-256 handle verbatim" in request["messages"][0]["content"]
    assert "/workspace/.skx_artifacts/evaluations/sha256-" + "a" * 64 + ".json" in request["messages"][1]["content"]
    assert "SHA-256:" + "b" * 64 in request["messages"][1]["content"]
    assert program.FALLBACK_SUMMARY in compacted[2]["content"]
    record = json.loads(tracker.read_text())
    assert record == {
        "summary": program.FALLBACK_SUMMARY,
        "fallback": True,
        "finish_reason": "length",
    }


@pytest.mark.asyncio
async def test_mutating_mcp_call_is_not_replayed(monkeypatch) -> None:
    calls = 0

    class Session:
        async def call_tool(self, _name, _arguments):
            nonlocal calls
            calls += 1
            raise ConnectionError("response lost after dispatch")

    @asynccontextmanager
    async def session(_spec):
        yield Session()

    monkeypatch.setattr(program, "mcp_session", session)
    with pytest.raises(ConnectionError, match="response lost"):
        await program.call_mcp(
            {"": {"url": "http://mcp"}}, {"edit": ("", "edit")}, "edit", {}
        )
    assert calls == 1


def test_observation_key_accepts_canonicalized_arguments() -> None:
    cache = {}
    first = program._dedup_observation(cache, "read", '{"a":1,"b":2}', "x" * 1600)
    second = program._dedup_observation(cache, "read", '{"a":1,"b":2}', "x" * 1600)
    assert first[1:] == (False, False)
    assert second[1:] == (True, True)


def test_repeat_eval_is_keyed_by_candidate_hash_and_workspace_revision() -> None:
    cache = {}
    hash_a = "a" * 64
    hash_b = "b" * 64
    first = program._dedup_observation(
        cache,
        "bash",
        '{"command":"skx-eval"}',
        f'{{"candidate_sha256":"{hash_a}"}}',
        workspace_revision=0,
    )
    repeated = program._dedup_observation(
        cache,
        "bash",
        '{"command":"skx-eval"}',
        f'{{"candidate_sha256":"{hash_a}"}}',
        workspace_revision=0,
    )
    edited = program._dedup_observation(
        cache,
        "bash",
        '{"command":"skx-eval"}',
        f'{{"candidate_sha256":"{hash_b}"}}',
        workspace_revision=1,
    )

    assert first[2] is False
    assert repeated[2] is True
    assert edited[2] is False
    assert program._candidate_sha256(edited[0]) == hash_b
    assert program._is_eval_call("bash", '{"command":"skx-eval"}') is True
    assert program._mutates_workspace("skx_edit") is True


def test_workspace_revision_only_advances_after_successful_mutation() -> None:
    success = json.dumps({"is_error": False, "output": "Successfully edited candidate.py"})
    failure = json.dumps({"is_error": True, "output": "oldText did not match"})

    assert program._successful_workspace_mutation("edit", success) is True
    assert program._successful_workspace_mutation("edit", failure) is False
    assert program._successful_workspace_mutation("read", success) is False
