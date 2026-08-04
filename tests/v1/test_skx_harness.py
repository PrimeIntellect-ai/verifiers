import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError
from tenacity import wait_none
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.skx import program
from verifiers.v1.harnesses.skx.harness import (
    _SUMMARIZER_PREFIX,
    _account_compaction_masking,
    _mask_summaries,
)
from verifiers.v1.interception.server import _effective_sampling
from verifiers.v1.types import (
    AssistantMessage,
    SamplingConfig,
    SystemMessage,
    UserMessage,
)


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


class _Trace:
    """The `Trace` surface `_account_compaction_masking` touches."""

    def __init__(self, nodes: list[MessageNode]) -> None:
        self.nodes = nodes
        self.metrics: dict[str, float] = {}

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = float(value)


def _summarizer_root() -> MessageNode:
    return MessageNode(
        message=SystemMessage(content=_SUMMARIZER_PREFIX + " summarize"),
        token_ids=[1],
        mask=[False],
    )


def _policy_root() -> MessageNode:
    return MessageNode(
        message=SystemMessage(content="ordinary policy"), token_ids=[9], mask=[False]
    )


def _untokenized_assistant(text: str, *, parent: int | None = None) -> MessageNode:
    """A turn as the eval relay commits it: message graph only, no token ids."""
    return MessageNode(
        parent=parent, message=AssistantMessage(content=text), sampled=True
    )


def _record(summary: str) -> dict:
    return {"summary": summary, "fallback": False, "finish_reason": "stop"}


@pytest.mark.parametrize("compactions", [1, 2])
def test_an_untokenized_trace_degrades_instead_of_killing_the_rollout(
    compactions,
) -> None:
    """The measured failure: summaries == branches == masked_nodes, zero masked tokens.

    An eval-relay trace commits every node with an empty mask, so a correct masking pass
    necessarily masks zero tokens. That used to raise and discard the whole rollout.
    """
    nodes = [_summarizer_root()]
    nodes += [
        _untokenized_assistant(f"summary {i}", parent=0) for i in range(compactions)
    ]
    nodes.append(_policy_root())
    nodes.append(_untokenized_assistant("policy turn", parent=len(nodes) - 1))
    trace = _Trace(nodes)

    _account_compaction_masking(
        trace, [_record(f"summary {i}") for i in range(compactions)]
    )

    assert trace.metrics["skx_compactions"] == compactions
    assert trace.metrics["skx_compaction_branches"] == compactions
    assert trace.metrics["skx_compaction_nodes_masked"] == compactions
    assert trace.metrics["skx_compaction_tokens_masked"] == 0
    assert trace.metrics["skx_compaction_tokenless"] == 1
    assert trace.metrics["skx_compaction_summaries_unmatched"] == 0
    assert trace.metrics["skx_compaction_masking_anomalies"] == 0
    assert trace.metrics["skx_compaction_summary_tokens_leaked"] == 0


def test_a_tokenized_rollout_still_masks_and_counts_summary_tokens() -> None:
    summary = _assistant("dense factual summary", parent=0)
    policy = _assistant("policy turn", parent=2)
    trace = _Trace([_summarizer_root(), summary, _policy_root(), policy])

    _account_compaction_masking(trace, [_record("dense factual summary")])

    assert summary.mask == [False, False, False]
    assert summary.logprobs == []
    assert policy.mask == [False, True, True]
    assert trace.metrics["skx_compaction_tokens_masked"] == 2
    assert trace.metrics["skx_compaction_tokenless"] == 0
    assert trace.metrics["skx_compaction_masking_anomalies"] == 0


def test_a_summary_missing_from_the_graph_is_recorded_not_fatal() -> None:
    """Two summaries generated, one summarizer branch present.

    A completion absent from the graph trains nothing, so the rollout survives with a
    counter rather than being thrown away.
    """
    summary = _assistant("first summary", parent=0)
    policy = _assistant("an ordinary policy turn", parent=2)
    trace = _Trace([_summarizer_root(), summary, _policy_root(), policy])

    _account_compaction_masking(
        trace, [_record("first summary"), _record("second summary")]
    )

    assert trace.metrics["skx_compactions"] == 2
    assert trace.metrics["skx_compaction_branches"] == 1
    assert trace.metrics["skx_compaction_summaries_unmatched"] == 1
    assert trace.metrics["skx_compaction_masking_anomalies"] == 1
    assert trace.metrics["skx_compaction_summary_tokens_leaked"] == 0
    assert policy.mask == [False, True, True]


def test_an_unmasked_summary_that_would_reach_the_loss_is_still_fatal() -> None:
    """The summary is in the graph and trainable, but off any recognized branch."""
    orphan = _assistant("dense factual summary", parent=0)
    trace = _Trace([_policy_root(), orphan])

    with pytest.raises(RuntimeError, match="unmasked_summary_tokens=2"):
        _account_compaction_masking(trace, [_record("dense factual summary")])

    assert trace.metrics["skx_compaction_summary_tokens_leaked"] == 2
    assert trace.metrics["skx_compaction_summaries_unmatched"] == 1
    # Refusing to text-match its way to a fix: the node is left exactly as found.
    assert orphan.mask == [False, True, True]


def test_the_compaction_bridge_is_not_mistaken_for_a_leak() -> None:
    """The bridge repeats the summary verbatim, but as an unsampled user message."""
    summary = _assistant("dense factual summary", parent=0)
    bridge = MessageNode(
        parent=2,
        message=UserMessage(
            content="Earlier context was compacted: dense factual summary"
        ),
        token_ids=[3, 4],
        mask=[False, False],
    )
    trace = _Trace([_summarizer_root(), summary, _policy_root(), bridge])

    _account_compaction_masking(
        trace, [_record("dense factual summary"), _record("dense factual summary")]
    )

    assert trace.metrics["skx_compaction_summaries_unmatched"] == 1
    assert trace.metrics["skx_compaction_summary_tokens_leaked"] == 0


def test_a_rollout_without_compaction_is_untouched() -> None:
    policy = _assistant("policy turn", parent=0)
    trace = _Trace([_policy_root(), policy])

    _account_compaction_masking(trace, [])

    assert policy.mask == [False, True, True]
    assert policy.logprobs == [-0.2, -0.3]
    assert trace.metrics == {
        "skx_compactions": 0.0,
        "skx_compaction_branches": 0.0,
        "skx_compaction_nodes_masked": 0.0,
        "skx_compaction_tokens_masked": 0.0,
        "skx_compaction_fallbacks": 0.0,
        "skx_compaction_truncations": 0.0,
        "skx_compaction_summaries_unmatched": 0.0,
        "skx_compaction_tokenless": 0.0,
        "skx_compaction_masking_anomalies": 0.0,
        "skx_compaction_summary_tokens_leaked": 0.0,
    }


def test_fallback_and_truncation_counters_survive_the_degraded_path() -> None:
    trace = _Trace([_summarizer_root(), _untokenized_assistant("s", parent=0)])

    _account_compaction_masking(
        trace,
        [
            {"summary": "s", "fallback": True, "finish_reason": "length"},
            {"summary": "t", "fallback": False, "finish_reason": "stop"},
        ],
    )

    assert trace.metrics["skx_compaction_fallbacks"] == 1
    assert trace.metrics["skx_compaction_truncations"] == 1
    assert trace.metrics["skx_compaction_summaries_unmatched"] == 1
    assert trace.metrics["skx_compaction_masking_anomalies"] == 1


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


@pytest.mark.asyncio
async def test_model_retry_exhaustion_is_bounded(monkeypatch) -> None:
    calls = 0

    async def create(**_kwargs):
        nonlocal calls
        calls += 1
        raise APIConnectionError(request=httpx.Request("POST", "http://model"))

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    stats = {"model_call_attempts": 0, "model_call_retries": 0}
    monkeypatch.setattr(program, "MODEL_RETRY_WAIT", wait_none())

    with pytest.raises(APIConnectionError):
        await program._create_with_retry(
            client, model="model", messages=[], stats=stats
        )
    assert calls == program.MODEL_CALL_ATTEMPTS
    assert stats == {"model_call_attempts": 4, "model_call_retries": 3}


@pytest.mark.asyncio
async def test_nonretryable_model_status_fails_immediately(monkeypatch) -> None:
    calls = 0

    async def create(**_kwargs):
        nonlocal calls
        calls += 1
        request = httpx.Request("POST", "http://model")
        response = httpx.Response(400, request=request)
        raise APIStatusError("bad request", response=response, body=None)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    stats = {"model_call_attempts": 0, "model_call_retries": 0}
    monkeypatch.setattr(program, "MODEL_RETRY_WAIT", wait_none())

    with pytest.raises(APIStatusError):
        await program._create_with_retry(
            client, model="model", messages=[], stats=stats
        )
    assert calls == 1
    assert stats == {"model_call_attempts": 1, "model_call_retries": 0}


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


def test_near_empty_compaction_does_not_start_the_cooldown() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "read"},
        {"role": "tool", "content": "brief"},
        {"role": "assistant", "content": "eval"},
        {"role": "tool", "content": "artifact handle"},
    ]

    assert program._compaction_ready(messages, keep_recent=2) is False
    messages.extend(
        [
            {"role": "assistant", "content": "aging read"},
            {"role": "tool", "content": "aging result"},
        ]
    )
    assert program._compaction_ready(messages, keep_recent=2) is True


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
    assert program.SUMMARIZER_MAX_TOKENS == 8192
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
async def test_compaction_bridge_preserves_latest_public_skx_state(tmp_path) -> None:
    async def create(**_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="The draft needs a repair."),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    candidate = "a" * 64
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "write candidate"},
        {"role": "tool", "content": "written"},
        {
            "role": "assistant",
            "content": "build candidate",
            "tool_calls": [
                {"function": {"name": "write", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "output": {
                        "candidate_sha256": candidate,
                        "build_passed": False,
                        "build_calls": 1,
                        "builds_remaining": 3,
                        "compile_error": "ModuleNotFoundError: No module named 'ATen'",
                        "diagnostics": {"eval_state": "compile"},
                    }
                }
            ),
        },
        {"role": "assistant", "content": "evaluate candidate"},
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "output": {
                        "artifacts": {"candidate": {"sha256": candidate}},
                        "passed": False,
                        "progress": {"attempt": 1, "current_state": "fail_correct"},
                        "diagnostics": {
                            "compile_diagnostic": {"passed": True},
                            "correctness_diagnostic": {
                                "passed": False,
                                "category": "shape_mismatch",
                                "runtime_type": "RuntimeError",
                            },
                        },
                    }
                }
            ),
        },
    ]

    compacted = await program.compact(
        client, "model", messages, 2, str(tmp_path / "compaction.jsonl")
    )

    bridge = compacted[2]["content"]
    assert f"evaluated_candidate_sha256={candidate}" in bridge
    assert "evaluation_state=fail_correct" in bridge
    assert "compilation_passed=True" in bridge
    assert "correctness_passed=False" in bridge
    assert "correctness_category=shape_mismatch" in bridge
    assert "runtime_type=RuntimeError" in bridge
    assert "current_candidate_evaluated=True" in bridge
    assert "Only evaluate after a candidate edit" in bridge


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


EVAL_OBSERVATION = json.dumps(
    {"is_error": False, "output": {"eval_state": "fail_compile", "eval_label": "compile"}}
)


def test_trusted_eval_matches_the_scored_envelope() -> None:
    assert program._is_trusted_eval(EVAL_OBSERVATION) is True
    # A build/status envelope carries no eval label, and a tool error is not evidence.
    assert program._is_trusted_eval(json.dumps({"is_error": False, "output": "built"})) is False
    assert (
        program._is_trusted_eval(
            json.dumps({"is_error": True, "output": {"eval_state": "pass"}})
        )
        is False
    )
    assert program._is_trusted_eval("Error executing tool bash: nope") is False


class _Message:
    """The minimal shape the program uses from a chat completion message."""

    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, **_kwargs) -> dict:
        return {"role": "assistant", "content": self.content}


def _eval_call() -> _Message:
    call = SimpleNamespace(
        id="call_0",
        function=SimpleNamespace(name="bash", arguments='{"command": "skx-eval"}'),
    )
    return _Message("running the evaluator", [call])


async def _run_program(monkeypatch, tmp_path, script: list[_Message], nudges: int) -> tuple[list, dict]:
    """Drive `program.main` over a scripted model, returning (requests, stats)."""
    requests: list[list[dict]] = []

    async def create(**kwargs):
        requests.append([dict(message) for message in kwargs["messages"]])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=script[len(requests) - 1])], usage=None
        )

    async def connect_mcp(_config):
        return [{"type": "function", "function": {"name": "bash"}}], {"bash": ("", "bash")}, {"": {}}

    monkeypatch.setattr(
        program,
        "AsyncOpenAI",
        lambda **_kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    monkeypatch.setattr(program, "connect_mcp", connect_mcp)
    monkeypatch.setattr(
        program, "call_mcp", lambda *_args, **_kwargs: _async(EVAL_OBSERVATION)
    )
    stats_file = tmp_path / "stats.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "program.py",
            "--base-url=http://model",
            "--api-key=secret",
            "--model=model",
            "--prompt=optimize the kernel",
            '--mcp-config={"mcpServers": {"": {"url": "http://mcp"}}}',
            f"--stats-file={stats_file}",
            f"--eval-nudges={nudges}",
        ],
    )
    await program.main()
    return requests, json.loads(stats_file.read_text())


async def _async(value):
    return value


@pytest.mark.asyncio
async def test_finishing_without_an_eval_is_answered_not_accepted(monkeypatch, tmp_path) -> None:
    script = [_Message("done, the kernel looks right"), _eval_call(), _Message("done")]

    requests, stats = await _run_program(monkeypatch, tmp_path, script, nudges=2)

    assert len(requests) == 3  # the premature finish bought a corrective turn
    assert requests[1][-1] == {"role": "user", "content": program.EVAL_NUDGE}
    assert stats["eval_nudges"] == 1
    assert stats["trusted_evals"] == 1
    # The eval satisfied the contract, so the second finish is accepted as-is.
    assert requests[2][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_eval_nudges_are_bounded(monkeypatch, tmp_path) -> None:
    script = [_Message("done") for _ in range(6)]

    requests, stats = await _run_program(monkeypatch, tmp_path, script, nudges=2)

    assert len(requests) == 3
    assert stats["eval_nudges"] == 2
    assert stats["trusted_evals"] == 0


@pytest.mark.asyncio
async def test_an_evaluated_rollout_is_never_nudged(monkeypatch, tmp_path) -> None:
    script = [_eval_call(), _Message("done")]

    requests, stats = await _run_program(monkeypatch, tmp_path, script, nudges=2)

    assert len(requests) == 2
    assert stats == {
        "repeat_tool_calls": 0,
        "deduped_observations": 0,
        "model_call_attempts": 2,
        "model_call_retries": 0,
        "trusted_evals": 1,
        "eval_nudges": 0,
    }


@pytest.mark.asyncio
async def test_zero_nudges_restores_the_previous_contract(monkeypatch, tmp_path) -> None:
    script = [_Message("done")]

    requests, stats = await _run_program(monkeypatch, tmp_path, script, nudges=0)

    assert len(requests) == 1
    assert stats["eval_nudges"] == 0
