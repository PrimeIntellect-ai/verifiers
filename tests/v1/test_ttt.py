"""The TTT hook: fork detection, update payloads, version stamping, cache salting, and
branch version uniformity — all against a stubbed TTT service (httpx transport), no GPU."""

import httpx
import pytest
import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.errors import TTTError
from verifiers.v1.ttt import TTTConfig, TTTRolloutHook
from verifiers.v1.types import AssistantMessage, SamplingConfig, TurnTokens, UserMessage


def response(content: str, prompt_ids: list[int], completion_ids: list[int]) -> vf.Response:
    return vf.Response(
        id="",
        created=0,
        model="m",
        message=AssistantMessage(content=content),
        finish_reason="stop",
        tokens=TurnTokens(prompt_ids=prompt_ids, completion_ids=completion_ids),
    )


class FakeService:
    """Collects /update and /release calls; returns the echoing version."""

    def __init__(self, fail: bool = False, wrong_version: bool = False):
        self.updates: list[dict] = []
        self.releases: list[dict] = []
        self.fail = fail
        self.wrong_version = wrong_version

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = None if not request.content else __import__("json").loads(request.content)
        if request.url.path == "/update":
            if self.fail:
                return httpx.Response(500, json={"detail": "boom"})
            self.updates.append(payload)
            version = 0 if self.wrong_version else payload["seq_no"]
            return httpx.Response(
                200,
                json={"version": version, "loss": 1.5, "ckpt_path": f"/ckpt/v{version}"},
            )
        if request.url.path == "/release":
            self.releases.append(payload)
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)


def hook_with(trace: vf.Trace, service: FakeService, **cfg) -> TTTRolloutHook:
    hook = TTTRolloutHook(TTTConfig(base_url="http://ttt", **cfg), trace)
    hook._http = httpx.AsyncClient(transport=httpx.MockTransport(service.handler))
    return hook


def commit_turn(trace: vf.Trace, hook: TTTRolloutHook, prompt, resp) -> graph.PendingTurn:
    """One committed turn with the hook's stamping, as the interception server drives it."""
    turn = graph.prepare_turn(trace, prompt)
    before = len(trace.nodes)
    turn.commit(resp)
    hook.after_commit(trace, before)
    return turn


async def test_linear_turns_never_update():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    hook = hook_with(trace, service)
    u1 = UserMessage(content="u1")
    commit_turn(trace, hook, [u1], response("a1", [1, 2], [3]))
    a1 = trace.nodes[-1].message
    # Next turn extends the leaf (same branch): no update.
    turn = graph.prepare_turn(trace, [u1, a1, UserMessage(content="u2")])
    await hook.on_turn_prepared(turn)
    assert service.updates == []
    assert hook.version == 0
    assert hook.model_override is None


async def test_fork_triggers_update_and_switches_model():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    hook = hook_with(trace, service)
    u1 = UserMessage(content="u1")
    commit_turn(trace, hook, [u1], response("a1", [1, 2], [3]))
    # Compaction: a fresh prompt that shares nothing → fork.
    turn = graph.prepare_turn(trace, [UserMessage(content="summary")])
    await hook.on_turn_prepared(turn)

    (update,) = service.updates
    assert update["rollout_id"] == trace.id
    assert update["adapter_name"] == f"ttt-{trace.id}"
    assert update["seq_no"] == 1
    # The abandoned branch: u1's prompt tokens + a1's completion, ALL in loss (scope=all).
    assert update["token_ids"] == [1, 2, 3]
    assert update["loss_mask"] == [True, True, True]

    assert hook.version == 1
    assert hook.model_override == f"ttt-{trace.id}"
    assert trace.info["ttt"]["updates"][0]["version"] == 1
    assert trace.info["ttt"]["updates"][0]["loss"] == 1.5

    # Commit the new branch's turn: its nodes are stamped with version 1.
    commit_turn(trace, hook, turn.prompt, response("a2", [7, 8], [9]))
    assert {n.ttt_version for n in trace.nodes[-2:]} == {1}
    # The first branch stays version 0.
    branches = trace.branches
    assert sorted(b.ttt_version for b in branches) == [0, 1]


async def test_shared_prefix_is_context_not_loss():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    hook = hook_with(trace, service)
    sys_msg = vf.SystemMessage(content="sys")
    u1 = UserMessage(content="u1")
    commit_turn(trace, hook, [sys_msg, u1], response("a1", [10, 20, 30], [40]))
    # Compaction keeps the system message: it lands in the new prompt's reused prefix.
    turn = graph.prepare_turn(trace, [sys_msg, UserMessage(content="summary")])
    assert turn.prefix_node_ids  # shares the system node
    await hook.on_turn_prepared(turn)
    (update,) = service.updates
    # The system node's token(s) ride as context (False); the rest is loss.
    node_sys = trace.nodes[0]
    n_sys = len(node_sys.token_ids)
    assert update["loss_mask"][:n_sys] == [False] * n_sys
    assert all(update["loss_mask"][n_sys:])


async def test_sampled_scope_masks_input_tokens():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    hook = hook_with(trace, service, loss_scope="sampled")
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1, 2], [3, 4]))
    turn = graph.prepare_turn(trace, [UserMessage(content="summary")])
    await hook.on_turn_prepared(turn)
    (update,) = service.updates
    assert update["token_ids"] == [1, 2, 3, 4]
    assert update["loss_mask"] == [False, False, True, True]


async def test_second_fork_trains_only_new_content():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    hook = hook_with(trace, service)
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1], [2]))
    turn = graph.prepare_turn(trace, [UserMessage(content="s1")])
    await hook.on_turn_prepared(turn)
    commit_turn(trace, hook, turn.prompt, response("a2", [5], [6]))
    turn2 = graph.prepare_turn(trace, [UserMessage(content="s2")])
    await hook.on_turn_prepared(turn2)
    assert len(service.updates) == 2
    second = service.updates[1]
    # Second update trains the second branch only; the first branch's nodes are `seen`.
    assert second["token_ids"] == [5, 6]
    assert second["loss_mask"] == [True, True]
    assert second["seq_no"] == 2
    assert hook.version == 2


async def test_service_failure_is_ttt_error():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    hook = hook_with(trace, FakeService(fail=True))
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1], [2]))
    turn = graph.prepare_turn(trace, [UserMessage(content="s")])
    with pytest.raises(TTTError, match="update 1 failed"):
        await hook.on_turn_prepared(turn)


async def test_version_mismatch_is_ttt_error():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    hook = hook_with(trace, FakeService(wrong_version=True))
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1], [2]))
    turn = graph.prepare_turn(trace, [UserMessage(content="s")])
    with pytest.raises(TTTError, match="out of sync"):
        await hook.on_turn_prepared(turn)


async def test_missing_token_ids_is_ttt_error():
    """An eval-relay trace (no token ids) can't feed TTT — fail loudly, not silently."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    hook = hook_with(trace, FakeService())
    turn = graph.prepare_turn(trace, [UserMessage(content="u1")])
    turn.commit(
        vf.Response(
            id="",
            created=0,
            model="m",
            message=AssistantMessage(content="a1"),
            finish_reason="stop",
        )
    )
    hook.after_commit(trace, 0)
    fork = graph.prepare_turn(trace, [UserMessage(content="s")])
    with pytest.raises(TTTError, match="renderer"):
        await hook.on_turn_prepared(fork)


async def test_release_called_and_never_raises():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    hook = hook_with(trace, service)
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1], [2]))
    turn = graph.prepare_turn(trace, [UserMessage(content="s")])
    await hook.on_turn_prepared(turn)
    await hook.aclose()
    (release,) = service.releases
    assert release == {"rollout_id": trace.id, "adapter_name": f"ttt-{trace.id}"}

    # A hook that never updated (and never opened a client) skips the call entirely.
    trace2 = vf.Trace(task=vf.Task(idx=1, prompt="t"))
    hook2 = TTTRolloutHook(TTTConfig(base_url="http://nowhere.invalid"), trace2)
    await hook2.aclose()  # no client, no update → no-op, no raise


async def test_final_branch_update_config():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    hook = hook_with(trace, service, train_final_branch=True)
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1], [2]))
    await hook.finalize_rollout()
    (update,) = service.updates
    assert update["token_ids"] == [1, 2]

    # Default off: no update.
    trace2 = vf.Trace(task=vf.Task(idx=1, prompt="t"))
    service2 = FakeService()
    hook2 = hook_with(trace2, service2)
    commit_turn(trace2, hook2, [UserMessage(content="u1")], response("a1", [1], [2]))
    await hook2.finalize_rollout()
    assert service2.updates == []


def test_salted_sampling_per_version():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    hook = TTTRolloutHook(TTTConfig(base_url="http://x"), trace)
    sampling = SamplingConfig(temperature=0.7, extra_body={"cache_salt": "policy-3"})
    # Version 0 (base model): untouched.
    assert hook.salted_sampling(sampling) is sampling
    hook.version = 2
    salted = hook.salted_sampling(sampling)
    assert salted.model_dump(exclude_none=True)["extra_body"]["cache_salt"] == (f"policy-3:ttt-{trace.id}-v2")
    # No pre-existing salt: the TTT salt stands alone.
    salted2 = hook.salted_sampling(SamplingConfig())
    assert salted2.model_dump(exclude_none=True)["extra_body"]["cache_salt"] == (f"ttt-{trace.id}-v2")
    # The original sampling is untouched.
    assert sampling.model_dump(exclude_none=True)["extra_body"] == {"cache_salt": "policy-3"}


def test_branch_ttt_version_uniformity_enforced():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    u1 = UserMessage(content="u1")
    turn = graph.prepare_turn(trace, [u1])
    turn.commit(response("a1", [1], [2]))
    trace.nodes[-1].ttt_version = 0
    a1 = trace.nodes[-1].message
    turn2 = graph.prepare_turn(trace, [u1, a1, UserMessage(content="u2")])
    turn2.commit(response("a2", [1, 2, 3, 4], [5]))
    # Corrupt: same branch, different version on the second sampled node.
    trace.nodes[-1].ttt_version = 1
    (branch,) = trace.branches
    with pytest.raises(ValueError, match="multiple TTT adapter versions"):
        _ = branch.ttt_version


# -- RolloutSession integration (model/sampling switching) ---------------------------------


def test_session_turn_model_and_sampling_switch():
    """The interception server samples each turn from `session.turn_model()` with
    `session.turn_sampling()`: base model + untouched sampling before any update, the
    adapter + salted sampling after."""
    from verifiers.v1.interception import RolloutSession

    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    hook = TTTRolloutHook(TTTConfig(base_url="http://x"), trace)
    ctx = vf.RolloutContext(model="base-model", client=object(), sampling=SamplingConfig(temperature=0.5))
    session = RolloutSession(ctx, trace, ttt=hook)

    assert session.turn_model() == "base-model"
    assert session.turn_sampling() is ctx.sampling

    hook.version = 1
    hook.model_override = hook.adapter_name
    assert session.turn_model() == hook.adapter_name
    salted = session.turn_sampling()
    assert salted.model_dump(exclude_none=True)["extra_body"]["cache_salt"] == (f"{hook.adapter_name}-v1")
    assert salted.temperature == 0.5

    # No hook: plain passthrough.
    session2 = RolloutSession(ctx, trace)
    assert session2.turn_model() == "base-model"
    assert session2.turn_sampling() is ctx.sampling


# -- interception-server path (end-to-end over HTTP, fake client + fake service) -----------


class TokenClient:
    """A fake token-carrying model client (the renderer stand-in): scripted responses with
    deterministic token ids. Records the (model, sampling) each call used."""

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[tuple[str, dict]] = []
        self.counter = 0

    async def get_response(self, dialect, body, model, sampling, headers=None, session_id=None, turn=None):
        self.calls.append((model, sampling.model_dump(exclude_none=True)))
        content = self.script.pop(0)
        base = 1000 + self.counter * 100
        self.counter += 1
        # Token-consistent like the real renderer bridge: the prompt reproduces the reused
        # prefix's stored tokens verbatim, then fresh ids for the tail.
        prefix_ids = [tok for nid in turn.prefix_node_ids for tok in turn.trace.nodes[nid].token_ids]
        tail_len = len(turn.tail) * 3 + 1  # a few ids per new message + generation prompt
        response = vf.Response(
            id=f"r{self.counter}",
            created=0,
            model=model,
            message=AssistantMessage(content=content),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[*prefix_ids, *range(base, base + tail_len)],
                completion_ids=list(range(base + 50, base + 53)),
            ),
        )
        response.raw = {
            "id": response.id,
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        return response

    async def close(self):
        pass


async def test_interception_server_drives_ttt_on_compaction():
    """The full server path: an append turn (no update), then a compaction-style rewrite —
    the hook updates on the abandoned branch, the next call samples from the adapter with a
    salted cache, and the committed nodes carry the right versions."""
    from aiohttp import ClientSession
    from verifiers.v1.interception import InterceptionServer, RolloutSession

    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    hook = hook_with(trace, service)
    client = TokenClient(["a1", "a2", "a3"])
    ctx = vf.RolloutContext(model="base", client=client, sampling=SamplingConfig())
    session = RolloutSession(ctx, trace, ttt=hook)

    async with InterceptionServer() as server:
        secret = server.register(session)
        url = f"http://127.0.0.1:{server.port}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {secret}"}
        async with ClientSession() as http:
            # Turn 1: base model.
            r1 = await http.post(
                url,
                json={"messages": [{"role": "user", "content": "u1"}]},
                headers=headers,
            )
            assert r1.status == 200
            a1 = (await r1.json())["choices"][0]["message"]["content"]
            # Turn 2 extends turn 1: still no update.
            r2 = await http.post(
                url,
                json={
                    "messages": [
                        {"role": "user", "content": "u1"},
                        {"role": "assistant", "content": a1},
                        {"role": "user", "content": "u2"},
                    ]
                },
                headers=headers,
            )
            assert r2.status == 200
            assert service.updates == []
            # Turn 3 rewrites the context (compaction): update fires before it samples.
            r3 = await http.post(
                url,
                json={"messages": [{"role": "user", "content": "summary"}]},
                headers=headers,
            )
            assert r3.status == 200

    (update,) = service.updates
    assert update["seq_no"] == 1
    assert len(update["token_ids"]) == len(update["loss_mask"])
    assert all(update["loss_mask"])  # nothing shared, scope=all

    # The first two calls used the base model; the post-update call used the adapter with
    # a version-salted cache.
    assert [model for model, _ in client.calls[:2]] == ["base", "base"]
    model3, sampling3 = client.calls[2]
    assert model3 == hook.adapter_name
    assert sampling3["extra_body"]["cache_salt"] == f"{hook.adapter_name}-v1"

    # Stamps: the first branch's nodes are v0, the compacted branch's are v1.
    assert sorted(b.ttt_version for b in trace.branches) == [0, 1]


async def test_interception_server_surfaces_ttt_failure():
    """A failed update is stashed as the rollout's real error and returned to the harness
    as a non-retryable 400."""
    from aiohttp import ClientSession
    from verifiers.v1.interception import InterceptionServer, RolloutSession

    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    hook = hook_with(trace, FakeService(fail=True))
    client = TokenClient(["a1", "a2"])
    ctx = vf.RolloutContext(model="base", client=client, sampling=SamplingConfig())
    session = RolloutSession(ctx, trace, ttt=hook)

    async with InterceptionServer() as server:
        secret = server.register(session)
        url = f"http://127.0.0.1:{server.port}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {secret}"}
        async with ClientSession() as http:
            r1 = await http.post(
                url,
                json={"messages": [{"role": "user", "content": "u1"}]},
                headers=headers,
            )
            assert r1.status == 200
            r2 = await http.post(
                url,
                json={"messages": [{"role": "user", "content": "rewrite"}]},
                headers=headers,
            )
            assert r2.status == 400  # non-retryable

    assert isinstance(session.error, TTTError)


# -- Q&A at compaction ----------------------------------------------------------------------


class QAClient:
    """A fake rollout client for Q&A generations: returns a canned answer per question and
    records every request body + (model, sampling)."""

    def __init__(self, answer="the answer"):
        self.answer = answer
        self.requests: list[tuple[dict, str, dict]] = []

    async def get_response(self, dialect, body, model, sampling, headers=None, session_id=None, turn=None):
        self.requests.append((body, model, sampling.model_dump(exclude_none=True)))
        return vf.Response(
            id="qa",
            created=0,
            model=model,
            message=AssistantMessage(content=self.answer),
            finish_reason="stop",
        )

    async def close(self):
        pass


def qa_hook(trace, service, client, **qa_overrides):
    config = TTTConfig(
        base_url="http://ttt",
        qa={"num_pairs": 3, "prompts": ["knowledge?", "what worked?"], **qa_overrides},
    )
    ctx = vf.RolloutContext(model="base", client=client, sampling=SamplingConfig(temperature=0.9))
    hook = TTTRolloutHook(config, trace, ctx=ctx)
    hook._http = httpx.AsyncClient(transport=httpx.MockTransport(service.handler))
    return hook


async def test_qa_generated_with_branch_in_context_and_shipped():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    client = QAClient(answer="learned fact")
    hook = qa_hook(trace, service, client)
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1, 2], [3]))
    turn = graph.prepare_turn(trace, [UserMessage(content="summary")])
    await hook.on_turn_prepared(turn)

    # 3 QA generations (prompts cycled), each with the FULL abandoned branch in context.
    assert len(client.requests) == 3
    questions = [body["messages"][-1]["content"] for body, _, _ in client.requests]
    assert questions == ["knowledge?", "what worked?", "knowledge?"]
    for body, model, sampling in client.requests:
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["user", "assistant", "user"]  # u1, a1, question
        assert body["messages"][0]["content"] == "u1"
        assert model == "base"  # version 0 at generation time — the branch's own model
        assert sampling["max_tokens"] == 1024  # the QA budget, not the rollout's

    # The update shipped the pairs and trained QA-only by default.
    (update,) = service.updates
    assert update["qa_pairs"] == [{"question": q, "answer": "learned fact"} for q in questions]
    assert update["train_rollout"] is False
    # And the trace records them.
    record = trace.info["ttt"]["updates"][0]
    assert record["num_qa_pairs"] == 3
    assert record["trained_rollout"] is False


async def test_qa_also_train_rollout():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    hook = qa_hook(trace, service, QAClient(), also_train_rollout=True)
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1], [2]))
    turn = graph.prepare_turn(trace, [UserMessage(content="s")])
    await hook.on_turn_prepared(turn)
    (update,) = service.updates
    assert update["train_rollout"] is True
    assert update["token_ids"] == [1, 2]


async def test_qa_all_empty_answers_fails_loudly():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    hook = qa_hook(trace, FakeService(), QAClient(answer="   "))
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1], [2]))
    turn = graph.prepare_turn(trace, [UserMessage(content="s")])
    with pytest.raises(TTTError, match="no usable pairs"):
        await hook.on_turn_prepared(turn)


async def test_qa_generation_failure_is_ttt_error():
    class BrokenClient(QAClient):
        async def get_response(self, *args, **kwargs):
            raise RuntimeError("model down")

    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    hook = qa_hook(trace, FakeService(), BrokenClient())
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1], [2]))
    turn = graph.prepare_turn(trace, [UserMessage(content="s")])
    with pytest.raises(TTTError, match="qa generation failed"):
        await hook.on_turn_prepared(turn)


def test_qa_requires_ctx():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    with pytest.raises(ValueError, match="needs the rollout context"):
        TTTRolloutHook(TTTConfig(base_url="http://x", qa={}), trace)


async def test_qa_generations_never_touch_the_trace():
    """QA exchanges are housekeeping: the trace's node count is unchanged by generation."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    hook = qa_hook(trace, service, QAClient())
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1], [2]))
    nodes_before = len(trace.nodes)
    turn = graph.prepare_turn(trace, [UserMessage(content="s")])
    await hook.on_turn_prepared(turn)
    assert len(trace.nodes) == nodes_before
