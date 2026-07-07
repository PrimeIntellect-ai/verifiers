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


def item_text(pairs: list[dict]) -> str:
    return "\n".join(
        f"<item><type>{p.get('type', 'qa')}</type><question>{p['question']}</question>"
        f"<answer>{p['answer']}</answer></item>"
        for p in pairs
    )


class QAClient:
    """A fake rollout client for Q&A generations: returns a canned structured generation
    (with token ids, like the renderer) and records every request body + (model, sampling)."""

    def __init__(self, items: list[dict] | None = None):
        self.items = items if items is not None else [{"question": "What is x for task T?", "answer": "42"}]
        self.requests: list[tuple[dict, str, dict]] = []
        self.counter = 0

    def content(self) -> str:
        return item_text(self.items)

    async def get_response(self, dialect, body, model, sampling, headers=None, session_id=None, turn=None):
        self.requests.append((body, model, sampling.model_dump(exclude_none=True)))
        base = 5000 + self.counter * 100
        self.counter += 1
        prefix_ids = [tok for nid in turn.prefix_node_ids for tok in turn.trace.nodes[nid].token_ids] if turn else []
        tail_len = (len(turn.tail) if turn else 1) * 2 + 1
        return vf.Response(
            id="qa",
            created=0,
            model=model,
            message=AssistantMessage(content=self.content()),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=[*prefix_ids, *range(base, base + tail_len)],
                completion_ids=list(range(base + 50, base + 53)),
            ),
        )

    async def close(self):
        pass


def qa_hook(trace, service, client, **qa_overrides):
    config = TTTConfig(
        base_url="http://ttt",
        qa={"num_generations": 2, "items_per_generation": 2, "seeds": ["facts", "lessons"], **qa_overrides},
    )
    ctx = vf.RolloutContext(model="base", client=client, sampling=SamplingConfig(temperature=0.9))
    hook = TTTRolloutHook(config, trace, ctx=ctx)
    hook._http = httpx.AsyncClient(transport=httpx.MockTransport(service.handler))
    return hook


async def test_qa_generated_with_branch_in_context_and_shipped():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    client = QAClient(
        items=[
            {"type": "qa", "question": "What is x for task T?", "answer": "42"},
            {"type": "lesson", "question": "When tool S rate-limits, what should you do?", "answer": "back off"},
        ]
    )
    hook = qa_hook(trace, service, client)
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1, 2], [3]))
    turn = graph.prepare_turn(trace, [UserMessage(content="summary")])
    await hook.on_turn_prepared(turn)

    # 2 seeded generations, each with the FULL abandoned branch in context + the seeded
    # generation instruction as the final user message.
    assert len(client.requests) == 2
    for i, (body, model, sampling) in enumerate(client.requests):
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["user", "assistant", "user"]  # u1, a1, instruction
        assert body["messages"][0]["content"] == "u1"
        instruction = body["messages"][-1]["content"]
        assert ["facts", "lessons"][i] in instruction  # the seed
        assert "2 question-answer items" in instruction  # items_per_generation
        assert "SELF-CONTAINED" in instruction
        assert model == "base"  # version 0 at generation time — the branch's own model
        assert sampling["max_tokens"] == 2048  # the QA budget, not the rollout's

    # The update shipped the extracted items (2 per generation, deduped to 2 distinct).
    (update,) = service.updates
    assert update["qa_pairs"] == client.items  # both generations' items dedup to the two
    assert update["train_rollout"] is False
    record = trace.info["ttt"]["updates"][0]
    assert record["num_qa_pairs"] == 2
    assert record["trained_rollout"] is False


async def test_qa_exchanges_become_tagged_branches():
    """QA generations are committed to the trace as ttt_qa branches (RL trains them), but
    stay out of the rollout's turn/token accounting."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    hook = qa_hook(trace, service, QAClient())
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1, 2], [3]))
    turns_before = trace.num_turns
    tokens_before = trace.num_total_tokens
    turn = graph.prepare_turn(trace, [UserMessage(content="summary")])
    await hook.on_turn_prepared(turn)

    qa_branches = [b for b in trace.branches if b.is_ttt_qa]
    assert len(qa_branches) == 2  # one per generation
    for branch in qa_branches:
        # The QA branch extends the abandoned branch: its prefix is the trajectory, its
        # tail the instruction + generation, sampled under the PRE-update version (0).
        assert branch.ttt_version == 0
        assert any(n.ttt_qa and n.sampled for n in branch.nodes)
    # The abandoned branch's leaf now has QA children, so the graph's full branch view
    # shows the QA branches; the main view (metrics) still sees the trajectory itself.

    # Rollout accounting is unchanged by QA: turns and token totals exclude tagged nodes.
    assert trace.num_turns == turns_before
    assert trace.num_total_tokens == tokens_before


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


async def test_qa_no_items_fails_loudly():
    class EmptyClient(QAClient):
        def content(self) -> str:
            return "no structured items here"

    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    hook = qa_hook(trace, FakeService(), EmptyClient())
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


async def test_qa_captures_tools_and_system_prompt():
    """The hook ships the rollout's system prompt + captured tool schemas with QA updates,
    and advertises the tools on the QA generations themselves."""
    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    client = QAClient()
    hook = qa_hook(trace, service, client)
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    hook.capture_request({"tools": tools, "messages": []})
    commit_turn(
        trace,
        hook,
        [vf.SystemMessage(content="you are an agent"), UserMessage(content="u1")],
        response("a1", [9, 1, 2], [3]),
    )
    turn = graph.prepare_turn(trace, [vf.SystemMessage(content="you are an agent"), UserMessage(content="s")])
    await hook.on_turn_prepared(turn)

    for body, _, _ in client.requests:
        assert body["tools"] == tools  # same rendered system block as regular turns
    (update,) = service.updates
    assert update["system_prompt"] == "you are an agent"
    assert update["tools"] == tools
    assert trace.info["ttt"]["system_prompt"] == "you are an agent"
    assert trace.info["ttt"]["tools"] == tools


def test_parse_and_dedup_items():
    from verifiers.v1.ttt import dedup_items, parse_qa_items

    text = (
        "preamble\n"
        "<item><type>qa</type><question>What is x?</question><answer>42</answer></item>\n"
        "<item><type>lesson</type><question>When Y happens, do what?</question><answer>Z</answer></item>\n"
        "<item><type>weird</type><question>Q3?</question><answer>A3</answer></item>\n"
        "<item><question>no answer</question></item>\n"  # malformed: dropped
        "trailing junk"
    )
    items = parse_qa_items(text)
    assert [i["type"] for i in items] == ["qa", "lesson", "qa"]  # unknown type -> qa

    duplicates = [
        {"type": "qa", "question": "What is the value of x in task T?", "answer": "a"},
        {"type": "qa", "question": "What is the value of x in task T??", "answer": "b"},
        {"type": "qa", "question": "Completely different question about z", "answer": "c"},
    ]
    assert len(dedup_items(duplicates, 0.85)) == 2


async def test_qa_verify_drops_flagged_items():
    """qa.verify: one extra call re-presents the branch + numbered items; flagged items
    are dropped from the training set and recorded as rejected on the trace."""

    class ScriptedClient(QAClient):
        def __init__(self, contents: list[str]):
            super().__init__()
            self.contents = list(contents)

        def content(self) -> str:
            return self.contents.pop(0)

    items = [
        {"type": "qa", "question": "What is x in task T?", "answer": "42"},
        {"type": "qa", "question": "What port does service S use?", "answer": "8080"},
    ]
    generation = item_text(items)
    verdict = "<failed>\n<num>1</num> <reason>unsupported</reason>\n</failed>"

    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    client = ScriptedClient([generation, generation, verdict])
    hook = qa_hook(trace, service, client, verify=True)
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1, 2], [3]))
    turn = graph.prepare_turn(trace, [UserMessage(content="summary")])
    await hook.on_turn_prepared(turn)

    # 2 generations + 1 verification call, all against the branch.
    assert len(client.requests) == 3
    verification_body = client.requests[-1][0]
    instruction = verification_body["messages"][-1]["content"]
    assert "Item 1:" in instruction and "Item 2:" in instruction
    assert "SELF-CONTAINMENT" in instruction

    # Item 1 dropped from the shipped training set; item 2 survives.
    (update,) = service.updates
    assert update["qa_pairs"] == [items[1]]
    # The rejected item is recorded with its reason.
    (rejected,) = trace.info["ttt"]["qa_rejected"]
    assert rejected["question"] == items[0]["question"]
    assert rejected["reason"] == "unsupported"
    # The verification exchange is itself a tagged branch (3 QA branches total).
    assert sum(1 for b in trace.branches if b.is_ttt_qa) == 3


async def test_qa_verify_fails_open_on_malformed_verdict():
    """A verification reply with no parseable <failed> block keeps ALL items — a malformed
    grader must not silently discard the study set."""

    class ScriptedClient(QAClient):
        def __init__(self, contents: list[str]):
            super().__init__()
            self.contents = list(contents)

        def content(self) -> str:
            return self.contents.pop(0)

    items = [{"type": "qa", "question": "What is x in task T?", "answer": "42"}]
    generation = item_text(items)

    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    service = FakeService()
    client = ScriptedClient([generation, generation, "gibberish with no block"])
    hook = qa_hook(trace, service, client, verify=True)
    commit_turn(trace, hook, [UserMessage(content="u1")], response("a1", [1], [2]))
    turn = graph.prepare_turn(trace, [UserMessage(content="s")])
    await hook.on_turn_prepared(turn)
    (update,) = service.updates
    assert update["qa_pairs"] == items
    assert "qa_rejected" not in trace.info["ttt"]


def test_parse_failed_items():
    from verifiers.v1.ttt import parse_failed_items

    text = (
        "Looking at the items...\n<failed>\n"
        "<num>2</num> <reason>refers to 'the error' without naming it</reason>\n"
        "<num>5</num>\n"  # no reason: fine
        "<num>99</num> <reason>out of range: ignored</reason>\n"
        "</failed>"
    )
    failed = parse_failed_items(text, num_items=6)
    assert failed == {2: "refers to 'the error' without naming it", 5: ""}
    # Empty block = everything passed; missing block = fail open (keep all).
    assert parse_failed_items("<failed></failed>", 3) == {}
    assert parse_failed_items("no block at all", 3) == {}
