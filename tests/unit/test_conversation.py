"""Conversations (`verifiers.v1.conversation`): turn-taking as resumed runs on one live
box, graded once at the end, and a conversation cut off mid-turn continuing from its
latest per-call save when run again.

Deterministic, no API key: the fakes of `test_agent_resume`, with a scripted model
that writes one file per user message, then answers it.
"""



import json

import pytest
from aiohttp import web
from test_agent_resume import (  # noqa: F401 - fixtures
    EVENTS,
    Endpoint,
    FakeBackend,
    NotesTask,
    RestoreConfig,
    ScriptHarness,
    box_lost,
    fake_boxes,
    make,
)

import verifiers.v1 as vf
import verifiers.v1.box as box_module
import verifiers.v1.runtimes as runtimes_module
from verifiers.v1.utils.retries import RETRY_NOTE


class GradedTask(NotesTask):
    """Rewards the files the conversation left in the box."""

    @vf.reward(weight=1.0)
    async def wrote_every_request(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        conversation = vf.Conversation.of(trace)
        assert conversation is not None
        asked = [turn.text for turn in conversation.turns if turn.role == "user"]
        listing = await runtime.run(["ls"], {})
        present = set(listing.stdout.split())
        return float(all(f"{text}.txt" in present for text in asked))


def task(prompt: str = "start") -> GradedTask:
    return GradedTask(vf.TaskData(idx=0, prompt=prompt))


def answer(messages: list[dict]) -> dict:
    """Write `<request>.txt` for the latest user request, then answer it. A retry note
    is no request: the request before it is still open."""
    requests = [
        (index, m["content"])
        for index, m in enumerate(messages)
        if m["role"] == "user" and m["content"] != RETRY_NOTE
    ]
    index, request = requests[-1]
    wrote = any(m["role"] == "tool" for m in messages[index:])
    if not wrote:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{len(messages)}",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps(
                            {"path": f"{request}.txt", "text": request}
                        ),
                    },
                }
            ],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": f"done: {request}"}
        finish = "stop"
    return {
        "id": "scripted",
        "object": "chat.completion",
        "created": 0,
        "model": "scripted",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def serve(respond) -> tuple[web.AppRunner, Endpoint]:
    requests: list[dict] = []

    async def handle(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        return web.json_response(respond(body["messages"]))

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, Endpoint(f"http://127.0.0.1:{port}/v1", requests)


@pytest.fixture
async def model():
    runner, endpoint = await serve(answer)
    yield endpoint
    await runner.cleanup()


@pytest.fixture(autouse=True)
def owned_fake_boxes(monkeypatch):
    def fake(config, name=None):
        return FakeBackend(config, name)

    def restore(config, checkpoint):
        return RestoreConfig(**config.model_dump(), checkpoint=checkpoint)

    monkeypatch.setattr(runtimes_module, "make_backend", fake)
    monkeypatch.setattr(box_module, "make_backend", fake)
    monkeypatch.setattr(box_module, "restore_config", restore)


@pytest.fixture
def store(tmp_path) -> vf.TraceDir:
    return vf.TraceDir(tmp_path / "traces")


def users(request: dict) -> list[str]:
    return [m["content"] for m in request["messages"] if m["role"] == "user"]


async def test_each_user_turn_is_a_resumed_run_on_the_live_box(model):
    agent = make(model, save="per_call")
    trace = await vf.converse(agent, task(), vf.ScriptedUser(["first", "second"]))
    assert trace.ok, trace.errors
    conversation = vf.Conversation.of(trace)
    assert conversation is not None
    assert [(turn.role, turn.text) for turn in conversation.turns] == [
        ("user", "start"),
        ("assistant", "done: start"),
        ("user", "first"),
        ("assistant", "done: first"),
        ("user", "second"),
        ("assistant", "done: second"),
    ]
    assert conversation.stop == "user_done"
    assert conversation.replies == 2
    # The task was set up once: the later turns continued on the box as it was.
    assert EVENTS == ["setup"]
    # The reply reached the harness as a normal user message, after the whole
    # conversation so far.
    last = model.requests[-1]
    assert users(last) == ["start", "first", "second"]
    # Graded once, on the end state, with the conversation on the trace.
    assert trace.rewards["wrote_every_request"].value == 1.0
    assert trace.resumed_from is not None


async def test_the_turn_cap_ends_the_conversation(model):
    agent = make(model)
    trace = await vf.converse(
        agent, task(), vf.ScriptedUser(["first", "second", "third"]), max_turns=1
    )
    conversation = vf.Conversation.of(trace)
    assert conversation.stop == "turn_cap"
    assert conversation.replies == 1
    assert conversation.last_agent_message == "done: first"
    assert trace.rewards["wrote_every_request"].value == 1.0


async def test_a_conversation_cut_mid_turn_continues_from_its_latest_save(model, store):
    agent = make(model, save="per_call")
    user = vf.ScriptedUser(["first", "second"])
    FakeBackend.doomed["second.txt"] = box_lost
    with vf.use_durable(vf.Durable(store, match={"job": "j1"})):
        failed = await vf.converse(agent, task(), user)
        assert not failed.ok
        assert vf.Conversation.of(failed).stop == "solver_failed"
        # The cut-off turn saved after its model call asked for the write.
        assert failed.checkpoints
        model.requests.clear()
        EVENTS.clear()
        trace = await vf.converse(agent, task(), user)
    assert trace.ok, trace.errors
    # The finished turns came back from their records; only the cut-off turn ran, and
    # it continued after its saved call, on the box restored from that save.
    assert trace.resumed_from == failed.id
    assert EVENTS == ["restore"]
    assert model.requests and all(
        users(request)[:3] == ["start", "first", "second"] for request in model.requests
    )
    assert users(model.requests[0])[-1] == RETRY_NOTE
    assert trace.rewards["wrote_every_request"].value == 1.0
    assert vf.Conversation.of(trace).stop == "user_done"


def user_answer(messages: list[dict]) -> dict:
    """The modeled user: asks for one more thing, then is done."""
    asked = sum(m["role"] == "assistant" for m in messages)
    content = "also" if asked < 2 else f"thanks {vf.conversation.DONE}"
    return {
        "id": "user",
        "object": "chat.completion",
        "created": 0,
        "model": "user",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def test_a_modeled_user_answers_with_the_roles_swapped(model):
    runner, endpoint = await serve(user_answer)
    try:
        user_agent = make(endpoint, harness=ScriptHarness)
        user = vf.ModelUser(user_agent, "You want a file per request.")
        trace = await vf.converse(make(model), task(), user)
    finally:
        await runner.cleanup()
    conversation = vf.Conversation.of(trace)
    assert [turn.text for turn in conversation.turns if turn.role == "user"] == [
        "start",
        "also",
        "thanks",
    ]
    assert conversation.stop == "user_done"
    # The user saw its own messages as the assistant's and the agent's as the user's,
    # with its scenario in the system prompt.
    first = endpoint.requests[0]["messages"]
    assert first[0]["role"] == "system" and "a file per request" in first[0]["content"]
    assert [m["role"] for m in first[1:]] == ["user", "assistant", "user"]
    assert first[-1]["content"] == "done: start"
