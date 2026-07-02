"""Tests for agent runs (`verifiers.v1.agent`), none needing a real model. Unit tests
cover the pure logic: model-table resolution, verdict-channel selection, fence
stripping, the duplicate-name guard, and the provenance record's wire round-trip. The
real coverage is two full-path integration tests that run rollouts + judge agent runs
in the subprocess runtime against a scripted local endpoint: a file-verdict judge whose
canned bash tool-call actually executes and writes the verdict file, and a
reply-verdict (`null`-harness) judge whose fenced JSON reply is parsed as the
verdict."""

import contextlib
import dataclasses
import json
import re
import shlex

import pytest
from aiohttp import web
from pydantic import BaseModel

import verifiers.v1 as vf
from verifiers.v1.agent import _strip_fences, resolve_model, run_judges
from verifiers.v1.clients import EvalClientConfig


class Verdict(BaseModel):
    passed: bool


def _ctx(model: str = "the-policy") -> vf.RolloutContext:
    return vf.RolloutContext(model=model, client=None, sampling=vf.SamplingConfig())


def test_resolve_model():
    ctx = _ctx()
    grader = _ctx("the-grader")
    # "policy" is the rollout's own context; other names hit the table.
    assert resolve_model(vf.AgentSpec(), ctx, {}) is ctx
    assert (
        resolve_model(vf.AgentSpec(model="grader"), ctx, {"grader": grader}) is grader
    )
    # An unknown name fails actionably, listing what IS configured.
    with pytest.raises(vf.JudgeError, match="grader.*entries: judge-small"):
        resolve_model(vf.AgentSpec(model="grader"), ctx, {"judge-small": _ctx()})
    # A spec's sampling overrides the resolved context's — and only the sampling.
    spec = vf.AgentSpec(sampling=vf.SamplingConfig(temperature=0))
    resolved = resolve_model(spec, ctx, {})
    assert resolved.sampling.temperature == 0
    assert dataclasses.replace(resolved, sampling=ctx.sampling) == ctx


def test_verdict_source_derived_from_harness():
    tools_judge = vf.JudgeSpec(name="j", prompt="p", verdict=Verdict)
    assert tools_judge.resolved_verdict_source == "file"
    null_judge = vf.JudgeSpec(
        name="j", prompt="p", verdict=Verdict, harness={"id": "null"}
    )
    assert null_judge.resolved_verdict_source == "reply"
    forced = vf.JudgeSpec(
        name="j",
        prompt="p",
        verdict=Verdict,
        harness={"id": "null"},
        verdict_source="file",
    )
    assert forced.resolved_verdict_source == "file"


def test_strip_fences():
    assert _strip_fences('{"a": 1}') == '{"a": 1}'
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'


async def test_run_judges_rejects_duplicate_names():
    # Duplicate names would silently overwrite each other's verdicts in the dict.
    trace = vf.Trace(task=vf.Task(idx=0, prompt="p"))
    specs = [
        vf.JudgeSpec(name="j", prompt="a", verdict=Verdict),
        vf.JudgeSpec(name="j", prompt="b", verdict=Verdict),
    ]
    with pytest.raises(vf.JudgeError, match="duplicate judge names"):
        await run_judges(specs, trace, None, ctx=_ctx())


PHRASE = "hello world"


def _completion(content: str | None = None, tool_calls: list | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": "stub-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _bash_call(command: str) -> list:
    return [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }
    ]


@contextlib.asynccontextmanager
async def _stub_ctx(handler):
    """Serve `handler` as a local OpenAI-compatible endpoint and yield a
    `RolloutContext` whose eval client points at it."""
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = vf.resolve_client(
        EvalClientConfig(
            base_url=f"http://127.0.0.1:{port}/v1", api_key_var="STUB_API_KEY"
        )
    )
    try:
        yield vf.RolloutContext(
            model="stub-model", client=client, sampling=vf.SamplingConfig(temperature=0)
        )
    finally:
        await runner.cleanup()


async def _stub_model(request: web.Request) -> web.Response:
    """A scripted OpenAI-compatible endpoint. The policy conversation gets the echoed
    phrase; the judge conversation (recognized by the materialized `/tmp/vf-agent/`
    paths in its prompt) gets one bash tool-call that writes the verdict file — really
    executed by the judge harness in the runtime — then a closing message."""
    body = await request.json()
    text = json.dumps(body["messages"])
    if "/tmp/vf-agent/" not in text:  # the policy conversation
        return web.json_response(_completion(content=f"Sure: {PHRASE}"))
    turns = sum(1 for m in body["messages"] if m["role"] == "assistant")
    if turns == 0:
        run_dir = re.search(r"(/tmp/vf-agent/[0-9a-f]+)/verdict\.json", text).group(1)
        verdict = json.dumps({"echoed": True, "evidence": PHRASE})
        command = f"mkdir -p {run_dir} && printf %s {shlex.quote(verdict)} > {run_dir}/verdict.json"
        return web.json_response(_completion(tool_calls=_bash_call(command)))
    return web.json_response(_completion(content="Verdict written."))


async def test_judged_rollout_against_stub(tmp_path):
    """The full judge path, no real model: a rollout of `echo-judged-v1` in the
    subprocess runtime against the scripted endpoint. Exercises input materialization,
    judge-harness provisioning into the live runtime, the judge's own interception
    session, a really-executed tool call writing the verdict, read-back + schema
    validation, verdict→reward mapping, and the provenance record on the trace."""
    config = vf.EnvConfig(
        taskset={"id": "echo-judged-v1"},
        harness={"id": "null", "runtime": {"type": "subprocess"}},
        max_turns=2,
        timeout={"rollout": 300, "scoring": 300},
    )
    env = vf.Environment(config)
    async with _stub_ctx(_stub_model) as ctx:
        (task,) = env.taskset.load_tasks()
        (trace,) = await env.episode(task, ctx, n=1).run()
    assert trace.errors == []
    assert trace.reward == 1.0
    (run,) = trace.agents
    assert run.name == "echoed"
    assert run.role == "judge"
    assert run.model == "policy"
    assert run.trainable is False
    assert run.verdict == {"echoed": True, "evidence": PHRASE}
    assert run.trace.num_turns == 2  # the bash tool-call turn + the closing message
    assert trace.extra_usage  # judge spend recorded off the policy's own usage


async def _stub_reply_judged(request: web.Request) -> web.Response:
    """Scripted endpoint for the reply-verdict path: the judge conversation (recognized
    by the reply contract's grading text) gets a *fenced* JSON verdict — exercising the
    fence tolerance — and everything else is the policy conversation."""
    body = await request.json()
    text = json.dumps(body["messages"])
    if "You are grading a completed agent rollout" in text:
        verdict = json.dumps({"echoed": True})
        return web.json_response(_completion(content=f"```json\n{verdict}\n```"))
    return web.json_response(_completion(content=f"Sure: {PHRASE}"))


async def test_reply_judged_rollout_against_stub(tmp_path):
    """The single-call judge path, no real model: a `null`-harness reply-verdict judge
    over `echo-reply-judged-v1`. Exercises the injectable `judges(task, trace)` hook
    (the evidence rides in the prompt), the reply contract, fence-stripping + schema
    validation of the final reply, and the provenance record — with no files
    materialized and no tool calls."""
    config = vf.EnvConfig(
        taskset={"id": "echo-reply-judged-v1"},
        harness={"id": "null", "runtime": {"type": "subprocess"}},
        max_turns=2,
        timeout={"rollout": 300, "scoring": 300},
    )
    env = vf.Environment(config)
    async with _stub_ctx(_stub_reply_judged) as ctx:
        (task,) = env.taskset.load_tasks()
        (trace,) = await env.episode(task, ctx, n=1).run()
    assert trace.errors == []
    assert trace.reward == 1.0
    (run,) = trace.agents
    assert run.name == "echoed"
    assert run.role == "judge"
    assert run.verdict == {"echoed": True}
    assert run.trace.num_turns == 1  # one completion — the reply is the verdict
    assert trace.extra_usage  # judge spend recorded off the policy's own usage


def test_agent_run_wire_roundtrip():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="p"))
    judge_trace = vf.Trace(task=vf.WireTask(idx=0, prompt="judge instructions"))
    trace.agents.append(
        vf.AgentRun(
            name="j",
            model="grader",
            trace=judge_trace,
            verdict={"passed": True},
        )
    )
    record = trace.to_record()
    loaded = vf.WireTrace.model_validate(record)
    (run,) = loaded.agents
    assert run.name == "j"
    assert run.role == "judge"
    assert run.trainable is False
    assert run.verdict == {"passed": True}
    assert run.trace.task.prompt == "judge instructions"
