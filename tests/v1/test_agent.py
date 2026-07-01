"""Tests for agent runs (`verifiers.v1.agent`), none needing a real model: spec
validation, model-table resolution, budget mapping, the verdict record's wire
round-trip, `run_judges`' precondition checks — plus one full-path integration test
(`test_judged_rollout_against_stub`) that runs a real rollout + judge agent run in the
subprocess runtime against a scripted local endpoint, so the judge's canned bash
tool-call actually executes and writes the verdict file the framework reads back."""

import dataclasses
import json
import re
import shlex

import pytest
from aiohttp import web
from pydantic import BaseModel

import verifiers.v1 as vf
from verifiers.v1.agent import resolve_model, run_judges
from verifiers.v1.clients import EvalClientConfig


class Verdict(BaseModel):
    passed: bool


def _ctx(model: str = "the-policy") -> vf.RolloutContext:
    return vf.RolloutContext(model=model, client=None, sampling=vf.SamplingConfig())


def test_spec_defaults():
    spec = vf.JudgeSpec(name="j", prompt="grade it", verdict=Verdict)
    assert spec.model == "policy"
    assert spec.placement == "rollout"
    assert spec.harness.id == "default"
    assert spec.trainable is None


def test_budget_maps_to_limits():
    budget = vf.AgentBudget(max_turns=5, max_total_tokens=1000)
    limits = budget.limits()
    assert limits.max_turns == 5
    assert limits.max_total_tokens == 1000
    assert limits.max_input_tokens is None


def test_resolve_model_policy_is_ctx():
    ctx = _ctx()
    spec = vf.AgentSpec()
    assert resolve_model(spec, ctx, {}) is ctx


def test_resolve_model_named_entry():
    ctx = _ctx()
    grader = _ctx("the-grader")
    spec = vf.AgentSpec(model="grader")
    assert resolve_model(spec, ctx, {"grader": grader}) is grader


def test_resolve_model_unknown_name_lists_table():
    with pytest.raises(vf.JudgeError, match="grader.*entries: judge-small"):
        resolve_model(vf.AgentSpec(model="grader"), _ctx(), {"judge-small": _ctx()})


def test_resolve_model_sampling_override():
    ctx = _ctx()
    spec = vf.AgentSpec(sampling=vf.SamplingConfig(temperature=0))
    resolved = resolve_model(spec, ctx, {})
    assert resolved.model == ctx.model
    assert resolved.sampling.temperature == 0
    assert dataclasses.replace(resolved, sampling=ctx.sampling) == ctx


async def test_run_judges_rejects_duplicate_names():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="p"))
    specs = [
        vf.JudgeSpec(name="j", prompt="a", verdict=Verdict),
        vf.JudgeSpec(name="j", prompt="b", verdict=Verdict),
    ]
    with pytest.raises(vf.JudgeError, match="duplicate judge names"):
        await run_judges(specs, trace, None, ctx=_ctx())


async def test_run_judges_no_specs_is_noop():
    trace = vf.Trace(task=vf.Task(idx=0, prompt="p"))
    assert await run_judges([], trace, None, ctx=_ctx()) == {}
    assert trace.agents == []


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
    app = web.Application()
    app.router.add_post("/v1/chat/completions", _stub_model)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        config = vf.EnvConfig(
            taskset={"id": "echo-judged-v1"},
            harness={"id": "null", "runtime": {"type": "subprocess"}},
            max_turns=2,
            timeout={"rollout": 300, "scoring": 300},
        )
        env = vf.Environment(config)
        client = vf.resolve_client(
            EvalClientConfig(
                base_url=f"http://127.0.0.1:{port}/v1", api_key_var="STUB_API_KEY"
            )
        )
        ctx = vf.RolloutContext(
            model="stub-model", client=client, sampling=vf.SamplingConfig(temperature=0)
        )
        (task,) = env.taskset.load_tasks()
        (trace,) = await env.episode(task, ctx, n=1).run()
    finally:
        await runner.cleanup()
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
