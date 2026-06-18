"""Each rollout error type is raised at its boundary and recorded on the trace.

These drive a real `Rollout` — a real subprocess runtime, a real interception server (bound to a
real port), and a real harness subprocess — but with a controllable model client and fixture
taskset/harness, so each boundary fails deterministically and the tests run in CI with no model
endpoint. Each asserts the captured `trace.error` has the right `type` and a populated `traceback`
(and that an overlong prompt is a clean truncation, not an error).
"""

import json

import verifiers.v1 as vf
from verifiers.v1 import retries
from verifiers.v1 import rollout as rollout_mod
from verifiers.v1 import runtimes
from verifiers.v1.clients import Client, RolloutContext
from verifiers.v1.errors import OverlongPromptError, ProviderError
from verifiers.v1.interception import RolloutLimits
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import SubprocessConfig
from verifiers.v1.runtimes.subprocess import SubprocessRuntime
from verifiers.v1.types import SamplingConfig


# --- controllable model clients ---


class _UnusedClient(Client):
    """A model client that must never be called — for boundaries that fail before any model turn."""

    async def get_response(self, *args, **kwargs):
        raise AssertionError("the model client should not be called for this rollout")


class _FailClient(Client):
    """Every completion raises the given error at the provider boundary."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def get_response(self, *args, **kwargs):
        raise self._error


# --- fixture harnesses (run in the subprocess runtime) ---


def _one_call_script(endpoint: str, secret: str, model: str) -> str:
    """A stdlib-only program that makes one chat-completions call to the interception endpoint; it
    exits non-zero if the call returns an error status (urlopen raises on a non-2xx response)."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]})
    return (
        "import urllib.request\n"
        f"req = urllib.request.Request(\n"
        f"    {endpoint!r} + '/chat/completions',\n"
        f"    data={body!r}.encode(),\n"
        f"    headers={{'Authorization': 'Bearer ' + {secret!r}, 'Content-Type': 'application/json'}},\n"
        ")\n"
        "urllib.request.urlopen(req).read()\n"
    )


class _OneCallHarness(vf.Harness):
    """Makes exactly one model call (driving the provider / overlong paths through the interception
    server), then exits with whatever that call implied."""

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls):
        return await runtime.run(
            ["python3", "-c", _one_call_script(endpoint, secret, ctx.model)], {}
        )


class _ExitNonZeroHarness(vf.Harness):
    """An agent process that crashes (non-zero exit) without making a model call."""

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls):
        return await runtime.run(["sh", "-c", "echo 'agent crashed' >&2; exit 7"], {})


class _RaisingHarness(vf.Harness):
    """A harness whose own `launch` raises (not a clean non-zero exit)."""

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls):
        raise ValueError("harness internals exploded")


class _NoopHarness(vf.Harness):
    """Completes cleanly without a model call, so the rollout reaches scoring."""

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls):
        return await runtime.run(["sh", "-c", "exit 0"], {})


# --- fixture tasksets ---


class _PlainTaskset(vf.Taskset):
    """One trivial task with a prompt — the harness drives it."""

    def load_tasks(self):
        return [vf.Task(idx=0, prompt="hello")]


class _ToolFailTaskset(_PlainTaskset):
    def tools(self, task):
        raise RuntimeError("could not build tool servers")


class _NoPromptTaskset(vf.Taskset):
    """A task with no prompt and no user simulator — the rollout can't open the conversation."""

    def load_tasks(self):
        return [vf.Task(idx=0, prompt=None)]


class _ScoreFailTaskset(_PlainTaskset):
    """A taskset whose `@reward` raises a plain Python error (e.g. a verifier program failing)."""

    @vf.reward(weight=1.0)
    async def boom(self, trace):
        raise RuntimeError("verifier program failed")


# --- driver ---


async def _run(taskset, harness, *, client, runtime_config=None, **kwargs):
    ctx = RolloutContext(
        model="test-model",
        client=client,
        sampling=SamplingConfig(max_tokens=64, temperature=0),
    )
    rollout = Rollout(
        task=taskset.load_tasks()[0],
        taskset=taskset,
        harness=harness,
        ctx=ctx,
        runtime_config=runtime_config or SubprocessConfig(),
        limits=RolloutLimits(max_turns=2),
        **kwargs,
    )
    return await rollout.run()


def _harness(cls):
    return cls(vf.HarnessConfig(id="test-harness"))


def _taskset(cls):
    return cls(vf.TasksetConfig(id="test-taskset"))


def _assert_recorded(trace, error_type: str):
    assert trace.error is not None, "expected an error recorded on the trace"
    assert trace.error.type == error_type, (
        f"got {trace.error.type}: {trace.error.message}"
    )
    assert trace.error.message
    assert "Traceback" in trace.error.traceback
    assert trace.errors == [trace.error]  # exactly one, no secondary error


# --- tests ---


async def test_provider_error_recorded():
    """A failed model call surfaces as ProviderError — re-raised past the harness's non-zero exit,
    not flattened into the secondary HarnessError."""
    trace = await _run(
        _taskset(_PlainTaskset),
        _harness(_OneCallHarness),
        client=_FailClient(
            ProviderError("provider returned HTTP 503: upstream unavailable")
        ),
    )
    _assert_recorded(trace, "ProviderError")
    assert "503" in trace.error.message


async def test_harness_error_on_nonzero_exit():
    trace = await _run(
        _taskset(_PlainTaskset), _harness(_ExitNonZeroHarness), client=_UnusedClient()
    )
    _assert_recorded(trace, "HarnessError")
    assert "exited 7" in trace.error.message


async def test_harness_error_on_launch_exception():
    trace = await _run(
        _taskset(_PlainTaskset), _harness(_RaisingHarness), client=_UnusedClient()
    )
    _assert_recorded(trace, "HarnessError")


async def test_tool_error_recorded():
    trace = await _run(
        _taskset(_ToolFailTaskset), _harness(_OneCallHarness), client=_UnusedClient()
    )
    _assert_recorded(trace, "ToolError")


async def test_sandbox_error_recorded():
    """No prompt and no user simulator — the rollout has no way to open the conversation."""
    trace = await _run(
        _taskset(_NoPromptTaskset), _harness(_OneCallHarness), client=_UnusedClient()
    )
    _assert_recorded(trace, "SandboxError")


async def test_taskset_error_recorded():
    """A plain Python error from a taskset `@reward` is wrapped by the framework as TasksetError —
    taskset code raises plain errors, never a vf.* type."""
    trace = await _run(
        _taskset(_ScoreFailTaskset), _harness(_NoopHarness), client=_UnusedClient()
    )
    _assert_recorded(trace, "TasksetError")


async def test_tunnel_error_recorded(monkeypatch):
    """A remote runtime whose interception-server tunnel can't be established surfaces as
    TunnelError (the tunnel is retried first, then the failure is recorded)."""
    import prime_tunnel
    from tenacity import wait_none

    class _FailingTunnel:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            raise ConnectionError("tunnel service unreachable")

        def sync_stop(self):
            pass

    monkeypatch.setattr(prime_tunnel, "Tunnel", _FailingTunnel)
    # don't actually sleep between tunnel retries
    monkeypatch.setattr(
        retries, "wait_exponential_jitter", lambda **kwargs: wait_none()
    )

    class _RemoteSubprocess(SubprocessRuntime):
        is_local = (
            False  # a remote runtime → the harness reaches the host over a tunnel
        )

    def _make_remote(config, name=None):
        runtime = _RemoteSubprocess(config, name)
        runtimes.base.register(runtime)
        return runtime

    monkeypatch.setattr(rollout_mod, "make_runtime", _make_remote)

    trace = await _run(
        _taskset(_PlainTaskset), _harness(_OneCallHarness), client=_UnusedClient()
    )
    _assert_recorded(trace, "TunnelError")
    assert issubclass(vf.TunnelError, vf.InterceptionError)  # the interception boundary


async def test_overlong_prompt_is_truncation_not_error():
    """An overlong prompt ends the rollout as a clean truncation (`context_length`), not an error."""
    trace = await _run(
        _taskset(_PlainTaskset),
        _harness(_OneCallHarness),
        client=_FailClient(
            OverlongPromptError("prompt too long: context length exceeded")
        ),
    )
    assert trace.errors == []
    assert trace.error is None
    assert trace.stop_condition == "context_length"
