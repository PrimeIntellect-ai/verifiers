"""The segment seam: `Harness.resume` continues an exchange (no live calls).

The default resume relaunches the program on the accreted conversation as a
Messages prompt; codex overrides it with a native `codex exec resume` so its own
session state — not a replay of our view — carries the context.
"""

import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.errors import HarnessError
from verifiers.v1.graph import MessageNode
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.trace import Trace, TraceTask
from verifiers.v1.types import AssistantMessage, SystemMessage, UserMessage


def _ctx() -> ModelContext:
    return ModelContext(model="test-model", client=None)  # type: ignore[arg-type]


def _trace_with_branch() -> Trace:
    """A trace mid-exchange: system prompt, the opening user turn, one answer."""
    trace = Trace(
        task=TraceTask(
            type="Task",
            data=vf.TaskData(idx=0, prompt=None, system_prompt="be brief"),
        )
    )
    trace.nodes = [
        MessageNode(parent=None, message=SystemMessage(content="be brief")),
        MessageNode(parent=0, message=UserMessage(content="hello")),
        MessageNode(parent=1, message=AssistantMessage(content="hi"), sampled=True),
    ]
    return trace


class _ReplayHarness(Harness[HarnessConfig]):
    """Captures what the default resume relaunches with."""

    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MESSAGE_PROMPT = True

    def __init__(self, config: HarnessConfig) -> None:
        super().__init__(config)
        self.launched_with: list[vf.TaskData] = []

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data
    ) -> ProgramResult:
        self.launched_with.append(data)
        return ProgramResult(exit_code=0, stdout="", stderr="")


async def test_default_resume_relaunches_on_the_conversation():
    """The default resume = launch on branch-so-far + the user's new turn(s), with
    the branch's system message filtered (resolve_prompt re-emits it from
    `data.system_prompt` — it must not double)."""
    harness = _ReplayHarness(HarnessConfig())
    trace = _trace_with_branch()
    await harness.resume(
        _ctx(),
        trace,
        None,
        "http://endpoint",
        "secret",
        {},
        trace.task.data,
        [UserMessage(content="and then?")],
    )
    (data,) = harness.launched_with
    assert [m.role for m in data.prompt] == ["user", "assistant", "user"]
    assert data.prompt[-1].content == "and then?"
    assert data.system_prompt == "be brief"  # untouched; the launch re-emits it


async def test_default_resume_opens_an_empty_exchange():
    """The first segment of a user-opened exchange is a resume onto an empty
    branch: the conversation IS the opening."""
    harness = _ReplayHarness(HarnessConfig())
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)))
    await harness.resume(
        _ctx(),
        trace,
        None,
        "http://endpoint",
        "secret",
        {},
        trace.task.data,
        [UserMessage(content="I want to change my flight")],
    )
    (data,) = harness.launched_with
    assert [m.role for m in data.prompt] == ["user"]


async def test_default_resume_needs_a_messages_prompt():
    """No Messages prompt and no override → the exchange cannot advance; the error
    is a HarnessError (it fires inside a segment, where it is rollout data)."""

    class StringOnly(_ReplayHarness):
        SUPPORTS_MESSAGE_PROMPT = False

    harness = StringOnly(HarnessConfig())
    with pytest.raises(HarnessError, match="cannot continue an exchange"):
        await harness.resume(
            _ctx(),
            _trace_with_branch(),
            None,
            "http://e",
            "s",
            {},
            vf.TaskData(idx=0),
            [UserMessage(content="next")],
        )


class _CapturingRuntime:
    def __init__(self) -> None:
        self.programs: list[tuple[list[str], dict[str, str]]] = []

    async def run(self, argv, env) -> ProgramResult:
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def run_program(self, argv, env) -> ProgramResult:
        self.programs.append((argv, env))
        return ProgramResult(exit_code=0, stdout="", stderr="")


async def test_codex_resume_continues_the_recorded_session():
    """Codex's native resume: `exec resume --last` in the rollout's own CODEX_HOME —
    no conversation replay, and `--last` can't grab a neighboring rollout's session
    because the home is per-trace."""
    from verifiers.v1.harnesses.codex.harness import CodexHarness, CodexHarnessConfig

    harness = CodexHarness(CodexHarnessConfig())
    runtime = _CapturingRuntime()
    trace = _trace_with_branch()
    await harness.resume(
        _ctx(),
        trace,
        runtime,
        "http://endpoint",
        "secret",
        {},
        trace.task.data,
        [UserMessage(content="and then?")],
    )
    ((argv, env),) = runtime.programs
    assert argv[1:4] == ["exec", "resume", "--last"]
    assert argv[-1] == "and then?" and argv[-2] == "--"
    assert env["CODEX_HOME"] == f"/tmp/vf-codex-home-{trace.id}"
    # The provider wiring must ride every segment (codex re-reads config per run).
    assert "model_provider=intercept" in argv


async def test_codex_resume_refuses_non_text_turns():
    from verifiers.v1.harnesses.codex.harness import CodexHarness, CodexHarnessConfig

    harness = CodexHarness(CodexHarnessConfig())
    with pytest.raises(ValueError, match="user turns only"):
        await harness.resume(
            _ctx(),
            _trace_with_branch(),
            _CapturingRuntime(),
            "http://e",
            "s",
            {},
            vf.TaskData(idx=0),
            [AssistantMessage(content="i am not a user")],
        )
