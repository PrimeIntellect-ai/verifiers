"""The bash harness and its chat program: model-call retries, command timeouts, skills.
No model or sandbox: the launch is captured and the program's pieces run locally."""

import os
import signal
import sys
import time
from types import SimpleNamespace

import pytest

from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.bash import harness as bash
from verifiers.v1.harnesses.utils import core
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime
from verifiers.v1.task import TaskData
from verifiers.v1.types import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class Launched:
    """What `BashHarness.launch` handed `launch_chat_program`."""

    args: list[str]
    system_prompt: str | None


async def launch(monkeypatch, runtime=None, **config) -> Launched:
    launched = Launched()

    async def fake_launch(
        source, config, ctx, trace, runtime, endpoint, secret, *rest, **kwargs
    ):
        launched.system_prompt = rest[1]
        launched.args = list(kwargs["extra_args"])

    monkeypatch.setattr(bash, "launch_chat_program", fake_launch)
    harness = bash.BashHarness(bash.BashHarnessConfig(**config))
    await harness.launch(
        None, None, runtime, "http://e", "s", {}, TaskData(prompt="go")
    )
    return launched


class StopMain(Exception):
    pass


async def client_kwargs(monkeypatch, *flags: str) -> dict:
    """The keyword arguments the program builds its OpenAI client with."""
    seen: dict = {}

    def fake_client(**kwargs):
        seen.update(kwargs)
        raise StopMain

    monkeypatch.setattr(core, "AsyncOpenAI", fake_client)
    argv = ["prog", "--base-url=http://e", "--api-key=k", "--model=m", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(StopMain):
        await core.main()
    return seen


# --- model-call retries -------------------------------------------------------------


async def test_bash_launch_passes_its_model_retries(monkeypatch):
    assert "--max-retries=10" in (await launch(monkeypatch)).args
    assert "--max-retries=0" in (await launch(monkeypatch, model_retries=0)).args


async def test_program_builds_its_client_with_the_retry_flag(monkeypatch):
    assert (await client_kwargs(monkeypatch, "--bash", "--max-retries=7"))[
        "max_retries"
    ] == 7


async def test_program_keeps_the_sdk_default_without_the_flag(monkeypatch):
    # NullHarness passes no flag; its program keeps the OpenAI SDK's two retries.
    assert (await client_kwargs(monkeypatch))["max_retries"] == 2


# --- bash command timeouts and output -----------------------------------------------


async def test_bash_launch_passes_its_command_timeout(monkeypatch):
    assert "--command-timeout=3600" in (await launch(monkeypatch)).args
    launched = await launch(monkeypatch, command_timeout=300)
    assert "--command-timeout=300" in launched.args


def test_a_timed_out_command_kills_its_process_group_and_keeps_its_output(tmp_path):
    pidfile = tmp_path / "child.pid"
    started = time.monotonic()
    output = core.run_bash(f"echo before; sleep 60 & echo $! > {pidfile}; sleep 60", 1)
    assert time.monotonic() - started < 10
    assert output == "before\n\nerror: command exceeded 1s and was killed"
    child = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while _running(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _running(child), "the backgrounded child outlived the timeout"


def test_a_finished_command_leaves_its_background_children_running(tmp_path):
    pidfile = tmp_path / "server.pid"
    started = time.monotonic()
    assert core.run_bash(f"sleep 30 & echo $! > {pidfile}; echo up", 10) == "up\n"
    assert time.monotonic() - started < 5
    child = int(pidfile.read_text())
    try:
        assert _running(child)
    finally:
        os.kill(child, signal.SIGKILL)


def test_undecodable_output_is_replaced_not_raised():
    assert core.run_bash("printf 'a\\x80b'; printf '\\xff' >&2") == "a�b�"


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child of an exited shell may linger as a zombie until init reaps it.
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


# --- resuming a conversation --------------------------------------------------------


class ResumableHarness(Harness[HarnessConfig]):
    SUPPORTS_RESUME = True

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **_):
        self.prompt = data.prompt


async def resumed(branch: list, data: TaskData | None = None) -> list:
    harness = ResumableHarness(HarnessConfig(id="resumable"))
    trace = SimpleNamespace(branches=[SimpleNamespace(messages=branch)])
    new = [UserMessage(content="continue")]
    await harness.resume(None, trace, None, "", "", {}, data or TaskData(), new)
    return harness.prompt


def call(call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="bash", arguments='{"command": "sleep 99"}')


async def test_resume_drops_tool_calls_a_cut_off_segment_left_unanswered():
    opened = [UserMessage(content="fix it"), AssistantMessage(content="looking")]
    answered = [
        AssistantMessage(tool_calls=[call("a")]),
        ToolMessage(tool_call_id="a", content="done"),
    ]
    cut_off = AssistantMessage(content="next", tool_calls=[call("b"), call("c")])
    prompt = await resumed([*opened, *answered, cut_off])
    assert prompt == [*opened, *answered, UserMessage(content="continue")]


async def test_resume_keeps_a_conversation_that_ends_answered():
    branch = [
        SystemMessage(content="rules"),
        UserMessage(content="fix it"),
        AssistantMessage(content="fixed"),
    ]
    prompt = await resumed(branch)
    assert prompt == [*branch, UserMessage(content="continue")]
    # A system prompt the task re-emits is not carried twice.
    prompt = await resumed(branch, TaskData(system_prompt="rules"))
    assert prompt == [*branch[1:], UserMessage(content="continue")]


# --- skills -------------------------------------------------------------------------


def write_skill(root, folder: str, card: str) -> None:
    (root / folder).mkdir(parents=True)
    (root / folder / "SKILL.md").write_text(card)
    (root / folder / "tool.sh").write_text("echo hi\n")


async def test_bash_installs_its_skills_and_lists_them_in_the_system_prompt(
    tmp_path, monkeypatch
):
    host = tmp_path / "host"
    write_skill(
        host,
        "git-forensics",
        '---\nname: "git-forensics"\ndescription: Recover lost commits.\n---\n# Body\n',
    )
    write_skill(host, "untitled", "# No front matter\n")
    runtime = SubprocessRuntime(SubprocessConfig())
    await runtime.start()
    try:
        harness = bash.BashHarness(
            bash.BashHarnessConfig(skills=[host / "git-forensics", host / "untitled"])
        )
        harness.skills_dir = str(tmp_path / "installed")
        monkeypatch.setattr(runtime, "prepare_uv_script", _no_program)
        await harness.setup(runtime)
        listing = await harness.skills_listing(runtime)
    finally:
        await runtime.teardown()
    root = tmp_path / "installed"
    assert (root / "git-forensics" / "tool.sh").read_text() == "echo hi\n"
    assert listing == "\n".join(
        [
            bash.SKILLS_PROMPT.format(root=root),
            f"- `git-forensics` ({root}/git-forensics/SKILL.md): Recover lost commits.",
            f"- `untitled` ({root}/untitled/SKILL.md)",
        ]
    )


async def test_bash_launch_appends_the_skill_listing(monkeypatch):
    async def listing(self, runtime):
        return "SKILLS"

    monkeypatch.setattr(bash.BashHarness, "skills_listing", listing)
    launched = await launch(monkeypatch)
    assert launched.system_prompt.endswith("\n\nSKILLS")


async def test_bash_without_skills_lists_none():
    harness = bash.BashHarness(bash.BashHarnessConfig())
    assert await harness.skills_listing(None) is None


async def _no_program(*args, **kwargs):
    return []
