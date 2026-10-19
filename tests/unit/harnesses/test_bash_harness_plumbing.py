"""The bash harness and its chat program: model-call retries, command timeouts, skills.
No model or sandbox: the launch is captured and the program's pieces run locally."""

import errno
import json
import os
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.chat import ChatCompletion

from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.bash import harness as bash
from verifiers.v1.harnesses.utils import compaction, core, mcp
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.subprocess import SubprocessBackend, SubprocessConfig
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
    prompt: Any
    credentials: dict[str, str]


async def launch(monkeypatch, runtime=None, prompt: Any = "go", **config) -> Launched:
    launched = Launched()

    async def fake_launch(
        source, config, ctx, trace, runtime, endpoint, secret, *rest, **kwargs
    ):
        launched.system_prompt, launched.prompt = rest[1], rest[2]
        launched.args = list(kwargs["extra_args"])
        launched.credentials = dict(kwargs.get("credentials") or {})

    monkeypatch.setattr(bash, "launch_chat_program", fake_launch)
    harness = bash.BashHarness(bash.BashHarnessConfig(**config))
    await harness.launch(
        None,
        None,
        runtime,
        "http://e",
        "s",
        {},
        TaskData(prompt=prompt, system_prompt="rules"),
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
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as launch_file:
        launch_file.write(json.dumps({"base_url": "http://e", "api_key": "k"}).encode())
    argv = ["prog", f"--launch-file={launch_file.name}", "--model=m", *flags]
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


async def test_program_reads_its_key_from_the_launch_file_and_deletes_it(
    monkeypatch, tmp_path
):
    launch_file = tmp_path / "launch.json"
    launch_file.write_text(
        json.dumps(
            {
                "base_url": "http://e",
                "api_key": "k",
                "serper_key": "s",
                "messages": [{"role": "user", "content": "go"}],
            }
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["prog", f"--launch-file={launch_file}", "--model=m"]
    )
    args = core.parse_args()
    assert (args.base_url, args.api_key, args.serper_key) == ("http://e", "k", "s")
    assert args.initial_messages == [{"role": "user", "content": "go"}]
    assert not launch_file.exists()


async def test_program_keeps_the_sdk_default_without_the_flag(monkeypatch):
    # NullHarness passes no flag; its program keeps the OpenAI SDK's two retries.
    assert (await client_kwargs(monkeypatch))["max_retries"] == 2


# --- search exclusions ----------------------------------------------------------------


async def test_search_exclusions_reach_the_program_in_its_launch_file_not_argv(
    monkeypatch, tmp_path
):
    launched = await launch(
        monkeypatch,
        search=True,
        env={"SERPER_API_KEY": "s"},
        search_exclude=["https://Bench.Example/", "github.com/some-org/"],
    )
    assert (
        launched.credentials["search_exclude"] == "bench.example\ngithub.com/some-org"
    )
    assert not any("bench.example" in arg for arg in launched.args)
    launch_file = tmp_path / "launch.json"
    launch_file.write_text(
        json.dumps({"base_url": "http://e", "api_key": "k", **launched.credentials})
    )
    monkeypatch.setattr(
        sys, "argv", ["prog", f"--launch-file={launch_file}", "--model=m"]
    )
    assert core.parse_args().search_exclude == ["bench.example", "github.com/some-org"]


def test_search_drops_excluded_sites_and_names_them_in_the_query(monkeypatch):
    sent: dict = {}
    links = [
        "https://www.bench.example/tasks/a",
        "https://github.com/some-org/repo",
        "https://github.com/some-org-fork/repo",
        "https://docs.example.org/page",
    ]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"organic": [{"title": link, "link": link} for link in links]}

    def post(url, json, **kwargs):
        sent.update(json)
        return Response()

    monkeypatch.setattr(core.httpx, "post", post)
    text = core.run_search("q", "key", 5, ["bench.example", "github.com/some-org"])
    assert sent["q"] == "q -site:bench.example -site:github.com/some-org"
    assert "bench.example" not in text and "some-org/repo" not in text
    assert "some-org-fork/repo" in text and "docs.example.org" in text
    assert core.run_search("q", "key", 5) and sent["q"] == "q"


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


def test_pkill_f_on_its_own_command_text_spares_the_shell_running_it():
    # `bash -c <command>` put the text in the shell's argv, so the pattern matched,
    # and killed, the shell before `echo` ran.
    marker = f"vf-selfkill-marker-{os.getpid()}"
    output = core.run_bash(f"pkill -f {marker}; echo survived", 10)
    assert output == "survived\n"


def test_a_command_keeps_its_cwd_env_exit_status_and_leaves_no_script(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("VF_RUN_BASH_PROBE", "probe")
    output = core.run_bash("pwd; echo $VF_RUN_BASH_PROBE; (exit 3); echo $?", 10)
    assert output.splitlines() == [os.getcwd(), "probe", "3"]
    assert list(tmp_path.iterdir()) == []


# --- a full disk ---------------------------------------------------------------------


def full_disk(monkeypatch) -> None:
    """Files fail to open as on a full disk, except on the memory-backed /dev/shm."""
    real_open = os.open

    def open_on_full_disk(path, *args, **kwargs):
        if not os.fspath(path).startswith("/dev/shm/"):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_on_full_disk)


def tool_call_completion(call_id: str, command: str | None) -> ChatCompletion:
    message: dict = {"role": "assistant", "content": None if command else "done"}
    if command is not None:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ]
    return ChatCompletion.model_validate(
        {
            "id": call_id,
            "object": "chat.completion",
            "created": 0,
            "model": "test/fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if command else "stop",
                }
            ],
        }
    )


async def agent_round(monkeypatch, commands: list[str]) -> list[dict]:
    """Run the chat program's loop over a scripted model that issues each command as
    a bash call, then stops. Returns the conversation."""
    # The bundled program defines these at top level; the package module imports them
    # only for type checking.
    for name in ("bound_tool_message", "compactable", "estimated_tokens"):
        monkeypatch.setattr(core, name, getattr(compaction, name), raising=False)
    monkeypatch.setattr(core, "call_mcp", mcp.call_mcp, raising=False)
    replies = [
        tool_call_completion(f"call-{i}", command)
        for i, command in enumerate([*commands, None])
    ]

    class Inbox:
        async def get(self, path, cast_to):
            return {"messages": []}

    class ScriptedModel:
        client = Inbox()
        enabled = False
        threshold = None

        async def complete(self, messages):
            return replies.pop(0), messages

        def reached(self, completion, extra_tokens=0):
            return False

    args = SimpleNamespace(
        tool_interception_url=None,
        bash=True,
        edit=False,
        search=False,
        command_timeout=30,
    )
    messages: list[dict] = []
    await core.run_chat_loop(args, ScriptedModel(), messages, {}, {}, None)
    assert replies == [], "the round ended before the model's last turn"
    return messages


@pytest.mark.parametrize("compacting", [False, True])
@pytest.mark.parametrize("with_tool", [False, True])
async def test_terminal_length_is_not_a_completed_answer_or_tool(compacting, with_tool):
    completion = tool_call_completion(
        "cut", "touch should-not-run" if with_tool else None
    )
    completion.choices[0].finish_reason = "length"
    completion.choices[0].message.content = None

    class Inbox:
        async def get(self, path, cast_to):
            return {"messages": []}

    class Exhausted:
        client = Inbox()
        enabled = compacting
        threshold = 100 if compacting else None

        async def complete(self, messages):
            return completion, messages

    with pytest.raises(RuntimeError, match="finish_reason=length"):
        await core.run_chat_loop(SimpleNamespace(), Exhausted(), [], {}, {}, None)


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="no memory-backed fs")
async def test_a_full_disk_still_runs_the_agents_commands(tmp_path, monkeypatch):
    victim = tmp_path / "big.log"
    victim.write_text("x" * 100)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    full_disk(monkeypatch)
    messages = await agent_round(
        monkeypatch, [f"du -b {victim}", f"rm {victim}; echo freed"]
    )
    results = [m["content"] for m in messages if m["role"] == "tool"]
    assert results == [f"100\t{victim}\n", "freed\n"]
    assert not victim.exists()
    assert messages[-1] == {"role": "assistant", "content": "done"}


async def test_a_command_that_cannot_start_on_a_full_disk_is_a_tool_error(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    full_disk(monkeypatch)
    monkeypatch.setattr(core, "scratch_dir", lambda: None)
    messages = await agent_round(monkeypatch, ["echo unreachable", "echo again"])
    results = [m["content"] for m in messages if m["role"] == "tool"]
    assert len(results) == 2
    assert all(os.strerror(errno.ENOSPC) in result for result in results)
    assert messages[-1] == {"role": "assistant", "content": "done"}


def test_undecodable_output_is_replaced_not_raised():
    assert core.run_bash("printf 'a\\x80b'; printf '\\xff' >&2") == "a�b�"


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child of an exited shell may linger as a zombie until init reaps it, and
    # a reap between the open and the read fails the read with ESRCH.
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().split(") ", 1)[1][0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
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


async def test_bash_installs_its_skills_and_lists_them(tmp_path, monkeypatch):
    host = tmp_path / "host"
    write_skill(
        host,
        "git-forensics",
        '---\nname: "git-forensics"\ndescription: Recover lost commits.\n---\n# Body\n',
    )
    write_skill(host, "untitled", "# No front matter\n")
    backend = SubprocessBackend(SubprocessConfig())
    runtime = backend.bind()
    await backend.start()
    try:
        harness = bash.BashHarness(
            bash.BashHarnessConfig(skills=[host / "git-forensics", host / "untitled"])
        )
        harness.skills_dir = str(tmp_path / "installed")
        monkeypatch.setattr(Runtime, "prepare_uv_script", _no_program)
        await harness.setup(runtime)
        listing = await harness.skills_listing(runtime)
    finally:
        await backend.teardown()
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


async def test_two_sessions_in_one_box_each_list_only_their_own_skills(
    tmp_path, monkeypatch
):
    host = tmp_path / "host"
    write_skill(host, "alpha", "---\nname: alpha\n---\n")
    write_skill(host, "beta", "---\nname: beta\n---\n")
    backend = SubprocessBackend(SubprocessConfig())
    runtime = backend.bind()
    await backend.start()
    monkeypatch.setattr(Runtime, "prepare_uv_script", _no_program)
    try:
        listings = []
        for name in ("alpha", "beta"):
            harness = bash.BashHarness(bash.BashHarnessConfig(skills=[host / name]))
            assert harness.skills_dir == bash.SKILLS_DIR
            harness.skills_dir = str(tmp_path / "shared")
            await harness.setup(runtime)
            listings.append(await harness.skills_listing(runtime))
    finally:
        await backend.teardown()
    assert "`alpha`" in listings[0] and "`beta`" not in listings[0]
    assert "`beta`" in listings[1] and "`alpha`" not in listings[1]


async def test_bash_without_skills_lists_none():
    harness = bash.BashHarness(bash.BashHarnessConfig())
    assert await harness.skills_listing(None) is None


async def _no_program(*args, **kwargs):
    return []
