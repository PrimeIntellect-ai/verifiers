"""The durable layer: `Box` operations over a runtime through platform outages and lost
replies, the provisioning hold and its bound, `bare_box`; scripted runtimes, no
sandbox."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from verifiers.v1.errors import (
    SandboxError,
    SandboxNotFoundError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from verifiers.v1.runtimes import SubprocessConfig, durable
from verifiers.v1.runtimes.durable import Box, InfraError, Platform

ROUTING = "Connect RPC failed (unavailable): The sandbox routing catalog is temporarily unavailable; retry shortly"


class _Outage:
    """A runtime whose operations fail `down` times with `error` (a typed
    `SandboxUnavailableError` by default, the platform's own outage), then answer; a
    read of an exit-code file says there is none."""

    def __init__(self, down: int, text: str, cls: type = SandboxUnavailableError):
        self.down, self.error, self.calls = down, cls(text), 0
        self.info, self.stopped = SimpleNamespace(id="box-1"), False

    async def _answer(self, result):
        self.calls += 1
        if self.calls <= self.down:
            raise self.error
        return result

    async def read(self, path):
        if path.endswith(".rc"):
            raise SandboxNotFoundError(f"read {path!r}: no such file")
        return await self._answer(b"content")

    async def write(self, path, data):
        return await self._answer(None)

    async def run(self, argv, env):
        return await self._answer(SimpleNamespace(exit_code=0, stdout="0\n", stderr=""))


@pytest.fixture
def platform(monkeypatch) -> Platform:
    """A hold with fast backoff and its log lines kept on `platform.lines`."""
    monkeypatch.setattr(durable, "BACKOFF_START", 0.001)
    monkeypatch.setattr(durable, "BACKOFF_MAX", 0.002)
    lines: list[str] = []
    held = Platform(log=lines.append)
    held.lines = lines
    return held


def box(runtime, platform: Platform | None = None) -> Box:
    return Box(runtime, "/w", home="/h", platform=platform or Platform())


async def test_a_reply_lost_in_transport_is_read_back_from_the_box_or_the_command_is_sent_once_more(
    monkeypatch,
):
    """The wrapper files the output and the exit code first: an empty exec reply after
    the command ran reads them back; an empty reply from a command the box never ran is
    sent again once; a second empty reply is the fault."""
    pause = asyncio.sleep
    monkeypatch.setattr(durable.asyncio, "sleep", lambda s: pause(0))

    class Runtime:
        def __init__(self, files, replies):
            self.files, self.replies, self.runs = files, list(replies), 0
            self.info, self.stopped = SimpleNamespace(id="box-1"), False

        async def run(self, argv, env):
            self.runs += 1
            assert argv[:2] == ["bash", "-c"] and "out-" in argv[2]
            return SimpleNamespace(exit_code=0, stdout=self.replies.pop(0), stderr="")

        async def read(self, path):
            if path.endswith(".rc") and "rc" in self.files:
                return self.files["rc"].encode()
            if not path.endswith(".rc") and "out" in self.files:
                return self.files["out"].encode()
            raise SandboxNotFoundError(f"read {path!r}: no such file")

    ran = Runtime({"rc": "3\n", "out": "hello\n"}, [""])
    assert await box(ran).run("echo hello; exit 3") == (3, "hello\n") and ran.runs == 1
    never_ran = Runtime({}, ["", "6\nhello\n"])
    assert (
        await box(never_ran).run("echo hello") == (0, "hello\n") and never_ran.runs == 2
    )
    with pytest.raises(InfraError, match="command output transport failed"):
        await box(Runtime({}, ["", ""])).run("true")


async def test_a_timed_out_command_says_so_and_its_argv_carries_the_bound():
    class Runtime:
        info, stopped = SimpleNamespace(id="box-1"), False

        async def run(self, argv, env):
            assert argv[:4] == ["timeout", "-k", "30", "7"]
            return SimpleNamespace(exit_code=124, stdout="partial", stderr="")

        async def read(self, path):
            return b"partial output\n"

    rc, out = await box(Runtime()).run("sleep 99", timeout=7)
    assert (
        rc == 124
        and out == "partial output\n\n<command timed out after 7s and was killed>"
    )


async def test_a_failed_upload_is_sent_once_more(monkeypatch):
    """An upload the gateway refused (a bare fault, not the platform's) is sent once
    more; a second refusal is the fault."""
    pause = asyncio.sleep
    monkeypatch.setattr(durable.asyncio, "sleep", lambda s: pause(0))

    class Runtime:
        def __init__(self, failures):
            self.failures, self.writes = failures, 0
            self.info, self.stopped = SimpleNamespace(id="box-1"), False

        async def write(self, path, data):
            self.writes += 1
            if self.writes <= self.failures:
                raise SandboxError("write '/f': Upload failed: HTTP 413: too large")

    once = Runtime(1)
    await box(once).write_bytes("/f", b"x")
    assert once.writes == 2
    with pytest.raises(InfraError, match="write /f failed"):
        await box(Runtime(2)).write_bytes("/f", b"x")


async def test_a_platform_outage_holds_a_box_operation_and_retries_it_without_a_fault(
    platform,
):
    """The class decides, never the text: a `SandboxUnavailableError` (typed from the
    SDK's exception, the HTTP status or the Connect RPC code) holds and retries; the
    notice names the operation, its box and the platform's text."""
    runtime = _Outage(3, "Failed to route request to sandbox box-1")
    assert await box(runtime, platform).read("/x") == "content" and runtime.calls == 4
    assert [line.split(";")[0] for line in platform.lines] == [
        "[platform] sandboxes unavailable since "
        + platform.lines[0].split("since ")[1].split(";")[0],
        "[platform] sandboxes available again after 0s",
    ]
    assert platform.lines[0].endswith(
        "; 1 operations waiting, this one read /x on box box-1: Failed to route request to sandbox box-1"
    )
    await box(
        _Outage(1, "The sandbox is being placed on a node; retry shortly"), platform
    ).write_bytes("/f", b"x")
    # the SDK's `upload_bytes` says `Upload failed: {str(e)}` with the httpx error as its context; the class is read
    # off the chain, so a dropped connection with no text at all is the same typed hold
    dropped = _Outage(2, "write '/k/code.py': Upload failed: ")
    await box(dropped, platform).write_bytes("/k/code.py", b"x")
    assert (
        dropped.calls == 3 and len(platform.lines) == 6
    )  # one notice and one recovery per outage


async def test_an_exec_the_platform_failed_reads_its_exit_code_back_or_is_held_and_sent_again(
    platform,
):
    """An exec that failed `unavailable` may have run (its status poll dropped) or never
    reached the box: the filed exit code decides. Present: the command ran once, its
    reply is read back. Absent: the command is held on the platform and sent again, no
    fault."""

    class Ran:
        def __init__(self):
            self.runs = 0
            self.info, self.stopped = SimpleNamespace(id="box-1"), False

        async def run(self, argv, env):
            self.runs += 1
            raise SandboxUnavailableError(
                "prime exec failed: Request failed: ReadError at POST .../status:batchGet: "
            )

        async def read(self, path):
            return b"7\n" if path.endswith(".rc") else b"done\n"

    ran = Ran()
    assert (
        await box(ran, platform).run("echo done; exit 7") == (7, "done\n")
        and ran.runs == 1
    )
    never_ran = _Outage(2, ROUTING)
    assert await box(never_ran, platform).run("true") == (0, "")
    assert (
        never_ran.calls == 3
        and len([line for line in platform.lines if "unavailable since" in line]) == 1
    )


async def test_an_exec_whose_status_poll_timed_out_waits_for_the_filed_exit_code_and_is_never_sent_again(
    monkeypatch,
):
    """One batched status poll of the SDK timed out across a stall of the run loop
    (`prime exec failed: Request timed out: `, a `SandboxTimeoutError`) while the
    commands ran on. The exec takes the lost-reply path and waits: the exit code the
    wrapper files is read again every `POLL_S` seconds up to the command's bound, the
    command is not sent again and nothing is held on the platform; past the bound with
    no exit code it is the fault."""
    monkeypatch.setattr(durable, "POLL_S", 0.001)
    monkeypatch.setattr(durable, "HANG_GRACE", 0)
    platform = Platform()

    class Runtime:
        def __init__(self, files_after: int | None):
            self.files_after, self.runs, self.reads = files_after, 0, 0
            self.info, self.stopped = SimpleNamespace(id="box-1"), False

        async def run(self, argv, env):
            self.runs += 1
            raise SandboxTimeoutError("prime exec failed: Request timed out: ")

        async def read(self, path):
            self.reads += 1
            if self.files_after is None or self.reads <= self.files_after:
                raise SandboxNotFoundError(f"read {path!r}: no such file")
            return b"5\n" if path.endswith(".rc") else b"late\n"

    slow = Runtime(files_after=4)
    assert await box(slow, platform).run("sleep 9; echo late; exit 5", timeout=2) == (
        5,
        "late\n",
    )
    assert (
        slow.runs == 1 and slow.reads == 6 and platform.since is None
    )  # four polls, then the code and output
    never = Runtime(files_after=None)
    with pytest.raises(
        InfraError, match="status poll timed out and no exit code was filed within 1s"
    ):
        await box(never, platform).run("true", timeout=1)
    assert never.runs == 1 and never.reads > 10


async def test_a_platform_outage_past_its_budget_surfaces_the_error_and_a_bare_fault_is_never_held(
    platform,
):
    """The budget is read at each hold (a run's settings may change): zero means no hold
    at all."""
    platform.budget = lambda: 0
    runtime = _Outage(5, ROUTING)
    with pytest.raises(
        InfraError,
        match="read /x failed: Connect RPC failed .unavailable.: The sandbox",
    ):
        await box(runtime, platform).read("/x")
    assert runtime.calls == 1 and platform.lines == []
    plain = _Outage(5, "exec failed: exit 137", SandboxError)
    with pytest.raises(InfraError, match="exec failed: exec failed"):
        await box(plain, platform).run("true")
    assert plain.calls == 1
    # the same texts as a bare `SandboxError` (nothing typed behind them) are the caller's fault, not the platform's
    for text in (
        ROUTING,
        "The sandbox is being placed on a node; retry shortly",
        "read '/f': Download failed: ",
    ):
        with pytest.raises(InfraError):
            await box(_Outage(1, text, SandboxError), platform).read("/f")
    assert platform.lines == []


async def test_a_closing_box_refuses_operations_and_a_hung_operation_is_abandoned():
    class Runtime:
        info, stopped = SimpleNamespace(id="box-1"), True

        async def read(self, path):
            return b""

    with pytest.raises(InfraError, match="read /f refused: box box-1 is closing"):
        await box(Runtime()).read("/f")

    class Hung:
        info, stopped = SimpleNamespace(id="box-1"), False

        async def read(self, path):
            await asyncio.sleep(60)

    with pytest.raises(InfraError, match="read /f hung past 0.01s"):
        await Box(Hung(), "/w", op_timeout=0.01).read("/f")


async def test_box_read_treats_a_typed_missing_path_as_absent_and_any_other_fault_as_infra():
    class Runtime:
        def __init__(self, error):
            self.error = error
            self.info, self.stopped = SimpleNamespace(id="box-1"), False

        async def read(self, path):
            raise self.error

    assert (
        await box(
            Runtime(SandboxNotFoundError("read '/runs.jsonl': no such file"))
        ).read("/f")
        is None
    )
    with pytest.raises(InfraError, match="read /f failed: No such file or directory"):
        await box(Runtime(SandboxError("No such file or directory"))).read("/f")


async def test_a_large_output_is_read_back_in_parts(monkeypatch):
    """An output over the inline size is read from the box; one over a transport part is
    split and read part by part, the parts cleaned after."""
    monkeypatch.setattr(durable, "INLINE_OUTPUT", 10)
    monkeypatch.setattr(durable, "READ_PART", 8)
    commands: list[str] = []

    class Runtime:
        info, stopped = SimpleNamespace(id="box-1"), False

        async def run(self, argv, env):
            commands.append(argv[-1])
            if "split -b" in argv[-1] or "rm -f" in argv[-1]:
                return SimpleNamespace(exit_code=0, stdout="0\n", stderr="")
            return SimpleNamespace(exit_code=0, stdout="20\n", stderr="")

        async def read(self, path):
            assert ".part." in path
            return {"0000": b"01234567", "0001": b"89abcdef", "0002": b"ghij"}[
                path.rsplit(".", 1)[1]
            ]

    rc, out = await box(Runtime()).run("printf ...")
    assert (rc, out) == (0, "0123456789abcdefghij")
    assert any("split -b 8 -d -a 4" in command for command in commands)
    assert "rm -f" in commands[-1] and ".part.*" in commands[-1]


async def test_write_with_a_mode_runs_chmod_and_a_refused_chmod_is_the_fault():
    class Runtime:
        def __init__(self, rc):
            self.rc, self.writes, self.commands = rc, [], []
            self.info, self.stopped = SimpleNamespace(id="box-1"), False

        async def write(self, path, data):
            self.writes.append((path, data))

        async def run(self, argv, env):
            self.commands.append(argv[-1])
            return SimpleNamespace(
                exit_code=self.rc,
                stdout=f"{self.rc}\nnope\n" if self.rc else "0\n",
                stderr="",
            )

    fine = Runtime(0)
    await box(fine).write("/f", "x", mode="755")
    assert fine.writes == [("/f", b"x")] and "chmod 755 /f" in fine.commands[-1]
    with pytest.raises(InfraError, match="chmod /f failed"):
        await box(Runtime(1)).write("/f", "x", mode="755")


async def test_a_provisioning_during_a_platform_outage_or_behind_the_pacing_holds_and_retries_without_a_fault(
    platform,
):
    """A create the platform failed `unavailable`, or one that timed out before any box
    existed (the creation limiter refusing a backlog), is held and tried again, never a
    fault; any other sandbox fault is an `InfraError`."""
    pacing = "prime sandbox provisioning failed: prime-sandbox creation limiter backlog of 300.9s exceeds 300s (/tmp/x)"
    failures = [SandboxUnavailableError(ROUTING), SandboxTimeoutError(pacing)]
    runtime = SimpleNamespace(info=SimpleNamespace(id="box-1"), stopped=False)
    attempts = 0

    @asynccontextmanager
    async def provision():
        nonlocal attempts
        attempts += 1
        if failures:
            raise failures.pop(0)
        yield runtime

    async with durable.provisioned(provision, platform=platform) as leased:
        assert leased is runtime
    assert attempts == 3 and len(platform.lines) == 2

    @asynccontextmanager
    async def broken():
        raise SandboxError("the box never came up")
        yield

    with pytest.raises(
        InfraError, match="box provisioning failed: the box never came up"
    ):
        await durable.provision_box(broken, platform=platform)


async def test_a_provisioning_past_its_bound_is_a_fault_and_a_box_it_makes_late_is_closed():
    closed = []

    @asynccontextmanager
    async def slow():
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            await asyncio.sleep(0.02)
        try:
            yield SimpleNamespace(info=SimpleNamespace(id="late"), stopped=False)
        finally:
            closed.append("late")

    with pytest.raises(InfraError, match="no box within 0s"):
        await durable.provision_box(slow, wait=0)
    await asyncio.sleep(0.2)
    assert closed == ["late"]


async def test_a_bare_box_is_provisioned_from_its_config_alone_and_closed_after(
    monkeypatch,
):
    provisioned, closed = [], []

    @asynccontextmanager
    async def provision_runtime(config):
        provisioned.append(config)
        yield SimpleNamespace(info=SimpleNamespace(id="bare-1"), stopped=False)
        closed.append(config)

    monkeypatch.setattr("verifiers.v1.runtimes.provision_runtime", provision_runtime)
    config = SubprocessConfig()
    async with durable.bare_box(config, workdir="/", home="/h") as bare:
        assert isinstance(bare, Box) and (
            bare.id,
            bare.workdir,
            bare.home,
            bare.tmp,
        ) == ("bare-1", "/", "/h", "/h/tmp")
        assert not closed
    assert provisioned == [config] and closed == [config]


async def test_a_bare_subprocess_box_runs_reads_and_writes(tmp_path):
    """The durable layer over a real subprocess runtime: a command's exit code and
    output, a file written and read."""
    async with durable.bare_box(
        SubprocessConfig(), workdir=str(tmp_path), home=str(tmp_path / ".box")
    ) as bare:
        assert await bare.run("echo hi; echo err >&2; exit 4") == (4, "hi\nerr\n")
        await bare.write(str(tmp_path / "f.sh"), "echo ran", mode="755")
        assert await bare.run("./f.sh") == (0, "ran\n")
        assert await bare.read(str(tmp_path / "f.sh")) == "echo ran"
        assert await bare.read(str(tmp_path / "missing")) is None
        assert (tmp_path / ".box" / "tmp").is_dir()


async def _swallow():
    """An SDK call that answers the task's cancellation with an ordinary exception, as
    connectrpc does."""
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        raise RuntimeError("Request was cancelled") from None


async def test_check_cancelled_raises_the_cancellation_a_library_swallowed():
    checked = []

    async def work():
        durable.check_cancelled()  # nothing pending: a no-op
        try:
            await _swallow()
        except RuntimeError:
            checked.append(durable.cancel_requested())
            durable.check_cancelled()
        checked.append("continued")

    task = asyncio.create_task(work())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert checked == [True]
    assert not durable.cancel_requested()
