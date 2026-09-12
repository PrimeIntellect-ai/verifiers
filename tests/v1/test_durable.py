"""The durable layer: `Box` operations over a runtime (lost replies read back, an exec
held through an outage, a large output in parts, one resend of an upload, every command
bounded), `provisioned`; scripted runtimes and one real subprocess box, no sandbox."""

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
from verifiers.v1.runtimes import SubprocessConfig, durable, provision_runtime
from verifiers.v1.runtimes.durable import Box, InfraError
from verifiers.v1.utils import aio

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
def fast(monkeypatch):
    """The exec hold's backoff made instant."""
    monkeypatch.setattr(durable, "BACKOFF_START", 0.001)
    monkeypatch.setattr(durable, "BACKOFF_MAX", 0.002)


def box(runtime, **kw) -> Box:
    return Box(runtime, "/w", home="/h", **kw)


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
            assert argv[:4] == ["timeout", "-k", "30", "900"] and "out-" in argv[-1]
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


async def test_every_command_is_bounded_and_a_timed_out_one_says_so():
    """A command given no timeout runs under the box's `op_timeout`; one past its bound
    is killed (`timeout -k 30`) and its output says so."""

    bounds: list[str] = []

    class Runtime:
        info, stopped = SimpleNamespace(id="box-1"), False

        async def run(self, argv, env):
            bounds.append(argv[3])
            assert argv[:3] == ["timeout", "-k", "30"]
            return SimpleNamespace(exit_code=124, stdout="partial", stderr="")

        async def read(self, path):
            return b"partial output\n"

    rc, out = await box(Runtime()).run("sleep 99", timeout=7)
    assert (
        rc == 124
        and out == "partial output\n\n<command timed out after 7s and was killed>"
    )
    await box(Runtime(), op_timeout=42).run("sleep 99")
    assert bounds == ["7", "42"]


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

    class Gone(Runtime):
        async def write(self, path, data):
            self.writes += 1
            raise SandboxNotFoundError("write '/f': box box-1 lost its placement")

    gone = Gone(2)
    with pytest.raises(
        InfraError, match="lost its placement"
    ):  # a box that is gone gets nothing twice
        await box(gone).write_bytes("/f", b"x")
    assert gone.writes == 1


async def test_an_exec_the_platform_failed_reads_its_exit_code_back_or_is_held_and_sent_again(
    fast,
):
    """An exec that failed `unavailable` may have run (its status poll dropped) or never
    reached the box: the filed exit code decides. Present: the command ran once, its
    reply is read back. Absent: the command is held and sent again within the outage
    budget, no fault; past the budget the fault surfaces with the typed cause."""

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
    assert await box(ran).run("echo done; exit 7") == (7, "done\n") and ran.runs == 1
    never_ran = _Outage(2, ROUTING)
    assert await box(never_ran).run("true") == (0, "")
    assert never_ran.calls == 3
    spent = _Outage(5, ROUTING)
    with pytest.raises(InfraError, match="exec failed: Connect RPC failed") as info:
        await box(spent, outage_s=0).run("true")
    assert spent.calls == 1 and isinstance(
        info.value.__cause__, SandboxUnavailableError
    )


async def test_an_exec_whose_status_poll_timed_out_waits_for_the_filed_exit_code_and_is_never_sent_again(
    monkeypatch,
):
    """One batched status poll of the SDK timed out across a stall of the run loop
    (`prime exec failed: Request timed out: `, a `SandboxTimeoutError`) while the
    commands ran on. The exec takes the lost-reply path and waits: the exit code the
    wrapper files is read again every `POLL_S` seconds up to the command's bound, the
    command is not sent again; past the bound with no exit code it is the fault."""
    monkeypatch.setattr(durable, "POLL_S", 0.001)
    monkeypatch.setattr(durable, "HANG_GRACE", 0)

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
    assert await box(slow).run("sleep 9; echo late; exit 5", timeout=2) == (5, "late\n")
    assert slow.runs == 1 and slow.reads == 6  # four polls, then the code and output
    never = Runtime(files_after=None)
    with pytest.raises(
        InfraError, match="status poll timed out and no exit code was filed within 1s"
    ):
        await box(never).run("true", timeout=1)
    assert never.runs == 1 and never.reads > 10


async def test_a_read_or_write_is_never_held_here_and_a_bare_fault_is_the_callers():
    """The runtime retries its own idempotent reads and writes (`PrimeConfig.
    outage_budget_s`); what reaches the box is the fault, typed as its cause."""
    runtime = _Outage(1, ROUTING)
    with pytest.raises(
        InfraError,
        match="read /x failed: Connect RPC failed .unavailable.: The sandbox",
    ) as info:
        await box(runtime).read("/x")
    assert runtime.calls == 1 and isinstance(
        info.value.__cause__, SandboxUnavailableError
    )
    plain = _Outage(5, "exec failed: exit 137", SandboxError)
    with pytest.raises(InfraError, match="exec failed: exec failed"):
        await box(plain).run("true")
    assert plain.calls == 1
    for text in (ROUTING, "The sandbox is being placed on a node; retry shortly"):
        with pytest.raises(InfraError):
            await box(_Outage(1, text, SandboxError)).read("/f")


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


async def test_provisioned_types_a_provisioning_fault_and_leaves_the_block_its_own():
    """A sandbox fault raised by the provisioning context is an `InfraError` with the
    typed cause; one raised inside the block passes as it is; the context is left on
    exit."""
    runtime = SimpleNamespace(info=SimpleNamespace(id="box-1"), stopped=False)
    left = []

    @asynccontextmanager
    async def provision():
        try:
            yield runtime
        finally:
            left.append(runtime)

    async with durable.provisioned(provision) as leased:
        assert leased is runtime
    assert left == [runtime]
    with pytest.raises(SandboxError, match="inside"):
        async with durable.provisioned(provision):
            raise SandboxError("inside")

    @asynccontextmanager
    async def broken():
        raise SandboxTimeoutError(
            "prime sandbox provisioning failed: no box within 480s"
        )
        yield

    with pytest.raises(
        InfraError, match="box provisioning failed: prime sandbox provisioning"
    ) as info:
        async with durable.provisioned(broken):
            pass
    assert isinstance(info.value.__cause__, SandboxTimeoutError)


async def test_a_subprocess_box_runs_reads_and_writes(tmp_path):
    """The durable layer over a real subprocess runtime: a command's exit code and
    output, a file written and read, a missing path None."""
    async with durable.provisioned(
        lambda: provision_runtime(SubprocessConfig())
    ) as runtime:
        bare = Box(runtime, str(tmp_path), home=str(tmp_path / ".box"))
        assert await bare.run("echo hi; echo err >&2; exit 4") == (4, "hi\nerr\n")
        assert await bare.run("echo $VF_A-$VF_B", env={"VF_A": "a", "VF_B": "b c"}) == (
            0,
            "a-b c\n",
        )
        await bare.write(str(tmp_path / "f.sh"), "echo ran", mode="755")
        assert await bare.run("./f.sh") == (0, "ran\n")
        assert await bare.read(str(tmp_path / "f.sh")) == b"echo ran"
        assert await bare.read(str(tmp_path / "missing")) is None
        assert (tmp_path / ".box" / "tmp").is_dir()
        with pytest.raises(InfraError, match="write .* failed"):
            await bare.write_bytes(str(tmp_path / "f.sh" / "under-a-file"), b"x")


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
        aio.check_cancelled()  # nothing pending: a no-op
        try:
            await _swallow()
        except RuntimeError:
            checked.append(aio.cancel_requested())
            aio.check_cancelled()
        checked.append("continued")

    task = asyncio.create_task(work())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert checked == [True]
    assert not aio.cancel_requested()
