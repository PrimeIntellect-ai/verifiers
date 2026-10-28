"""ACP stdout EOF diagnostics retain process status and bounded stderr context."""

import asyncio
import json

import pytest

from verifiers.v1.acp import ACPHarnessSession, _packet, _PacketReader

_EOF = object()


class _Process:
    def __init__(
        self,
        *,
        poll_status: int | None = None,
        wait_status: int | None = None,
        respond: bool = False,
        block_poll: bool = False,
    ) -> None:
        self.poll_status = poll_status
        self.wait_status = wait_status
        self.respond = respond
        self.block_poll = block_poll
        self.requests = []
        self.written = asyncio.Event()
        self.poll_started = asyncio.Event()
        self.poll_release = asyncio.Event()
        self.poll_calls = 0
        self.wait_calls = 0
        self._output = asyncio.Queue()

    async def stdout(self):
        while True:
            item = await self._output.get()
            if item is _EOF:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def write(self, data: bytes) -> None:
        request = json.loads(data[8:].decode())
        self.requests.append(request)
        self.written.set()
        if self.respond:
            self._output.put_nowait(
                _packet(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {"operation": request["operation"]},
                    }
                )
            )

    async def poll(self) -> int | None:
        self.poll_calls += 1
        if self.block_poll:
            self.poll_started.set()
            await self.poll_release.wait()
        return self.poll_status

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.wait_status is None:
            raise TimeoutError
        return self.wait_status

    def disconnect(self, error: BaseException | object = _EOF) -> None:
        self._output.put_nowait(error)


async def _read_stream(reader: asyncio.StreamReader):
    while chunk := await reader.read(1024):
        yield chunk


class _StreamProcess(_Process):
    def __init__(self) -> None:
        super().__init__(poll_status=0)
        self.stdout_reader = asyncio.StreamReader()
        self.stderr_reader = asyncio.StreamReader()
        self.stderr = _read_stream(self.stderr_reader)

    def stdout(self):
        return _read_stream(self.stdout_reader)


def _make_session(process: _Process) -> ACPHarnessSession:
    session = object.__new__(ACPHarnessSession)
    session._process = process
    session._reader = _PacketReader(process.stdout())
    session._stderr_tail = bytearray()
    session._stderr_task = None
    session._lock = asyncio.Lock()
    session._write_lock = asyncio.Lock()
    session._responses = {}
    session._request_id = 0
    session._disconnected = False
    session._response_task = asyncio.create_task(
        session._read_responses(session._reader)
    )
    return session


async def _pending_request(session: ACPHarnessSession, process: _Process):
    request = asyncio.create_task(session._request("prompt"))
    await process.written.wait()
    return request


@pytest.mark.parametrize(
    ("poll_status", "wait_status", "exit_code"),
    [(0, None, 0), (None, 17, 17)],
)
async def test_stdout_eof_reports_exit_status_and_stderr(
    poll_status, wait_status, exit_code
):
    process = _Process(poll_status=poll_status, wait_status=wait_status)
    session = _make_session(process)
    session._stderr_tail.extend(b"runner failed\n")
    request = await _pending_request(session, process)
    process.disconnect()

    with pytest.raises(RuntimeError) as caught:
        await request

    assert str(caught.value) == (
        "ACP connection closed: ACP process closed its stdout\n"
        f"ACP process exited with code {exit_code}\n\n"
        "ACP process stderr:\nrunner failed"
    )
    assert process.poll_calls == 1
    assert process.wait_calls == (1 if poll_status is None else 0)


async def test_stdout_eof_without_exit_status_or_stderr_preserves_original_error():
    process = _Process()
    session = _make_session(process)
    request = await _pending_request(session, process)
    process.disconnect()

    with pytest.raises(RuntimeError) as caught:
        await request

    assert str(caught.value) == "ACP connection closed: ACP process closed its stdout"
    assert process.poll_calls == 1
    assert process.wait_calls == 1


async def test_stdout_stream_fault_does_not_invent_process_status():
    process = _Process(poll_status=9, wait_status=9)
    session = _make_session(process)
    session._stderr_tail.extend(b"unrelated stderr")
    request = await _pending_request(session, process)
    process.disconnect(RuntimeError("stdout read failed"))

    with pytest.raises(RuntimeError) as caught:
        await request

    assert str(caught.value) == "ACP connection closed: stdout read failed"
    assert process.poll_calls == 0
    assert process.wait_calls == 0


async def test_normal_response_and_shutdown_do_not_collect_eof_diagnostics():
    process = _Process(respond=True)
    session = _make_session(process)

    response = await session._request("prompt")
    shutdown = await session._request("shutdown")

    assert response["result"] == {"operation": "prompt"}
    assert shutdown["result"] == {"operation": "shutdown"}
    process.disconnect()
    await session._response_task
    assert process.poll_calls == 0
    assert process.wait_calls == 0


async def test_stdout_eof_does_not_replay_request():
    process = _Process(poll_status=0)
    session = _make_session(process)
    request = await _pending_request(session, process)
    process.disconnect()

    with pytest.raises(RuntimeError, match="ACP process closed its stdout"):
        await request
    with pytest.raises(RuntimeError, match="ACP process is not running"):
        await session._request("prompt")

    assert [request["operation"] for request in process.requests] == ["prompt"]


async def test_request_during_eof_diagnostics_cannot_write():
    process = _Process(poll_status=0, block_poll=True)
    session = _make_session(process)
    request = await _pending_request(session, process)
    process.disconnect()
    await process.poll_started.wait()

    with pytest.raises(RuntimeError, match="ACP process is not running"):
        await session._request("prompt")

    assert [request["operation"] for request in process.requests] == ["prompt"]
    process.poll_release.set()
    with pytest.raises(RuntimeError, match="ACP process closed its stdout"):
        await request
    await session._response_task


async def test_cancelled_stderr_diagnostics_settle_request_and_preserve_cancellation():
    process = _StreamProcess()
    session = _make_session(process)
    session._stderr_task = asyncio.create_task(session._drain_stderr(process.stderr))
    request = await _pending_request(session, process)
    process.stdout_reader.feed_eof()
    await asyncio.sleep(0)

    session._response_task.cancel()

    with pytest.raises(RuntimeError, match="ACP connection closed"):
        await request
    with pytest.raises(asyncio.CancelledError):
        await session._response_task
    assert not session._responses


async def test_stdout_eof_drains_buffered_stderr_from_real_streams():
    process = _StreamProcess()
    session = _make_session(process)
    session._stderr_task = asyncio.create_task(session._drain_stderr(process.stderr))
    request = await _pending_request(session, process)

    process.stdout_reader.feed_eof()
    process.stderr_reader.feed_data(b"runner failed\n")
    process.stderr_reader.feed_eof()

    with pytest.raises(RuntimeError) as caught:
        await request

    assert str(caught.value) == (
        "ACP connection closed: ACP process closed its stdout\n"
        "ACP process exited with code 0\n\n"
        "ACP process stderr:\nrunner failed"
    )
    await session._stderr_task
