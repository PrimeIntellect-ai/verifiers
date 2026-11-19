"""Shared stdlib TCP client and framing for a process-scoped LAN stream."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import uuid

from verifiers.v1.runtimes.rpc_wire import MAX_FRAME, encode

WIRE_SOURCE = r"""
import json
import struct

_MAX_FRAME = 32 * 1024 * 1024

def _read_exact(sock, size):
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise EOFError("LAN execution stream closed")
        buf.extend(chunk)
    return bytes(buf)

def _read_frame(sock):
    size = struct.unpack("!I", _read_exact(sock, 4))[0]
    if size > _MAX_FRAME:
        raise ValueError("LAN execution frame exceeds 32 MiB")
    return json.loads(_read_exact(sock, size))

def _send_frame(sock, value):
    data = json.dumps(value).encode()
    if len(data) > _MAX_FRAME:
        raise ValueError("LAN execution frame exceeds 32 MiB")
    sock.sendall(struct.pack("!I", len(data)) + data)
"""

# Embedded in sandbox-side sync bridges and relays. Pure Python stdlib (no uv/pip).
# A single reader demultiplexes replies while writes, reads and lifecycle calls overlap.
CLIENT_SOURCE = (
    WIRE_SOURCE
    + r"""
import concurrent.futures
import socket
import threading
import time

class LanStream:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.lock = threading.RLock()
        self.send_lock = threading.Lock()
        self.connection = None
        self.pending = {}
        self.closed = False

    def disconnect(self, connection):
        with self.lock:
            if self.connection is connection:
                self.connection = None
                for future in self.pending.values():
                    if not future.done():
                        future.set_exception(OSError("LAN execution stream disconnected"))
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass

    def receive(self, connection):
        try:
            while True:
                response = _read_frame(connection)
                with self.lock:
                    if self.connection is not connection:
                        return
                    future = self.pending.get(response["id"])
                    if future is not None and not future.done():
                        if "error" in response:
                            future.set_exception(RuntimeError(response["error"]))
                        else:
                            future.set_result(response["result"])
        except (EOFError, OSError, ValueError):
            pass
        finally:
            self.disconnect(connection)

    def _connect(self):
        host = self.endpoint["host"]
        port = int(self.endpoint["port"])
        target = (
            socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)[0][4][:2]
            if isinstance(host, str) and host.endswith(".sandbox.internal")
            else (host, port)
        )
        sock = socket.create_connection(target, timeout=10)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _send_frame(sock, {"secret": self.endpoint["secret"]})
            reply = _read_frame(sock)
            if not isinstance(reply, dict) or not reply.get("ok"):
                raise PermissionError(reply.get("error") or "Unauthorized")
            sock.settimeout(None)
            return sock
        except BaseException:
            sock.close()
            raise

    def request(self, body):
        request_id = body["request_id"]
        for attempt in range(3):
            connection = None
            future = concurrent.futures.Future()
            try:
                with self.lock:
                    if self.closed:
                        raise RuntimeError("LAN execution stream is closed")
                    if request_id in self.pending:
                        raise RuntimeError("LAN request is already pending")
                    if self.connection is None:
                        self.connection = self._connect()
                        threading.Thread(
                            target=self.receive, args=(self.connection,), daemon=True
                        ).start()
                    connection = self.connection
                    self.pending[request_id] = future
                with self.send_lock:
                    _send_frame(connection, body)
                return future.result(timeout=3600)
            except PermissionError:
                self.disconnect(connection)
                raise
            except (EOFError, OSError, concurrent.futures.TimeoutError):
                self.disconnect(connection)
                if attempt == 2:
                    raise
                time.sleep(0.2 * (attempt + 1))
            finally:
                with self.lock:
                    if self.pending.get(request_id) is future:
                        del self.pending[request_id]

    def close(self):
        with self.lock:
            self.closed = True
            connection = self.connection
        self.disconnect(connection)
"""
)


async def _read_frame(reader: asyncio.StreamReader) -> dict:
    import json

    size = int.from_bytes(await reader.readexactly(4), "big")
    if size > MAX_FRAME:
        raise ValueError("LAN execution frame exceeds 32 MiB")
    return json.loads(await reader.readexactly(size))


class AsyncLanStream:
    """In-process async counterpart to `LanStream` for LAN peers (A -> B and C -> A)."""

    def __init__(self, endpoint: dict) -> None:
        self.endpoint = endpoint
        self._connect_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._closed = False

    def _disconnect(self, writer: asyncio.StreamWriter | None) -> None:
        if writer is not None and self._writer is writer:
            self._writer = None
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(OSError("LAN execution stream disconnected"))
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()

    async def _receive(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                response = await _read_frame(reader)
                if self._writer is not writer:
                    return
                future = self._pending.get(response["id"])
                if future is not None and not future.done():
                    if "error" in response:
                        future.set_exception(RuntimeError(response["error"]))
                    else:
                        future.set_result(response["result"])
        except (asyncio.IncompleteReadError, OSError, ValueError):
            pass
        finally:
            self._disconnect(writer)

    async def _ensure_connected(self) -> asyncio.StreamWriter:
        if self._writer is not None and not self._writer.is_closing():
            return self._writer
        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("LAN execution stream is closed")
            if self._writer is not None and not self._writer.is_closing():
                return self._writer
            async with asyncio.timeout(10):
                host = self.endpoint["host"]
                port = int(self.endpoint["port"])
                fam = (
                    socket.AF_INET
                    if isinstance(host, str) and host.endswith(".sandbox.internal")
                    else 0
                )
                reader, writer = await asyncio.open_connection(host, port, family=fam)
                try:
                    if sock := writer.get_extra_info("socket"):
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    writer.write(encode({"secret": self.endpoint["secret"]}))
                    await writer.drain()
                    reply = await _read_frame(reader)
                    if not isinstance(reply, dict) or not reply.get("ok"):
                        raise PermissionError(reply.get("error") or "Unauthorized")
                except BaseException:
                    writer.close()
                    with contextlib.suppress(Exception):
                        await writer.wait_closed()
                    raise
            self._writer = writer
            self._reader_task = asyncio.create_task(self._receive(reader, writer))
            return writer

    async def request(self, operation: str, **values) -> dict:
        request_id = uuid.uuid4().hex
        frame = encode(
            {
                "operation": operation,
                "process_id": self.endpoint["process_id"],
                "request_id": request_id,
                **values,
            }
        )
        for attempt in range(3):
            writer = None
            future = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future
            try:
                if self._closed:
                    raise RuntimeError("LAN execution stream is closed")
                writer = await self._ensure_connected()
                async with self._write_lock:
                    writer.write(frame)
                    await writer.drain()
                async with asyncio.timeout(3600):
                    return await future
            except PermissionError:
                self._disconnect(writer)
                raise
            except (asyncio.IncompleteReadError, OSError, TimeoutError):
                self._disconnect(writer)
                if attempt == 2:
                    raise
                await asyncio.sleep(0.2 * (attempt + 1))
            finally:
                if self._pending.get(request_id) is future:
                    del self._pending[request_id]
        raise RuntimeError("unreachable")

    async def aclose(self) -> None:
        self._closed = True
        writer = self._writer
        task, self._reader_task = self._reader_task, None
        self._disconnect(writer)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if writer is not None:
            with contextlib.suppress(Exception):
                await writer.wait_closed()
