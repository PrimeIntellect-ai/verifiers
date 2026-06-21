import asyncio
import json

import websockets

from verifiers.v1.types import ConfigData

from ._http import request_json


class CDPError(RuntimeError):
    """Raised when the browser returns a CDP error or discovery fails."""


async def browser_ws_from_http(http_base: str) -> str:
    """Resolve an ``http(s)://host:port`` debugging address to a browser socket."""
    base = http_base.rstrip("/")
    try:
        status, data = await request_json(f"{base}/json/version")
    except Exception as exc:  # noqa: BLE001 - surface a clean error
        raise CDPError(
            f"Could not reach the CDP HTTP endpoint at {base}: {exc}"
        ) from exc
    ws_url = data.get("webSocketDebuggerUrl") if status == 200 else None
    if not isinstance(ws_url, str) or not ws_url:
        raise CDPError(
            f"No webSocketDebuggerUrl from {base}/json/version (status {status})."
        )
    return ws_url


class CDPClient:
    """A CDP WebSocket connection; ``send`` correlates responses by message id."""

    def __init__(self, ws_url: str):
        self._ws_url = ws_url
        self._ws: websockets.ClientConnection | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[ConfigData]] = {}
        self._reader: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        if self._ws is not None:
            return
        # Disable the inbound size cap: screenshots can be large.
        self._ws = await websockets.connect(self._ws_url, max_size=None)
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                message = json.loads(raw)
                msg_id = message.get("id")
                if msg_id is None:
                    continue  # An event; we don't subscribe to any yet.
                future = self._pending.pop(msg_id, None)
                if future is not None and not future.done():
                    future.set_result(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - propagate to waiters
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(exc)
            self._pending.clear()

    async def send(
        self,
        method: str,
        params: ConfigData | None = None,
        *,
        session_id: str | None = None,
    ) -> ConfigData:
        if self._ws is None:
            raise CDPError("CDP client is not connected.")
        self._next_id += 1
        msg_id = self._next_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ConfigData] = loop.create_future()
        self._pending[msg_id] = future
        message: ConfigData = {"id": msg_id, "method": method, "params": params or {}}
        if session_id is not None:
            message["sessionId"] = session_id
        await self._ws.send(json.dumps(message))
        result = await future
        if "error" in result:
            raise CDPError(f"{method} failed: {result['error']}")
        payload = result.get("result")
        return payload if isinstance(payload, dict) else {}

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
            self._reader = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
