import asyncio
import json
import urllib.error
import urllib.request

from verifiers.v1.types import ConfigData


async def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: ConfigData | None = None,
    timeout: float = 30.0,
) -> tuple[int, ConfigData]:
    """HTTP request returning (status, parsed json object); 4xx/5xx return, not raise."""

    def _do() -> tuple[int, ConfigData]:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        hdrs = dict(headers or {})
        if payload is not None:
            hdrs.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(url, data=payload, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8")
                status = response.status
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            status = exc.code
        parsed = json.loads(text) if text.strip() else {}
        return status, parsed if isinstance(parsed, dict) else {}

    return await asyncio.to_thread(_do)
