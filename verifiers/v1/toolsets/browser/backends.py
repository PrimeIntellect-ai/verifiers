import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from verifiers.v1.types import ConfigData

from ._http import request_json
from .cdp import browser_ws_from_http

BROWSERBASE_API_URL = "https://api.browserbase.com/v1"
_SESSION_CREATE_TIMEOUT_S = 60.0
_SUCCESS_STATUSES = {200, 201, 202}


@dataclass(frozen=True, slots=True)
class BrowserSessionHandle:
    """The result of provisioning a browser: an id and a CDP socket to drive it."""

    session_id: str
    cdp_ws_url: str


@runtime_checkable
class BrowserBackend(Protocol):
    """Provisions and releases browser sessions, returning CDP endpoints."""

    async def create(self) -> BrowserSessionHandle:
        """Provision a browser and return a handle to its CDP endpoint."""
        ...

    async def close(self, session_id: str) -> None:
        """Release the session identified by ``session_id`` (best effort)."""
        ...


class CDPBackend:
    """Connect to an existing browser via a ws(s):// socket or http(s)://host:port."""

    def __init__(self, cdp_url: str | None = None):
        self._cdp_url = cdp_url

    def _url(self) -> str:
        url = self._cdp_url or os.environ.get("BROWSERTOOLSET_CDP_URL", "")
        if not url:
            raise RuntimeError(
                "CDPBackend requires a cdp_url (or $BROWSERTOOLSET_CDP_URL): a "
                "ws(s):// browser socket or an http(s)://host:port address."
            )
        return url

    async def create(self) -> BrowserSessionHandle:
        url = self._url()
        scheme = url.split(":", 1)[0].lower()
        ws_url = url if scheme in ("ws", "wss") else await browser_ws_from_http(url)
        return BrowserSessionHandle(session_id="", cdp_ws_url=ws_url)

    async def close(self, session_id: str) -> None:
        return None


class BrowserbaseBackend:
    """Provision an isolated Browserbase session per rollout via its REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        *,
        proxies: bool = False,
        keep_alive: bool = False,
        session_create_kwargs: ConfigData | None = None,
    ):
        self._api_key = api_key
        self._project_id = project_id
        self._proxies = proxies
        self._keep_alive = keep_alive
        self._session_create_kwargs = dict(session_create_kwargs or {})

    def _require_api_key(self) -> str:
        value = (self._api_key or os.environ.get("BROWSERBASE_API_KEY", "")).strip()
        if not value:
            raise RuntimeError(
                "BROWSERBASE_API_KEY is not configured. Pass api_key=... or set "
                "the environment variable before creating a session."
            )
        return value

    def _require_project_id(self) -> str:
        value = (
            self._project_id or os.environ.get("BROWSERBASE_PROJECT_ID", "")
        ).strip()
        if not value:
            raise RuntimeError(
                "BROWSERBASE_PROJECT_ID is not configured. Pass project_id=... or "
                "set the environment variable before creating a session."
            )
        return value

    def _headers(self) -> dict[str, str]:
        return {
            "X-BB-API-Key": self._require_api_key(),
            "Content-Type": "application/json",
        }

    async def create(self) -> BrowserSessionHandle:
        payload: ConfigData = {
            "projectId": self._require_project_id(),
            "proxies": self._proxies,
            "keepAlive": self._keep_alive,
            **self._session_create_kwargs,
        }
        status, data = await request_json(
            f"{BROWSERBASE_API_URL}/sessions",
            method="POST",
            headers=self._headers(),
            body=payload,
            timeout=_SESSION_CREATE_TIMEOUT_S,
        )
        if status not in _SUCCESS_STATUSES:
            raise RuntimeError(f"Failed to create Browserbase session: {status} {data}")
        session_id = data.get("id")
        connect_url = data.get("connectUrl")
        if not isinstance(session_id, str) or not isinstance(connect_url, str):
            raise RuntimeError(f"Browserbase response missing id/connectUrl: {data}")
        return BrowserSessionHandle(session_id=session_id, cdp_ws_url=connect_url)

    async def close(self, session_id: str) -> None:
        if not session_id:
            return
        try:
            await request_json(
                f"{BROWSERBASE_API_URL}/sessions/{session_id}",
                method="POST",
                headers=self._headers(),
                body={
                    "projectId": self._require_project_id(),
                    "status": "REQUEST_RELEASE",
                },
            )
        except Exception:  # noqa: BLE001 - close is best effort
            return
