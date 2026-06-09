"""Backend behaviour with HTTP faked (no network, no browser)."""

import pytest

import verifiers.v1.toolsets.browser.backends as backends_mod
import verifiers.v1.toolsets.browser.cdp as cdp_mod
from verifiers.v1.toolsets.browser import BrowserbaseBackend, CDPBackend


@pytest.fixture
def fake_http(monkeypatch):
    calls = []

    async def fake_request_json(
        url, *, method="GET", headers=None, body=None, timeout=30.0
    ):
        calls.append({"url": url, "method": method, "headers": headers, "body": body})
        if "/json/version" in url:
            return 200, {"webSocketDebuggerUrl": "ws://browser/devtools/browser/abc"}
        if url.endswith("/sessions") and method == "POST":
            return 201, {
                "id": "bb-session-1",
                "connectUrl": "wss://connect.browserbase.com/x",
            }
        if "/sessions/" in url and method == "POST":
            return 200, {}
        return 404, {}

    monkeypatch.setattr(backends_mod, "request_json", fake_request_json)
    monkeypatch.setattr(cdp_mod, "request_json", fake_request_json)
    return calls


async def test_cdp_backend_passthrough_ws():
    handle = await CDPBackend(cdp_url="ws://host/devtools/browser/x").create()
    assert handle.cdp_ws_url == "ws://host/devtools/browser/x"
    assert handle.session_id == ""


async def test_cdp_backend_resolves_http(fake_http):
    handle = await CDPBackend(cdp_url="http://localhost:9222").create()
    assert handle.cdp_ws_url == "ws://browser/devtools/browser/abc"


async def test_cdp_backend_requires_url(monkeypatch):
    monkeypatch.delenv("BROWSERTOOLSET_CDP_URL", raising=False)
    with pytest.raises(RuntimeError):
        await CDPBackend().create()


async def test_browserbase_create(fake_http):
    backend = BrowserbaseBackend(api_key="k", project_id="p")
    handle = await backend.create()
    assert handle.session_id == "bb-session-1"
    assert handle.cdp_ws_url == "wss://connect.browserbase.com/x"
    post = fake_http[0]
    assert post["headers"]["X-BB-API-Key"] == "k"
    assert post["body"]["projectId"] == "p"


async def test_browserbase_requires_api_key(monkeypatch):
    monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        await BrowserbaseBackend(project_id="p").create()


async def test_browserbase_close_releases_session(fake_http):
    backend = BrowserbaseBackend(api_key="k", project_id="p")
    await backend.close("bb-session-1")
    release = fake_http[-1]
    assert release["url"].endswith("/sessions/bb-session-1")
    assert release["body"]["status"] == "REQUEST_RELEASE"


async def test_browserbase_close_noop_on_empty():
    await BrowserbaseBackend(api_key="k", project_id="p").close("")
