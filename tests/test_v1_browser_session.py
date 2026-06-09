"""Session + tool behaviour against a fake backend + CDP client (no browser)."""

import base64

import pytest

import verifiers.v1.toolsets.browser.session as session_mod
from verifiers.v1.toolsets.browser import BrowserSession, tools
from verifiers.v1.toolsets.browser.backends import BrowserSessionHandle


class FakeBackend:
    def __init__(self):
        self.created = 0
        self.closed = []

    async def create(self):
        self.created += 1
        return BrowserSessionHandle(session_id="sess-1", cdp_ws_url="ws://fake/browser")

    async def close(self, session_id):
        self.closed.append(session_id)


class FakeCDPClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.calls = []

    async def connect(self):
        self.calls.append(("connect", {}, None))

    async def send(self, method, params=None, *, session_id=None):
        self.calls.append((method, params or {}, session_id))
        if method == "Target.createTarget":
            return {"targetId": "target-1"}
        if method == "Target.attachToTarget":
            return {"sessionId": "cdp-sess-1"}
        if method == "Page.captureScreenshot":
            return {"data": base64.b64encode(b"fake-png").decode()}
        return {}

    async def close(self):
        self.calls.append(("close", {}, None))


@pytest.fixture
def fake_session(monkeypatch):
    monkeypatch.setattr(session_mod, "CDPClient", FakeCDPClient)
    return BrowserSession(FakeBackend(), width=800, height=600)


def _page_calls(client):
    """CDP calls scoped to the attached page session."""
    return [c for c in client.calls if c[2] == "cdp-sess-1"]


async def test_start_attaches_page_and_enables_domains(fake_session):
    await fake_session.start()
    methods = [c[0] for c in fake_session._client.calls]
    assert "Target.createTarget" in methods
    assert "Target.attachToTarget" in methods
    page_methods = [c[0] for c in _page_calls(fake_session._client)]
    assert "Page.enable" in page_methods
    assert "Emulation.setDeviceMetricsOverride" in page_methods


async def test_page_commands_carry_session_id(fake_session):
    await tools.left_click([100, 150], session=fake_session)
    mouse = [
        c for c in fake_session._client.calls if c[0] == "Input.dispatchMouseEvent"
    ]
    assert mouse and all(c[2] == "cdp-sess-1" for c in mouse)


async def test_left_click_dispatches_press_release(fake_session):
    result = await tools.left_click([100, 150], session=fake_session)
    mouse = [
        c for c in fake_session._client.calls if c[0] == "Input.dispatchMouseEvent"
    ]
    assert mouse[0][1]["type"] == "mousePressed"
    assert mouse[0][1]["x"] == 100 and mouse[0][1]["y"] == 150
    assert mouse[1][1]["type"] == "mouseReleased"
    assert result[0]["type"] == "image_url"
    assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert fake_session.cursor_position == (100, 150)


async def test_double_click_count(fake_session):
    await tools.double_click([10, 20], session=fake_session)
    presses = [
        c
        for c in fake_session._client.calls
        if c[0] == "Input.dispatchMouseEvent" and c[1].get("type") == "mousePressed"
    ]
    assert [p[1]["clickCount"] for p in presses] == [1, 2]


async def test_type_text_uses_insert_text(fake_session):
    await tools.type_text("hello", session=fake_session)
    insert = [c for c in fake_session._client.calls if c[0] == "Input.insertText"]
    assert insert and insert[0][1]["text"] == "hello"


async def test_key_chord_sets_modifier(fake_session):
    await tools.key("ctrl+s", session=fake_session)
    key_events = [
        c for c in fake_session._client.calls if c[0] == "Input.dispatchKeyEvent"
    ]
    assert key_events[0][1]["type"] == "keyDown"
    assert key_events[0][1]["modifiers"] == 2  # CTRL
    assert "text" not in key_events[0][1]


async def test_scroll_direction(fake_session):
    await tools.scroll([5, 5], "down", 2, session=fake_session)
    wheel = [
        c for c in fake_session._client.calls if c[0] == "Input.dispatchMouseEvent"
    ]
    assert wheel[-1][1]["type"] == "mouseWheel"
    assert wheel[-1][1]["deltaY"] > 0


async def test_drag(fake_session):
    await tools.left_click_drag([0, 0], [50, 60], session=fake_session)
    mouse = [
        c for c in fake_session._client.calls if c[0] == "Input.dispatchMouseEvent"
    ]
    types = [m[1]["type"] for m in mouse]
    assert types == ["mousePressed", "mouseMoved", "mouseReleased"]
    assert fake_session.cursor_position == (50, 60)


async def test_computer_tool_screenshot(fake_session):
    result = await tools.computer("screenshot", session=fake_session)
    assert result[0]["type"] == "image_url"


async def test_computer_tool_cursor_position(fake_session):
    await fake_session.move_mouse(42, 24)
    result = await tools.computer("cursor_position", session=fake_session)
    assert result == "42, 24"


async def test_computer_tool_left_click(fake_session):
    await tools.computer("left_click", coordinate=[7, 8], session=fake_session)
    assert fake_session.cursor_position == (7, 8)


class FlakyCDPClient(FakeCDPClient):
    """Fails the first page command with a detach error, then succeeds."""

    def __init__(self, ws_url):
        super().__init__(ws_url)
        self._tripped = False

    async def send(self, method, params=None, *, session_id=None):
        if method == "Input.dispatchMouseEvent" and not self._tripped:
            self._tripped = True
            self.calls.append((method, params or {}, session_id))
            from verifiers.v1.toolsets.browser.cdp import CDPError

            raise CDPError(
                "Input.dispatchMouseEvent failed: Not attached to an active page"
            )
        if method == "Target.getTargets":
            self.calls.append((method, params or {}, session_id))
            return {"targetInfos": [{"type": "page", "targetId": "target-2"}]}
        return await super().send(method, params, session_id=session_id)


async def test_send_recovers_from_detached_page(monkeypatch):
    monkeypatch.setattr(session_mod, "CDPClient", FlakyCDPClient)
    session = BrowserSession(FakeBackend(), width=800, height=600)
    await session.start()
    # The first mouse event detaches; the session should re-attach and retry,
    # so the click still completes and returns a screenshot.
    result = await tools.left_click([5, 5], session=session)
    methods = [c[0] for c in session._client.calls]
    assert "Target.getTargets" in methods  # recovery re-resolved a live page
    assert result[0]["type"] == "image_url"


async def test_aclose_closes_target_client_and_backend(fake_session):
    await fake_session.start()
    backend = fake_session.backend
    client = fake_session._client
    await fake_session.aclose()
    methods = [c[0] for c in client.calls]
    assert "Target.closeTarget" in methods
    assert ("close", {}, None) in client.calls
    assert backend.closed == ["sess-1"]
    assert fake_session._client is None
