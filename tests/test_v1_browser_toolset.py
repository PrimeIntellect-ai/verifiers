"""Toolset structure + end-to-end integration with the verifiers runtime.

The backend and CDP are faked, so no browser or network is required.
"""

import base64

import pytest

import verifiers as vf
import verifiers.v1.toolsets.browser.session as session_mod
from verifiers.v1.toolsets.browser import CDPBackend, browser_toolset
from verifiers.v1.toolsets.browser.backends import BrowserSessionHandle


# --- structural checks ----------------------------------------------------


def _names(toolset):
    return {getattr(t, "__name__", None) for t in toolset.tools}


def _toolset(mode):
    return browser_toolset(
        mode=mode, backend=CDPBackend(cdp_url="http://localhost:9222")
    )


def test_mode_computer_only():
    assert _names(_toolset("computer")) == {"computer"}


def test_mode_decomposed():
    names = _names(_toolset("decomposed"))
    assert "left_click" in names and "computer" not in names


def test_mode_both():
    names = _names(_toolset("both"))
    assert "computer" in names and "left_click" in names


def test_backend_is_required():
    with pytest.raises(TypeError):
        browser_toolset(mode="both")  # type: ignore[call-arg]


def test_session_is_bound_hidden_for_every_tool():
    ts = _toolset("both")
    for tool in ts.tools:
        assert ts.bindings[f"{tool.__name__}.session"] == "objects.browser"


def test_rollout_scoped_and_writable():
    ts = _toolset("both")
    assert ts.write is True
    assert ts.scope == "rollout"


# --- runtime integration --------------------------------------------------


class FakeBackend:
    async def create(self):
        return BrowserSessionHandle(session_id="s1", cdp_ws_url="ws://fake/browser")

    async def close(self, session_id):
        pass


class FakeCDPClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url

    async def connect(self):
        pass

    async def send(self, method, params=None, *, session_id=None):
        if method == "Target.createTarget":
            return {"targetId": "t1"}
        if method == "Target.attachToTarget":
            return {"sessionId": "cdp1"}
        if method == "Page.captureScreenshot":
            return {"data": base64.b64encode(b"png").decode()}
        return {}

    async def close(self):
        pass


@pytest.fixture
def patched_cdp(monkeypatch):
    monkeypatch.setattr(session_mod, "CDPClient", FakeCDPClient)


def _build():
    harness = vf.Harness(config=vf.HarnessConfig())
    harness.add_toolset(browser_toolset(mode="both", backend=FakeBackend()))
    task = vf.Task({"prompt": [{"role": "user", "content": "browse"}]}).freeze()
    state = vf.State.for_task(task)
    return harness, task, state


async def test_schema_hides_session_and_keeps_action(patched_cdp):
    harness, task, state = _build()
    harness.runtime.prepare_state(task, state)
    defs = harness.runtime.tool_defs(state)
    by_name = {d.name: d for d in defs}

    assert "computer" in by_name and "left_click" in by_name
    for tool_def in defs:
        props = tool_def.parameters.get("properties", {})
        assert "session" not in props, f"{tool_def.name} leaks session"

    computer_props = by_name["computer"].parameters["properties"]
    assert "action" in computer_props and "coordinate" in computer_props


async def test_runtime_dispatches_tool_with_injected_session(patched_cdp):
    harness, task, state = _build()
    harness.runtime.prepare_state(task, state)
    result = await harness.runtime.call_tool("screenshot", task, state)
    assert isinstance(result, list)
    assert result[0]["type"] == "image_url"


async def test_runtime_dispatches_computer_left_click(patched_cdp):
    harness, task, state = _build()
    harness.runtime.prepare_state(task, state)
    result = await harness.runtime.call_tool(
        "computer", task, state, action="left_click", coordinate=[12, 34]
    )
    assert result[0]["type"] == "image_url"
