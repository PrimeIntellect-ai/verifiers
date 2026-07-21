import asyncio
import base64
import io
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import verifiers.v1 as vf
import verifiers.v1.utils.textify as textify
from verifiers.v1.dialects import AnthropicDialect, ResponsesDialect
from verifiers.v1.errors import TaskError
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession
from verifiers.v1.utils.textify import _grid_shape, render_url


def _data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_renderer_modes() -> None:
    gradient = np.tile(np.linspace(0, 255, 8, dtype=np.uint8), (4, 1))
    ascii_cfg = vf.TextifyConfig(width=8, char_aspect=0.5, invert=False)
    assert vf.image_to_text(gradient, ascii_cfg) == " .-=+*%@\n .-=+*%@"
    assert (
        vf.image_to_text(gradient, ascii_cfg.model_copy(update={"invert": True}))
        == "@%*+=-. \n@%*+=-. "
    )

    dot = np.zeros((4, 2, 3), dtype=np.uint8)
    dot[0, 0] = 255
    assert (
        vf.image_to_text(
            dot,
            vf.TextifyConfig(mode="braille", width=1, threshold=0.5, invert=False),
        )
        == "⠁"
    )

    otsu = vf.TextifyConfig(width=4, height=2, invert=False, threshold="otsu")
    art = vf.image_to_text(
        np.array([[0, 10, 20, 30], [0, 10, 220, 255]], dtype=np.uint8),
        otsu,
    )
    assert set(art) <= {otsu.ramp[0], otsu.ramp[-1], "\n"}
    assert otsu.ramp[0] in art and otsu.ramp[-1] in art


def test_output_and_input_safety_limits(monkeypatch) -> None:
    cfg = vf.TextifyConfig(width=10, height=10, max_chars=20)
    assert len(vf.image_to_text(np.zeros((10, 10), dtype=np.uint8), cfg)) <= 20

    assert _grid_shape(
        1, 1, vf.TextifyConfig(width=999, height=1000, max_chars=None)
    ) == (1000, 999)
    with pytest.raises(ValueError, match="one-million-character"):
        _grid_shape(1, 1, vf.TextifyConfig(width=1000, height=1000, max_chars=None))

    monkeypatch.setattr(textify, "_MAX_IMAGE_BYTES", 3)
    with pytest.raises(ValueError, match="byte safety limit"):
        render_url("data:image/png;base64," + "A" * 5, vf.TextifyConfig())
    monkeypatch.setattr(textify.base64, "b64decode", lambda *_, **__: b"four")
    with pytest.raises(ValueError, match="byte safety limit"):
        textify.data_url_bytes("data:image/png;base64,AAAA")


def test_non_chat_dialect_image_shapes() -> None:
    url = _data_url(np.zeros((2, 2, 3), dtype=np.uint8))
    encoded = url.split(",", 1)[1]
    cfg = vf.TextifyConfig(enabled=True, width=2)

    def render(value: str) -> str | None:
        return render_url(value, cfg)

    responses = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": url},
                    {"type": "input_image", "file_id": "file_123"},
                ],
            },
            {"type": "computer_call", "call_id": "call_123", "action": {}},
            {
                "type": "computer_call_output",
                "call_id": "call_123",
                "output": {"type": "computer_screenshot", "image_url": url},
            },
        ]
    }
    responses_out = ResponsesDialect().textify_body(responses, render)
    assert responses_out["input"][0]["content"][0]["type"] == "input_text"
    assert responses_out["input"][0]["content"][1]["file_id"] == "file_123"
    assert responses_out["input"][1]["role"] == "user"
    assert responses_out["input"][1]["content"][0]["type"] == "input_text"

    base64_image = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": encoded,
        },
    }
    file_image = {
        "type": "image",
        "source": {"type": "file", "file_id": "file_123"},
    }
    anthropic = {
        "system": [base64_image, file_image],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": [base64_image, file_image],
                    }
                ],
            }
        ],
    }
    anthropic_out = AnthropicDialect().textify_body(anthropic, render)
    assert anthropic_out["system"][0]["type"] == "text"
    assert anthropic_out["system"][1] == file_image
    nested = anthropic_out["messages"][0]["content"][0]["content"]
    assert nested[0]["type"] == "text"
    assert nested[1] == file_image
    assert anthropic["system"][0] == base64_image


def test_legacy_textify_rejected() -> None:
    with pytest.raises(ValueError, match="native v1 tasksets only"):
        vf.EnvConfig(id="legacy", textify=vf.TextifyConfig(enabled=True))


def test_textify_cache_is_bounded_and_scan_resistant(monkeypatch) -> None:
    calls: dict[str, int] = {}

    def render(url: str, _cfg) -> str:
        calls[url] = calls.get(url, 0) + 1
        return f"rendered:{url}"

    monkeypatch.setattr("verifiers.v1.session.render_url", render)
    session = SimpleNamespace(
        textify=vf.TextifyConfig(enabled=True),
        _textify_cache=OrderedDict(),
        _textify_seen=bytearray(8192),
        _textify_lock=threading.Lock(),
    )
    urls = [f"data:image/png;base64,{index}" for index in range(40)]
    first = [RolloutSession.render_image(session, url) for url in urls]
    retained = set(session._textify_cache)
    second = [RolloutSession.render_image(session, url) for url in urls]

    assert first == second
    assert len(session._textify_cache) == 32
    assert set(session._textify_cache) == retained
    assert sorted(calls.values()).count(1) == 32
    assert sorted(calls.values()).count(2) == 8

    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(
            pool.map(lambda url: RolloutSession.render_image(session, url), urls * 2)
        )
    assert concurrent == [f"rendered:{url}" for url in urls * 2]
    assert len(session._textify_cache) <= 32


@pytest.mark.asyncio
async def test_concurrent_opening_textify_is_serialized(monkeypatch) -> None:
    calls = 0
    active = 0

    async def user(_: str, _turn: int) -> vf.Messages:
        return [vf.UserMessage(content="raw")]

    async def textify_opening(_session, _messages, *_) -> vf.Messages:
        nonlocal calls, active
        calls += 1
        active += 1
        try:
            assert active == 1
            if calls == 1:
                await asyncio.sleep(0)
                raise TaskError("first textify failed")
            return [vf.UserMessage(content="rendered")]
        finally:
            active -= 1

    session = SimpleNamespace(
        opening=None,
        user=user,
        _opening_textified=False,
        _opening_lock=asyncio.Lock(),
        error=None,
        textify=vf.TextifyConfig(enabled=True),
        render_image=lambda _: None,
    )
    server = InterceptionServer()
    monkeypatch.setattr(server, "_textify", textify_opening)
    results = await asyncio.gather(
        server._opening_messages(session, 0),
        server._opening_messages(session, 0),
        return_exceptions=True,
    )

    assert isinstance(results[0], TaskError)
    assert results[1] == [vf.UserMessage(content="rendered")]
    assert calls == 2
    assert session.opening == results[1]
    assert session.error is None
