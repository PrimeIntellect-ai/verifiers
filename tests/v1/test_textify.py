import asyncio
import base64
import io
from types import MethodType, SimpleNamespace
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

import numpy as np
from PIL import Image

import verifiers.v1 as vf
import verifiers.v1.utils.textify as textify
from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect
from verifiers.v1.errors import TaskError
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession
from verifiers.v1.utils.textify import _grid_shape, render_url


def _data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_ascii_render_and_auto_invert() -> None:
    img = np.tile(np.linspace(0, 255, 8, dtype=np.uint8), (4, 1))
    cfg = vf.TextifyConfig(enabled=True, width=8, char_aspect=0.5, invert=False)
    assert vf.image_to_text(img, cfg) == " .-=+*%@\n .-=+*%@"
    auto = cfg.model_copy(update={"invert": None})
    # Mean lightness is exactly 0.5-ish here; explicit inversion is separately deterministic.
    inverted = cfg.model_copy(update={"invert": True})
    assert vf.image_to_text(img, inverted) == "@%*+=-. \n@%*+=-. "
    assert vf.image_to_text(img, auto)


def test_explicit_resolution_and_max_chars() -> None:
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    cfg = vf.TextifyConfig(enabled=True, width=12, height=7, max_chars=20)
    lines = vf.image_to_text(img, cfg).splitlines()
    assert len(lines) * len(lines[0]) <= 20
    assert all(len(line) == len(lines[0]) for line in lines)


def test_braille_bit_packing() -> None:
    # One bright source pixel in the top-left dot -> U+2801.
    img = np.zeros((4, 2, 3), dtype=np.uint8)
    img[0, 0] = 255
    cfg = vf.TextifyConfig(
        enabled=True,
        mode="braille",
        width=1,
        char_aspect=0.5,
        threshold=0.5,
        invert=False,
    )
    assert vf.image_to_text(img, cfg) == "⠁"


def test_textify_messages_identity_and_replace() -> None:
    url = _data_url(np.full((2, 2, 3), 255, dtype=np.uint8))
    messages: vf.Messages = [
        vf.UserMessage(
            content=[
                vf.TextContentPart(text="before"),
                vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=url)),
                vf.TextContentPart(text="after"),
            ]
        )
    ]
    assert vf.textify_messages(messages, vf.TextifyConfig()) is messages
    out = vf.textify_messages(
        messages,
        vf.TextifyConfig(enabled=True, width=2, char_aspect=0.5, invert=False),
    )
    assert out is not messages
    assert isinstance(out[0].content, list)
    assert [part.type for part in out[0].content] == ["text", "text", "text"]
    assert out[0].content[1].text.startswith("```image[ascii]\n")
    assert isinstance(messages[0].content[1], vf.ImageUrlContentPart)


def test_all_dialects_replace_images_without_mutating_input() -> None:
    url = _data_url(np.zeros((2, 2, 3), dtype=np.uint8))
    cfg = vf.TextifyConfig(enabled=True, width=2)

    def render(value: str) -> str | None:
        return render_url(value, cfg)

    chat = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}],
            }
        ]
    }
    chat_out = ChatDialect().textify_body(chat, render)
    assert chat_out["messages"][0]["content"][0]["type"] == "text"
    assert chat["messages"][0]["content"][0]["type"] == "image_url"

    responses = {
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": url}],
            },
            {
                "type": "function_call_output",
                "output": [{"type": "input_image", "image_url": url}],
            },
        ]
    }
    responses_out = ResponsesDialect().textify_body(responses, render)
    assert responses_out["input"][0]["content"][0]["type"] == "input_text"
    assert responses_out["input"][1]["output"][0]["type"] == "input_text"

    data = url.split(",", 1)[1]
    anthropic = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "x",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": data,
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }
    anthropic_out = AnthropicDialect().textify_body(anthropic, render)
    nested = anthropic_out["messages"][0]["content"][0]["content"][0]
    assert nested["type"] == "text"


def test_http_url_passes_through() -> None:
    cfg = vf.TextifyConfig(enabled=True)

    def render(value: str) -> str | None:
        return render_url(value, cfg)

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    }
                ],
            }
        ]
    }
    assert ChatDialect().textify_body(body, render) == body


def test_safe_default_budget_and_describe() -> None:
    img = np.zeros((1000, 1, 3), dtype=np.uint8)
    cfg = vf.TextifyConfig(enabled=True)
    lines = vf.image_to_text(img, cfg).splitlines()
    assert len(lines) * len(lines[0]) <= 40_000
    assert "invert=auto" in vf.describe_textify(cfg)
    assert "max_chars=40000" in vf.describe_textify(cfg)


def test_malformed_data_url_raises() -> None:
    cfg = vf.TextifyConfig(enabled=True)
    try:
        render_url("data:image/png;base64,not-base64!", cfg)
    except ValueError as error:
        assert "base64" in str(error)
    else:
        raise AssertionError("malformed image data must not pass through")


def test_signed_array_rejected() -> None:
    cfg = vf.TextifyConfig(enabled=True, width=2)
    try:
        vf.image_to_text(np.zeros((2, 2), dtype=np.int16), cfg)
    except ValueError as error:
        assert "unsigned" in str(error)
    else:
        raise AssertionError("signed image arrays must not wrap into uint8")


@pytest.mark.asyncio
async def test_interception_helper_attributes_textify_failures() -> None:
    cfg = vf.TextifyConfig(enabled=True)
    session = SimpleNamespace(
        textify=cfg,
        render_image=lambda url: render_url(url, cfg),
    )
    bad = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,not-valid!"},
                    }
                ],
            }
        ]
    }
    server = InterceptionServer()
    with pytest.raises(TaskError, match="textify failed: ValueError"):
        await server._textify_body(session, ChatDialect(), bad)
    messages: vf.Messages = [
        vf.UserMessage(
            content=[
                vf.ImageUrlContentPart(
                    image_url=vf.ImageUrlSource(url="data:image/png;base64,not-valid!")
                )
            ]
        )
    ]
    with pytest.raises(TaskError, match="textify failed: ValueError"):
        await server._textify_messages(session, messages)


def test_responses_file_id_passes_and_computer_screenshot_textifies() -> None:
    url = _data_url(np.zeros((2, 2, 3), dtype=np.uint8))
    cfg = vf.TextifyConfig(enabled=True, width=2)

    def render(value: str) -> str | None:
        return render_url(value, cfg)

    body = {
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_image", "file_id": "file_123"}],
            },
            {"type": "computer_call", "call_id": "call_123", "action": {}},
            {
                "type": "computer_call_output",
                "call_id": "call_123",
                "output": {"type": "computer_screenshot", "image_url": url},
            },
        ]
    }
    out = ResponsesDialect().textify_body(body, render)
    assert out["input"][0] == body["input"][0]
    assert len(out["input"]) == 2
    assert out["input"][1]["role"] == "user"
    assert out["input"][1]["content"][0]["type"] == "input_text"


def test_nonfinite_config_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        vf.TextifyConfig(char_aspect=float("inf"))
    with pytest.raises(ValueError):
        vf.TextifyConfig(gamma=float("nan"))


def test_legacy_textify_rejected() -> None:
    with pytest.raises(ValueError, match="native v1 tasksets only"):
        vf.EnvConfig(id="legacy", textify=vf.TextifyConfig(enabled=True))


def test_output_budget_counts_newlines() -> None:
    cfg = vf.TextifyConfig(enabled=True, width=10, height=10, max_chars=20)
    art = vf.image_to_text(np.zeros((10, 10), dtype=np.uint8), cfg)
    assert len(art) <= 20


def test_oversized_payload_rejected_before_decode(monkeypatch) -> None:
    monkeypatch.setattr(textify, "_MAX_IMAGE_BYTES", 3)
    cfg = vf.TextifyConfig(enabled=True)
    with pytest.raises(ValueError, match="byte safety limit"):
        render_url("data:image/png;base64," + "A" * 5, cfg)


def test_otsu_ascii_is_binary_and_braille_uses_adaptive_cutoff() -> None:
    image = np.array(
        [
            [0, 10, 20, 30],
            [0, 10, 220, 255],
        ],
        dtype=np.uint8,
    )
    ascii_cfg = vf.TextifyConfig(
        enabled=True,
        width=4,
        height=2,
        invert=False,
        threshold="otsu",
    )
    ascii_art = vf.image_to_text(image, ascii_cfg)
    assert set(ascii_art) <= {ascii_cfg.ramp[0], ascii_cfg.ramp[-1], "\n"}
    assert ascii_cfg.ramp[0] in ascii_art and ascii_cfg.ramp[-1] in ascii_art

    braille_cfg = vf.TextifyConfig(
        enabled=True,
        mode="braille",
        width=1,
        height=1,
        invert=False,
        threshold="otsu",
    )
    assert vf.image_to_text(np.tile(image, (2, 1)), braille_cfg) != "⠀"


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        vf.TextifyConfig(threshold=1.1)
    assert vf.TextifyConfig(threshold="otsu").threshold == "otsu"


def test_anthropic_file_images_pass_through() -> None:
    file_image = {
        "type": "image",
        "source": {"type": "file", "file_id": "file_123"},
    }
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    file_image,
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": [file_image],
                    },
                ],
            }
        ]
    }

    def should_not_render(_: str) -> str:
        raise AssertionError("file-backed images must pass through")

    out = AnthropicDialect().textify_body(body, should_not_render)
    assert out == body
    assert out is not body


@pytest.mark.asyncio
async def test_opening_textify_failure_does_not_advance_user(monkeypatch) -> None:
    calls = 0
    raw: vf.Messages = [vf.UserMessage(content="opening")]

    async def user(_: str, _turn: int) -> vf.Messages:
        nonlocal calls
        calls += 1
        return raw

    async def fail(*_) -> vf.Messages:
        raise TaskError("textify failed")

    session = SimpleNamespace(
        opening=None,
        user=user,
        _opening_textified=False,
        _opening_lock=asyncio.Lock(),
        error=None,
    )
    server = InterceptionServer()
    monkeypatch.setattr(server, "_textify_messages", fail)

    with pytest.raises(TaskError, match="textify failed"):
        await server._opening_messages(session, 0)
    with pytest.raises(TaskError, match="textify failed"):
        await server._opening_messages(session, 0)

    assert calls == 1
    assert session.opening is raw


@pytest.mark.asyncio
async def test_concurrent_opening_calls_user_once() -> None:
    calls = 0
    release = asyncio.Event()

    async def user(_: str, _turn: int) -> vf.Messages:
        nonlocal calls
        calls += 1
        await release.wait()
        return [vf.UserMessage(content="opening")]

    session = SimpleNamespace(
        opening=None,
        user=user,
        _opening_textified=False,
        _opening_lock=asyncio.Lock(),
        textify=vf.TextifyConfig(),
    )
    server = InterceptionServer()
    first = asyncio.create_task(server._opening_messages(session, 0))
    second = asyncio.create_task(server._opening_messages(session, 0))
    await asyncio.sleep(0)
    release.set()
    one, two = await asyncio.gather(first, second)

    assert calls == 1
    assert one == two == session.opening


def test_describe_includes_output_affecting_threshold() -> None:
    ascii_fixed = vf.describe_textify(vf.TextifyConfig(mode="ascii", threshold=0.5))
    ascii_otsu = vf.describe_textify(vf.TextifyConfig(mode="ascii", threshold="otsu"))
    braille = vf.describe_textify(vf.TextifyConfig(mode="braille", threshold=0.25))

    assert "ramp=" in ascii_fixed and "threshold=" not in ascii_fixed
    assert "ramp=" in ascii_otsu and "threshold=otsu" in ascii_otsu
    assert "ramp=" not in braille and "threshold=0.25" in braille


def test_textify_cache_resists_full_history_scan(monkeypatch) -> None:
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
    assert sum(calls.values()) == 48  # 40 first renders + 8 overflow misses
    assert sorted(calls.values()).count(1) == 32
    assert sorted(calls.values()).count(2) == 8


def test_textify_cache_concurrent_access_is_bounded(monkeypatch) -> None:
    def render(url: str, _cfg) -> str:
        return f"rendered:{url}"

    monkeypatch.setattr("verifiers.v1.session.render_url", render)
    session = SimpleNamespace(
        textify=vf.TextifyConfig(enabled=True),
        _textify_cache=OrderedDict(),
        _textify_seen=bytearray(8192),
        _textify_lock=threading.Lock(),
    )
    urls = [f"data:image/png;base64,{index % 40}" for index in range(400)]

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(lambda url: RolloutSession.render_image(session, url), urls)
        )

    assert results == [f"rendered:{url}" for url in urls]
    assert len(session._textify_cache) <= 32
    assert len(session._textify_seen) == 8192


@pytest.mark.asyncio
async def test_typed_and_wire_paths_share_render_cache(monkeypatch) -> None:
    calls = 0

    def render(url: str, _cfg) -> str:
        nonlocal calls
        calls += 1
        return f"rendered:{url}"

    monkeypatch.setattr("verifiers.v1.session.render_url", render)
    session = SimpleNamespace(
        textify=vf.TextifyConfig(enabled=True),
        _textify_cache=OrderedDict(),
        _textify_seen=bytearray(8192),
        _textify_lock=threading.Lock(),
    )
    session.render_image = MethodType(RolloutSession.render_image, session)
    url = "data:image/png;base64,same-image"
    assert session.render_image(url) == f"rendered:{url}"

    messages: vf.Messages = [
        vf.UserMessage(
            content=[vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=url))]
        )
    ]
    out = await InterceptionServer()._textify_messages(session, messages)

    assert calls == 1
    assert isinstance(out[0].content, list)
    assert out[0].content[0] == vf.TextContentPart(text=f"rendered:{url}")


@pytest.mark.asyncio
async def test_concurrent_opening_textify_is_serialized(monkeypatch) -> None:
    user_calls = 0
    textify_calls = 0
    active = 0
    max_active = 0
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def user(_: str, _turn: int) -> vf.Messages:
        nonlocal user_calls
        user_calls += 1
        return [vf.UserMessage(content="raw")]

    async def textify_opening(_session, _messages) -> vf.Messages:
        nonlocal textify_calls, active, max_active
        textify_calls += 1
        active += 1
        max_active = max(max_active, active)
        try:
            if textify_calls == 1:
                first_started.set()
                await release_first.wait()
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
        trace=SimpleNamespace(id="trace"),
    )
    server = InterceptionServer()
    monkeypatch.setattr(server, "_textify_messages", textify_opening)
    first = asyncio.create_task(server._opening_messages(session, 0))
    await first_started.wait()
    second = asyncio.create_task(server._opening_messages(session, 0))
    await asyncio.sleep(0)
    release_first.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert isinstance(results[0], TaskError)
    assert results[1] == [vf.UserMessage(content="rendered")]
    assert user_calls == 1
    assert textify_calls == 2
    assert max_active == 1
    assert session._opening_textified
    assert session.opening == results[1]
    assert session.error is None
    server._error_response(session, ChatDialect(), results[0])
    assert session.error is None


def test_oversized_payload_rejected_after_decode(monkeypatch) -> None:
    monkeypatch.setattr(textify, "_MAX_IMAGE_BYTES", 3)
    monkeypatch.setattr(textify.base64, "b64decode", lambda *_, **__: b"four")
    with pytest.raises(ValueError, match="byte safety limit"):
        textify.data_url_bytes("data:image/png;base64,AAAA")


def test_anthropic_system_images_textify_or_pass_through() -> None:
    url = _data_url(np.zeros((2, 2, 3), dtype=np.uint8))
    encoded = url.split(",", 1)[1]
    cfg = vf.TextifyConfig(enabled=True, width=2)

    def render(value: str) -> str | None:
        return render_url(value, cfg)

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
    body = {
        "system": [
            {"type": "text", "text": "system text"},
            base64_image,
            file_image,
        ],
        "messages": [{"role": "user", "content": "hello"}],
    }

    out = AnthropicDialect().textify_body(body, render)
    assert out["system"][0] == body["system"][0]
    assert out["system"][1]["type"] == "text"
    assert out["system"][1]["text"].startswith("```image[ascii]\n")
    assert out["system"][2] == file_image
    assert body["system"][1] == base64_image

    without_system = {"messages": []}
    assert "system" not in AnthropicDialect().textify_body(without_system, render)


def test_hard_output_ceiling_counts_newlines() -> None:
    allowed = vf.TextifyConfig(width=999, height=1000, max_chars=None)
    assert _grid_shape(1, 1, allowed) == (1000, 999)

    oversized = vf.TextifyConfig(width=1000, height=1000, max_chars=None)
    with pytest.raises(ValueError, match="one-million-character"):
        _grid_shape(1, 1, oversized)


@pytest.mark.asyncio
async def test_postcommit_failure_caches_response_before_error() -> None:
    session = SimpleNamespace(
        last_request=None,
        last_response=None,
        error=None,
        trace=SimpleNamespace(id="trace"),
    )
    response = SimpleNamespace(raw={"id": "served"})
    error = TaskError("user image textify failed")
    server = InterceptionServer()

    future: asyncio.Future[dict | None] = asyncio.get_running_loop().create_future()
    result = server._fail_after_commit(
        session, ChatDialect(), b"request-digest", response, error, future
    )

    assert result.status == 502
    assert await future == {"id": "served"}
    assert session.last_request == b"request-digest"
    assert session.last_response == {"id": "served"}
    assert session.error is error
    replay = server._replay_response(session, session.last_response)
    assert replay.status == 200
    assert session.error is None
