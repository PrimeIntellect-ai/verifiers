import base64
import io
from types import SimpleNamespace

import pytest

import numpy as np
from PIL import Image

import verifiers.v1 as vf
import verifiers.v1.utils.textify as textify
from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect
from verifiers.v1.errors import TaskError
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.utils.textify import render_url


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
