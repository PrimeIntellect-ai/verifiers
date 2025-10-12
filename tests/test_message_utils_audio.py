# tests/test_message_utils_audio.py
import copy

from verifiers.utils.message_utils import (
    message_to_printable,
    messages_to_printable,
    cleanup_message,
)

DUMMY_B64 = "ZHVtbXk="


def test_message_to_printable_renders_audio_placeholder():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {
                "type": "input_audio",
                "input_audio": {"data": DUMMY_B64, "format": "wav"},
            },
        ],
    }
    out = message_to_printable(msg)
    assert out["role"] == "user"
    assert "[audio]" in out["content"]

    assert "hello" in out["content"]


def test_messages_to_printable_order_and_joining():
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": DUMMY_B64, "format": "wav"},
                },
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    out = messages_to_printable(msgs)
    assert isinstance(out, list) and len(out) == 1

    printable = out[0]["content"]
    assert "[audio]" in printable and "describe" in printable


def test_cleanup_message_strips_extraneous_fields_from_audio():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "t"},
            {
                "type": "input_audio",
                "input_audio": {"data": DUMMY_B64, "format": "wav"},
                "text": "ignore",
                "image_url": {"url": "ignore"},
                "random": "ignore",
            },
        ],
    }
    cleaned = cleanup_message(copy.deepcopy(msg))
    assert cleaned["role"] == "user"
    assert len(cleaned["content"]) == 2
    assert cleaned["content"][1] == {
        "type": "input_audio",
        "input_audio": {"data": DUMMY_B64, "format": "wav"},
    }


def test_cleanup_message_is_idempotent():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "t"},
            {
                "type": "input_audio",
                "input_audio": {"data": DUMMY_B64, "format": "wav"},
            },
        ],
    }
    once = cleanup_message(copy.deepcopy(msg))
    twice = cleanup_message(copy.deepcopy(once))
    assert twice == once
