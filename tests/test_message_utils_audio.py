# tests/test_message_utils_audio.py
import copy

from verifiers.utils.message_utils import (
    cleanup_message,
    message_to_printable,
    messages_to_printable,
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


def test_cleanup_message_strips_none_fields_from_multimodal():
    """
    HuggingFace Dataset.map() unifies schemas across content items,
    adding None values for missing keys. This test ensures cleanup_message
    strips those None fields to produce valid OpenAI API payloads.
    """
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this image", "image_url": None},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc123"},
                "text": None,
            },
        ],
    }
    cleaned = cleanup_message(copy.deepcopy(msg))
    assert cleaned["role"] == "user"
    assert len(cleaned["content"]) == 2
    assert cleaned["content"][0] == {"type": "text", "text": "describe this image"}
    assert cleaned["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123"},
    }


def test_dataset_map_introduces_none_fields_and_cleanup_fixes():
    """
    Demonstrates that HuggingFace Dataset.map() introduces None values
    when content items have different schemas, and cleanup_messages fixes this.
    """
    from datasets import Dataset

    from verifiers.utils.message_utils import cleanup_messages

    def format_prompt(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example["question"]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{example['image']}"
                            },
                        },
                    ],
                }
            ]
        }

    ds = Dataset.from_dict({"question": ["What is this?"], "image": ["abc123"]})
    ds = ds.map(format_prompt)

    prompt = ds[0]["prompt"]
    content = prompt[0]["content"]

    # Dataset.map() unifies schemas, adding None for missing keys
    assert "image_url" in content[0], (
        "Dataset.map should add image_url key to text item"
    )
    assert content[0]["image_url"] is None, "text item should have image_url=None"
    assert "text" in content[1], "Dataset.map should add text key to image_url item"
    assert content[1]["text"] is None, "image_url item should have text=None"

    # cleanup_messages should fix this
    cleaned_prompt = cleanup_messages(prompt)
    cleaned_content = cleaned_prompt[0]["content"]

    assert "image_url" not in cleaned_content[0], (
        "cleanup should remove image_url from text item"
    )
    assert "text" not in cleaned_content[1], (
        "cleanup should remove text from image_url item"
    )
    assert cleaned_content[0] == {"type": "text", "text": "What is this?"}
    assert cleaned_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123"},
    }
