# tests/test_message_utils_audio.py
from verifiers.types import (
    ImageUrlContentPart,
    ImageUrlSource,
    InputAudioContentPart,
    InputAudioSource,
    UserMessage,
)
from verifiers.utils.message_utils import (
    message_to_printable,
    messages_to_printable,
    serialize_message_for_output,
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


def test_serialize_message_for_output_preserves_data_urls_as_image_url():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is shown?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:IMAGE/PNG;charset=utf-8;base64,ab c123="},
            },
        ],
    }

    serialized = serialize_message_for_output(message)

    assert serialized == {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is shown?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:IMAGE/PNG;charset=utf-8;base64,ab c123="},
            },
        ],
    }


def test_serialize_message_for_output_preserves_non_data_image_url():
    message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            }
        ],
    }

    serialized = serialize_message_for_output(message)

    assert serialized == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            }
        ],
    }


def test_serialize_message_for_output_preserves_input_audio_payload():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "listen to this"},
            {
                "type": "input_audio",
                "input_audio": {"data": " ZHVt\nbXk= ", "format": "WAV"},
            },
        ],
    }

    serialized = serialize_message_for_output(message)

    assert serialized == {
        "role": "user",
        "content": [
            {"type": "text", "text": "listen to this"},
            {
                "type": "input_audio",
                "input_audio": {"data": "ZHVtbXk=", "format": "wav"},
            },
        ],
    }


def test_serialize_message_for_output_preserves_audio_alias_payload():
    message = {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "data": " ZHVt\nbXk= ",
                "format": "WAV",
            }
        ],
    }

    serialized = serialize_message_for_output(message)

    assert serialized == {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "ZHVtbXk=", "format": "wav"},
            }
        ],
    }


def test_serialize_message_for_output_falls_back_for_invalid_audio_payload():
    message = {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "", "format": "wav"},
            }
        ],
    }

    serialized = serialize_message_for_output(message)

    assert serialized == {
        "role": "user",
        "content": [{"type": "text", "text": "[audio]"}],
    }


def test_serialize_message_for_output_supports_typed_messages():
    message = UserMessage(
        content=[
            ImageUrlContentPart(
                image_url=ImageUrlSource(url="data:image/png;base64,abc123")
            ),
            InputAudioContentPart(
                input_audio=InputAudioSource(data=" ZHVt\nbXk= ", format="WAV")
            ),
        ]
    )

    serialized = serialize_message_for_output(message)

    assert serialized == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc123"},
            },
            {
                "type": "input_audio",
                "input_audio": {"data": "ZHVtbXk=", "format": "wav"},
            },
        ],
    }


def test_dataset_map_may_introduce_none_fields_and_stripping_fixes():
    """
    HuggingFace Dataset.map() may materialize missing fields as None when
    content items have different schemas. Regardless of whether that happens
    in the installed datasets version, stripping None values should preserve
    the expected multimodal content shape.
    """
    from datasets import Dataset

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

    assert content[0]["type"] == "text"
    assert content[0]["text"] == "What is this?"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"] == {"url": "data:image/png;base64,abc123"}

    # Older/newer datasets versions differ here: some inject missing keys as None,
    # others leave them absent. Normalize to the problematic shape before testing
    # the stripping logic that production code relies on.
    assert content[0].get("image_url") is None, "text item should have image_url=None"
    content[0]["image_url"] = None

    assert content[1].get("text") is None, "image_url item should have text=None"
    content[1]["text"] = None

    # Strip None values (same logic as in get_model_response)
    for msg in prompt:
        msg_content = msg.get("content")
        if isinstance(msg_content, list):
            msg["content"] = [
                {k: v for k, v in c.items() if v is not None}
                if isinstance(c, dict)
                else c
                for c in msg_content
            ]

    cleaned_content = prompt[0]["content"]

    assert "image_url" not in cleaned_content[0], (
        "stripping should remove image_url from text item"
    )
    assert "text" not in cleaned_content[1], (
        "stripping should remove text from image_url item"
    )
    assert cleaned_content[0] == {"type": "text", "text": "What is this?"}
    assert cleaned_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123"},
    }
