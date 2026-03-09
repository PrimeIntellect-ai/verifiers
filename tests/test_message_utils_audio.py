# tests/test_message_utils_audio.py
from verifiers.utils.message_utils import (
    message_to_printable,
    messages_to_printable,
    normalize_messages,
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


def test_normalize_messages_strips_none_fields_after_dataset_map():
    """
    Dataset-backed prompts may pick up schema-padding None fields when content
    items have different shapes. Message normalization should strip them.
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

    # Some `datasets` versions materialize the padding fields, others do not.
    content[0].setdefault("image_url", None)
    content[1].setdefault("text", None)

    normalized = normalize_messages(prompt, field_name="prompt")
    normalized_content = normalized[0]["content"]
    assert isinstance(normalized_content, list)
    cleaned_content = [
        part.model_dump() if hasattr(part, "model_dump") else part
        for part in normalized_content
    ]

    assert cleaned_content[0] == {"type": "text", "text": "What is this?"}
    assert cleaned_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123"},
    }
