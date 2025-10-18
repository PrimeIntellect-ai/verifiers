# tests/test_message_utils_image.py
import copy

from verifiers.utils.message_utils import (
    cleanup_message
)

def test_cleanup_message_image_url_without_text_field():
    msg = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ],
    }
    cleaned = cleanup_message(copy.deepcopy(msg))
    assert cleaned["role"] == "user"
    assert len(cleaned["content"]) == 1
    assert cleaned["content"][0]["type"] == "image_url"
    assert "text" not in cleaned["content"][0]
