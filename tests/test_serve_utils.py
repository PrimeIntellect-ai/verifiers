"""Tests for verifiers.utils.serve_utils msgpack encoder.

Focused on the torch dispatch path: tensors must round-trip, and
non-tensor torch objects (``torch.dtype``, ``torch.device``,
``torch.Size``) must raise a clean ``TypeError`` rather than
``AttributeError`` from a bogus ``.detach()`` call.
"""

import numpy as np
import pytest

from verifiers.utils.serve_utils import (
    decode_tensor_payload,
    msgpack_encoder,
    walk_decode_tensors,
)

torch = pytest.importorskip("torch")


class TestMsgpackEncoder:
    def test_tensor_round_trips(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        encoded = msgpack_encoder(t)
        assert encoded["__torch_tensor__"] is True
        assert encoded["dtype"] == "float32"
        assert tuple(encoded["shape"]) == (2, 2)

        rehydrated = decode_tensor_payload(encoded, to_torch=True)
        assert torch.equal(rehydrated, t)

    def test_non_tensor_torch_objects_raise_typeerror_not_attributeerror(self):
        # The previous string-module check matched any object with
        # __module__.startswith("torch") and crashed on .detach().
        # These three are the common offenders.
        for non_tensor in (torch.float32, torch.device("cpu"), torch.Size([1, 2])):
            with pytest.raises(TypeError, match="not msgpack serializable"):
                msgpack_encoder(non_tensor)

    def test_numpy_array_encodes(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        encoded = msgpack_encoder(arr)
        assert encoded["__torch_tensor__"] is True  # uses the same encoded shape
        assert encoded["dtype"] == "int64"
        rehydrated = decode_tensor_payload(encoded, to_torch=False)
        assert np.array_equal(rehydrated, arr)

    def test_walk_decodes_nested_tensors(self):
        payload = {
            "mm_items": {
                "image": [
                    {"pixel_values": msgpack_encoder(torch.ones(2, 3))},
                    {"pixel_values": msgpack_encoder(torch.zeros(1, 5))},
                ]
            }
        }
        decoded = walk_decode_tensors(payload, to_torch=True)
        first = decoded["mm_items"]["image"][0]["pixel_values"]
        second = decoded["mm_items"]["image"][1]["pixel_values"]
        assert torch.equal(first, torch.ones(2, 3))
        assert torch.equal(second, torch.zeros(1, 5))
