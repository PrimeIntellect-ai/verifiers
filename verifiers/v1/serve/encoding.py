"""msgpack encoding for the env-serve wire."""

import dataclasses
import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from uuid import UUID

import numpy as np

# Marker key inside the encoded payload so a decoder can recognize a
# tensor round-trip without disturbing arbitrary user dicts.
TENSOR_TAG = "__torch_tensor__"


def _encode_array_like(arr: "np.ndarray") -> dict:
    return {
        TENSOR_TAG: True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": arr.tobytes(),
    }


def msgpack_encoder(obj):
    """
    Custom encoder for non-standard types.

    IMPORTANT: msgpack traverses lists/dicts in optimized C code. This function
    is ONLY called for types msgpack doesn't recognize. This avoids the massive
    performance penalty of recursing through millions of tokens in Python.

    Handles: Path, UUID, Enum, datetime, Pydantic models, numpy scalars,
    numpy arrays, torch tensors, and dataclasses (e.g. renderers'
    ``MultiModalData`` / ``PlaceholderRange``). Tensors and ndarrays are
    encoded as ``{__torch_tensor__: True, dtype, shape, data}``.
    Does NOT handle: lists, dicts, basic types (msgpack does this natively in C).
    """
    if isinstance(obj, (Path, UUID)):
        return str(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return _encode_array_like(obj)
    elif (_torch := sys.modules.get("torch")) is not None and isinstance(
        obj, _torch.Tensor
    ):
        # Read torch off ``sys.modules`` instead of importing: text-only
        # consumers never load torch, so this branch stays cold for them.
        # Any tensor reaching the encoder implies torch is already in the
        # process (you can't construct one otherwise).
        arr = obj.detach().cpu().contiguous().numpy()
        return _encode_array_like(arr)
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    else:
        # raise on unknown types to make issues visible
        raise TypeError(f"Object of type {type(obj)} is not msgpack serializable")
