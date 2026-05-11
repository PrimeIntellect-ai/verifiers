import dataclasses
import logging
import socket
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


# Marker key inside the encoded payload so the decoder can recognize a
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
    encoded as ``{__torch_tensor__: True, dtype, shape, data}`` so the
    receiving side can rehydrate them via ``decode_tensor_payload``.
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
    elif type(obj).__module__.startswith("torch"):
        # Lazy: only import torch when a tensor actually shows up so
        # text-only consumers don't pay for it.
        arr = obj.detach().cpu().contiguous().numpy()
        return _encode_array_like(arr)
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    else:
        # raise on unknown types to make issues visible
        raise TypeError(f"Object of type {type(obj)} is not msgpack serializable")


def decode_tensor_payload(obj: Any, *, to_torch: bool = True):
    """Rehydrate a tensor encoded by :func:`msgpack_encoder`.

    Accepts either the encoded dict shape (``{__torch_tensor__: True,
    dtype, shape, data}``) or an already-rehydrated tensor/ndarray and
    returns a torch tensor (or numpy ndarray if ``to_torch=False``).
    """
    if obj is None:
        return None
    if isinstance(obj, dict) and obj.get(TENSOR_TAG):
        arr = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(
            obj["shape"]
        )
        if to_torch:
            # importlib (not ``import torch``) so static type checkers in
            # downstream consumers without torch installed don't fail on
            # unresolved-import. Torch is a soft runtime dep here: callers
            # that pass ``to_torch=True`` are expected to have it.
            import importlib

            torch = importlib.import_module("torch")
            return torch.from_numpy(arr.copy())
        return arr.copy()
    # Already a tensor / ndarray — pass through.
    return obj


def walk_decode_tensors(obj: Any, *, to_torch: bool = True):
    """Recursively decode any tensor payloads inside nested dicts/lists.

    Used by the orchestrator after msgpack-decoding a multimodal sidecar
    so downstream code sees real tensors without each consumer threading
    the decode call manually.
    """
    if isinstance(obj, dict):
        if obj.get(TENSOR_TAG):
            return decode_tensor_payload(obj, to_torch=to_torch)
        return {k: walk_decode_tensors(v, to_torch=to_torch) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk_decode_tensors(v, to_torch=to_torch) for v in obj]
    return obj


def make_ipc_address(session_id: str, name: str) -> str:
    """Build an IPC address for inter-process communication."""
    return f"ipc:///tmp/vf-{session_id}-{name.replace('/', '--')}"


def get_free_port() -> int:
    """Get a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]
