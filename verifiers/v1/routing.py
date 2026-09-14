"""Immutable paired MoE routing and codecs for the single-turn TRR v1 transport.

Weights are the captured FP32 coefficients, not scores to normalize again. The MVP
uses Qwen3 normalized top-k weights (route scale 1). A false validity bit denotes
only a final sampled token which inference has not forwarded yet.
"""

from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

FORMAT_VERSION = 1
_ID_DTYPES = {"uint8": np.dtype("u1"), "uint16": np.dtype("<u2")}
_WEIGHT_DTYPE = np.dtype("<f4")


def _shape(value: Any) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(n) is not int for n in value)
        or value[0] < 0
        or value[1] <= 0
        or value[2] <= 0
    ):
        raise ValueError("routing shape must be [tokens >= 0, layers > 0, top_k > 0]")
    return tuple(value)


def _id_dtype(value: Any) -> np.dtype:
    if not isinstance(value, str) or value not in _ID_DTYPES:
        raise ValueError("routing ID dtype must be uint8 or uint16 (little-endian)")
    return _ID_DTYPES[value]


def validate_ids(ids: np.ndarray, num_experts: int | None = None) -> None:
    if not isinstance(ids, np.ndarray):
        raise ValueError("routing IDs must be a NumPy array")  # noqa: TRY004 - ingress validation
    _shape(ids.shape)
    if ids.dtype not in _ID_DTYPES.values():
        raise ValueError("routing ID dtype must be uint8 or uint16 (little-endian)")
    if num_experts is not None:
        if type(num_experts) is not int or num_experts <= 0:
            raise ValueError("num_experts must be a positive integer")
        if ids.shape[2] > num_experts or (ids.size and int(ids.max()) >= num_experts):
            raise ValueError("routing expert ID/top_k exceeds model expert range")


def _array_bytes(arr: np.ndarray) -> bytes:
    """Reuse only a complete contiguous immutable buffer, never a larger allocation."""
    owner = arr
    while isinstance(owner, (np.ndarray, memoryview)):
        owner = owner.base if isinstance(owner, np.ndarray) else owner.obj
    if isinstance(owner, bytes) and arr.flags.c_contiguous and len(owner) == arr.nbytes:
        return owner
    return arr.tobytes(order="C")


def _immutable(arr: np.ndarray) -> np.ndarray:
    # Read-only alone is insufficient: the owner could enable writes again. Reuse
    # complete bytes buffers (e.g. freshly decoded HTTP/msgpack), but copy partial
    # slices so a small graph node cannot pin a full response.
    return np.frombuffer(_array_bytes(arr), dtype=arr.dtype).reshape(arr.shape)


def _joined_array(parts: Sequence[np.ndarray]) -> np.ndarray:
    # RoutingData fields are contiguous and already have matching dtypes/layouts.
    # Join directly into immutable storage instead of concatenate -> tobytes.
    shape = (sum(len(part) for part in parts), *parts[0].shape[1:])
    raw = b"".join(memoryview(part) for part in parts)
    return np.frombuffer(raw, dtype=parts[0].dtype).reshape(shape)


@dataclass(frozen=True, eq=False)
class RoutingData:
    """Read-only IDs/FP32 weights [T,L,K] and captured-row validity [T].

    Mutable inputs and partial slices are copied. Complete immutable byte buffers
    can be reused. No caller can change array data through the source or by enabling
    NumPy writes. Model-specific bounds are checked by ``validate(num_experts=...)``.
    """

    ids: np.ndarray
    weights: np.ndarray
    valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        validate_ids(self.ids)
        if (
            not isinstance(self.weights, np.ndarray)
            or self.weights.dtype != _WEIGHT_DTYPE
        ):
            raise ValueError("routing weights dtype must be little-endian float32")
        if self.weights.shape != self.ids.shape:
            raise ValueError("routing IDs and weights must have equal [T,L,K] shape")
        # Scalar reductions avoid full-size temporary masks. NaNs propagate.
        minimum = self.weights.min(initial=0)
        maximum = self.weights.max(initial=0)
        if not (np.isfinite(minimum) and np.isfinite(maximum)):
            raise ValueError("routing weights must be finite")
        if minimum < 0 or maximum > 1:
            raise ValueError("routing weights must be in [0, 1] for TRR v1")
        valid = self.valid
        if valid is None:
            valid = np.frombuffer(b"\x01" * self.ids.shape[0], dtype=np.bool_)
        if (
            not isinstance(valid, np.ndarray)
            or valid.dtype != np.dtype("bool")
            or valid.shape != (self.ids.shape[0],)
        ):
            raise ValueError("routing valid must be bool [T]")
        if not valid[:-1].all():
            raise ValueError("only the final routing row may be invalid")
        if len(valid) and not valid[-1] and np.any(self.weights[-1] != 0):
            raise ValueError("invalid routing rows must have zero placeholder weights")
        object.__setattr__(self, "ids", _immutable(self.ids))
        object.__setattr__(self, "weights", _immutable(self.weights))
        object.__setattr__(self, "valid", _immutable(valid))

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __copy__(self) -> RoutingData:
        return self

    def __deepcopy__(self, memo: dict) -> RoutingData:
        return self

    def validate(self, *, num_experts: int) -> None:
        """Check the model-dependent bound not available in HTTP/graph metadata."""
        validate_ids(self.ids, num_experts)

    def to_wire(self) -> dict[str, Any]:
        """Encode raw bytes for graph msgpack (not the HTTP base64 envelope)."""
        return {
            "__routing__": True,
            "format_version": FORMAT_VERSION,
            "data": _array_bytes(self.ids),
            "shape": list(self.ids.shape),
            "dtype": str(self.ids.dtype),
            "weights": {"data": _array_bytes(self.weights), "dtype": "float32"},
            "valid": _array_bytes(self.valid),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> RoutingData:
        if payload.get("__routing__") is not True:
            raise ValueError("missing graph routing marker")
        _version(payload, full=True)
        shape = _shape(payload.get("shape"))
        dtype = _id_dtype(payload.get("dtype"))
        count = math.prod(shape)
        ids = _raw_array(payload.get("data"), dtype, count).reshape(shape)
        weights = _weights(payload, shape, base64_encoded=False)
        validity = _raw_array(payload.get("valid"), np.dtype("u1"), shape[0])
        if np.any(validity > 1):
            raise ValueError("routing valid bytes must be 0 or 1")
        return cls(ids, weights, validity.view(np.bool_))


def _version(payload: Mapping[str, Any], *, full: bool) -> None:
    version = payload.get("format_version", 0)
    if type(version) is not int or version != (FORMAT_VERSION if full else 0):
        raise ValueError("unsupported routing format_version or missing weights")


def _raw_array(data: Any, dtype: np.dtype, count: int) -> np.ndarray:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("routing data must be bytes")  # noqa: TRY004 - ingress validation
    if memoryview(data).nbytes != count * dtype.itemsize:
        raise ValueError("routing byte count does not match shape and dtype")
    return np.frombuffer(data, dtype=dtype, count=count)


def _base64_array(data: Any, dtype: np.dtype, count: int) -> np.ndarray:
    if not isinstance(data, (str, bytes, bytearray, memoryview)):
        raise ValueError("routing data must be base64 text or bytes")  # noqa: TRY004 - ingress validation
    size = memoryview(data).nbytes if isinstance(data, memoryview) else len(data)
    if size != 4 * ((count * dtype.itemsize + 2) // 3):
        raise ValueError("routing base64 byte count does not match shape and dtype")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid routing base64 data") from exc
    return _raw_array(raw, dtype, count)


def _weights(
    payload: Mapping[str, Any], shape: tuple, *, base64_encoded: bool
) -> np.ndarray:
    weights = payload.get("weights")
    if not isinstance(weights, Mapping) or weights.get("dtype") != "float32":
        raise ValueError("routing weights must include data and dtype=float32")
    decode = _base64_array if base64_encoded else _raw_array
    return decode(weights.get("data"), _WEIGHT_DTYPE, math.prod(shape)).reshape(shape)


def decode_routing_payload(
    payload: Mapping[str, Any],
    *,
    require_weights: bool = False,
    num_experts: int | None = None,
) -> tuple[RoutingData | np.ndarray, int]:
    """Validate the Prime HTTP envelope and return paired or legacy routing + start.

    Only the legacy format permits omitted dtype/start (historically uint8/0).
    Absence of weights is legacy, never an implicit zero coefficient stream.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("routing payload must be an object")  # noqa: TRY004 - ingress validation
    full = "weights" in payload
    if require_weights and not full:
        raise ValueError("full routing replay requires captured weights")
    _version(payload, full=full)
    shape = _shape(payload.get("shape"))
    dtype = _id_dtype(payload.get("dtype", None if full else "uint8"))
    start = payload.get("start", None if full else 0)
    if type(start) is not int or start < 0:
        raise ValueError("routing start must be a nonnegative integer")
    ids = _base64_array(payload.get("data"), dtype, math.prod(shape)).reshape(shape)
    validate_ids(ids, num_experts)
    if not full:
        return ids, start
    return RoutingData(ids, _weights(payload, shape, base64_encoded=True)), start


def slice_routing(
    data: RoutingData | np.ndarray, start: int, stop: int
) -> RoutingData | np.ndarray:
    """Own only a contiguous token span; do not retain a whole response allocation."""
    if not 0 <= start <= stop <= len(data):
        raise ValueError("routing token slice is out of range")
    if isinstance(data, RoutingData):
        if start == 0 and stop == len(data):
            return data
        return RoutingData(
            data.ids[start:stop], data.weights[start:stop], data.valid[start:stop]
        )
    return data[start:stop].copy()


def concatenate_routing(
    parts: Sequence[RoutingData | np.ndarray],
) -> RoutingData | np.ndarray:
    if not parts:
        raise ValueError("cannot concatenate empty routing parts")
    full = [isinstance(part, RoutingData) for part in parts]
    if not any(full):
        return np.concatenate(parts, axis=0)
    if not all(full):
        raise ValueError("cannot mix full routing and legacy IDs-only routing")
    first = parts[0]
    if len(parts) == 1:
        return first
    if any(
        part.ids.shape[1:] != first.ids.shape[1:] or part.ids.dtype != first.ids.dtype
        for part in parts
    ):
        raise ValueError(
            "routing parts must have matching layer/top_k layout and ID dtype"
        )
    return RoutingData(
        _joined_array([part.ids for part in parts]),
        _joined_array([part.weights for part in parts]),
        _joined_array([part.valid for part in parts]),
    )


def complete_routing_capture(
    data: RoutingData, *, total_tokens: int, completion_tokens: int
) -> RoutingData:
    """Allow only exact capture coverage or one unforwarded final sampled token."""
    if not data.valid.all():
        raise ValueError("capture input must contain only real routing rows")
    if len(data) == total_tokens:
        return data
    if completion_tokens <= 0 or len(data) != total_tokens - 1:
        raise ValueError(
            "full routing capture must cover P+C or P+C-1 with a final sampled token"
        )
    shape = (1, *data.ids.shape[1:])
    terminal = RoutingData(
        np.zeros(shape, dtype=data.ids.dtype),
        np.zeros(shape, dtype=_WEIGHT_DTYPE),
        np.zeros(1, dtype=np.bool_),
    )
    return concatenate_routing([data, terminal])
