import dataclasses
import hashlib
import logging
import os
import socket
import sys
import tempfile
from collections.abc import Container
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import zmq

logger = logging.getLogger(__name__)


# Marker key inside the encoded payload so the decoder can recognize a
# tensor round-trip without disturbing arbitrary user dicts.
TENSOR_TAG = "__torch_tensor__"

# Bounds the search in get_free_port. Each rejected probe is held open, so the OS has
# to offer a distinct port every attempt and the search converges in a few tries even
# with many ports already issued.
_PORT_PROBE_ATTEMPTS = 100

# A unix socket path goes in `sockaddr_un.sun_path`, which is 108 bytes on Linux and
# 104 on macOS, both including the terminating NUL. Hold to the smaller one everywhere
# rather than branching on the platform, and leave a few bytes spare.
_SUN_PATH_MAX = 100


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
    elif (_torch := sys.modules.get("torch")) is not None and isinstance(
        obj, _torch.Tensor
    ):
        # Read torch off ``sys.modules`` instead of importing: text-only
        # consumers never load torch, so this branch stays cold for
        # them. Any tensor reaching the encoder implies torch is
        # already in the process (you can't construct one otherwise).
        # ``isinstance`` is precise — the previous string-module check
        # also matched non-tensor torch objects (``torch.dtype``,
        # ``torch.device``, ``torchvision.*``) and crashed on
        # ``.detach()``.
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


def _ipc_socket_dir() -> Path:
    """Shortest writable directory to hold the unix socket files.

    The directory is charged against the same `sun_path` budget as the socket name, so
    a long temp dir is paid for out of the env id. macOS gives every process a per-user
    `TMPDIR` under `/var/folders/` that runs to about 48 bytes, against 4 for `/tmp`,
    which is most of the budget gone before the name starts. Prefer whichever candidate
    is shorter and actually writable: `/tmp` on a normal machine, `gettempdir()`
    wherever `/tmp` is not writable, which is the case this stopped hardcoding for.
    """
    candidates = sorted(
        {Path(tempfile.gettempdir()), Path("/tmp")},
        key=lambda directory: len(str(directory).encode()),
    )
    for directory in candidates:
        if os.access(directory, os.W_OK):
            return directory
    return Path(tempfile.gettempdir())


def _fit_socket_path(directory: Path, filename: str) -> Path:
    """Path under `directory` for `filename`, shortened if `sun_path` cannot hold it.

    Worker names carry the env id, which is arbitrarily long and often an `org/name`
    pair, so a name that fits on one machine can overflow on another. Overflowing means
    the worker fails to bind after the router has already come up, which is the same
    class of late startup failure this module is trying to remove. Replacing the tail
    with a digest of the whole name keeps distinct workers distinct.
    """
    if len(str(directory / filename).encode()) <= _SUN_PATH_MAX:
        return directory / filename
    digest = hashlib.sha256(filename.encode()).hexdigest()[:8]
    room = _SUN_PATH_MAX - len(str(directory / f"-{digest}").encode())
    if room < 1:
        raise RuntimeError(
            f"cannot fit a socket name under {directory}: the directory alone takes "
            f"{len(str(directory).encode())} of the {_SUN_PATH_MAX} bytes available"
        )
    head = filename.encode()[:room].decode(errors="ignore")
    return directory / f"{head}-{digest}"


def make_ipc_address(
    session_id: str, name: str, issued_ports: set[int] | None = None
) -> str:
    """Build an address for router-to-worker communication.

    Prefers a Unix domain socket, which needs no port and is faster, but falls back to
    loopback TCP wherever libzmq was built without ipc support. That is always the case
    on Windows, where `ipc://` cannot bind at all and the router dies in `__init__`.
    Capability is read from `zmq.has("ipc")` rather than sniffing the platform, so a
    libzmq built without ipc on any OS is handled too.

    The socket directory is no longer a hardcoded `/tmp`, so it also lands somewhere
    writable when `/tmp` is not, and the result is length-checked against `sun_path`.

    `issued_ports` is read and updated on the TCP path. A worker address is built in
    the parent but only bound in the child, after `load_environment`, so the port sits
    free across the spawn. Callers that allocate several addresses in a row must pass a
    shared set, or a later allocation can be handed a port an earlier worker has not
    bound yet.
    """
    if not zmq.has("ipc"):
        port = get_free_port(exclude=issued_ports if issued_ports is not None else ())
        if issued_ports is not None:
            issued_ports.add(port)
        return f"tcp://127.0.0.1:{port}"
    safe_name = name.replace("/", "--")
    directory = _ipc_socket_dir()
    return f"ipc://{_fit_socket_path(directory, f'vf-{session_id}-{safe_name}')}"


def ipc_path_of(address: str) -> str | None:
    """Filesystem path backing an `ipc://` address, or None for any other transport.

    Callers unlink these on shutdown. A TCP address has no file behind it, so it must
    not be handed to `os.unlink`.
    """
    prefix = "ipc://"
    return address[len(prefix) :] if address.startswith(prefix) else None


def get_free_port(exclude: Container[int] = ()) -> int:
    """Get a free port on the system, skipping any in `exclude`.

    The probe socket is closed before returning, so the port is only free until
    something binds it. Callers that bind later, and allocate more than one port in
    the meantime, have to pass back what they were already given.

    Rejected probes are held open rather than closed immediately, which forces the OS
    to offer a different port on the next attempt instead of possibly repeating the
    one just refused.
    """
    held: list[socket.socket] = []
    try:
        for _ in range(_PORT_PROBE_ATTEMPTS):
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            held.append(probe)
            probe.bind(("localhost", 0))
            port = probe.getsockname()[1]
            if port not in exclude:
                return port
        raise RuntimeError(
            f"no free port outside the set already issued, after {_PORT_PROBE_ATTEMPTS} attempts"
        )
    finally:
        for probe in held:
            probe.close()
