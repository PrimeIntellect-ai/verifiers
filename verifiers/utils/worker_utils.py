import logging
import socket
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


def msgpack_encoder(obj):
    """
    Custom encoder for non-standard types.

    IMPORTANT: msgpack traverses lists/dicts in optimized C code. This function
    is ONLY called for types msgpack doesn't recognize. This avoids the massive
    performance penalty of recursing through millions of tokens in Python.

    Handles: Path, UUID, Enum, datetime, Pydantic models, numpy scalars.
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
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    else:
        # raise on unknown types to make issues visible
        raise TypeError(f"Object of type {type(obj)} is not msgpack serializable")


def get_free_port() -> int:
    """Get a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


_reserved_sockets: list[socket.socket] = []


def _make_reusable_socket(port: int = 0) -> socket.socket:
    """Create a TCP socket with SO_REUSEADDR bound to the given port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("localhost", port))
    except OSError:
        s.close()
        raise
    return s


def get_free_port_pair() -> int:
    """Get a free port whose successor (port+1) is also free.

    The sockets are kept alive in ``_reserved_sockets`` to prevent the OS
    from reassigning the ports before the caller can bind them.  Call
    :func:`release_reserved_ports` once the ports have been bound (e.g.
    by a subprocess) or are no longer needed.
    """
    for _ in range(10):
        s1 = _make_reusable_socket()
        port = s1.getsockname()[1]
        try:
            s2 = _make_reusable_socket(port + 1)
        except OSError:
            s1.close()
            continue
        _reserved_sockets.extend([s1, s2])
        return port
    raise RuntimeError("Could not find a free port pair after 10 attempts")


def release_reserved_ports() -> None:
    """Close all sockets held by :func:`get_free_port_pair`.

    On macOS, ZMQ cannot bind a port that is held open by another socket
    (even with ``SO_REUSEADDR``), so reserved sockets must be released
    before a subprocess can bind the same ports.  With the ``spawn``
    multiprocessing context the subprocess does not inherit file
    descriptors, so the reservation is only useful for preventing the
    *main* process from accidentally reusing the port.
    """
    for s in _reserved_sockets:
        try:
            s.close()
        except OSError:
            pass
    _reserved_sockets.clear()
