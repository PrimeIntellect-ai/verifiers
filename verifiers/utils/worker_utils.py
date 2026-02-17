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


def get_free_port_pair() -> int:
    """Get a free port whose successor (port+1) is also free.

    Used when the server needs two adjacent ports (e.g. work + health).
    """
    for _ in range(10):
        with (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1,
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2,
        ):
            s1.bind(("localhost", 0))
            port = s1.getsockname()[1]
            try:
                s2.bind(("localhost", port + 1))
                return port
            except OSError:
                continue
    raise RuntimeError("Could not find a free port pair after 10 attempts")
