import asyncio
import logging
import socket
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from verifiers.utils.logging_utils import print_time

if TYPE_CHECKING:
    from verifiers.workers.client.env_client import EnvClient

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


async def wait_for_env_server(
    env_client: "EnvClient",
    interval: float = 1,
    log_interval: float = 10,
    timeout: float = 3600,  # 1h
) -> None:
    wait_time = 0
    logger.debug(f"Starting pinging environment server at {env_client.address}")
    while wait_time < timeout:
        try:
            await env_client.health(timeout=1)  # quick timeout
            logger.debug(
                f"Environment server at {env_client.address} is ready after {print_time(wait_time)}"
            )
            return
        except Exception as e:
            if wait_time % log_interval == 0 and wait_time > 0:
                logger.warning(
                    f"Environment server at {env_client.address} was not reached after {print_time(wait_time)} (Error: {e})"
                )
            await asyncio.sleep(interval)
            wait_time += interval
    msg = f"Environment server at {env_client.address} is not ready after {print_time(wait_time)} (>{print_time(timeout)}). Aborting..."
    logger.error(msg)
    raise TimeoutError(msg)
