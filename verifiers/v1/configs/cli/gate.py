"""Configuration for `vf-gate`, which shows and resizes the host's request gates."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_config import BaseConfig


class GateConfig(BaseConfig):
    resize: int | None = Field(None, ge=1)
    """Set the cap of the live gates to this many requests in flight (only `model`'s, if
    given). It holds until no process uses the gate. None = show the gates."""
    model: str | None = None
    """Only this model's gates."""
    dir: Path | None = None
    """The gate directory (`client.request_gate.dir`). None = the host's default."""
    output: Literal["table", "json"] = "table"
