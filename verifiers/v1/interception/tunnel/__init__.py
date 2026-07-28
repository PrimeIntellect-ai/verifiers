from typing import Annotated

from pydantic import Field

from verifiers.v1.interception.tunnel.base import (
    BaseTunnelConfig,
    Endpoint,
    FixedEndpoint,
    Tunnel,
)
from verifiers.v1.interception.tunnel.custom import CustomTunnel, CustomTunnelConfig
from verifiers.v1.interception.tunnel.prime import PrimeTunnel, PrimeTunnelConfig

# Discriminated on `type` so the CLI selects with `--interception.tunnel.type prime|custom`.
TunnelConfig = Annotated[
    PrimeTunnelConfig | CustomTunnelConfig, Field(discriminator="type")
]


def make_tunnel(config: TunnelConfig) -> Tunnel:
    """The tunnel for a config, picked by type."""
    if isinstance(config, CustomTunnelConfig):
        return CustomTunnel(config)
    return PrimeTunnel(config)


__all__ = [
    "BaseTunnelConfig",
    "CustomTunnel",
    "CustomTunnelConfig",
    "Endpoint",
    "FixedEndpoint",
    "PrimeTunnel",
    "PrimeTunnelConfig",
    "Tunnel",
    "TunnelConfig",
    "make_tunnel",
]
