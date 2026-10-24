from verifiers.v1.configs.interception import (
    BaseTunnelConfig,
    CustomTunnelConfig,
    PrimeTunnelConfig,
    TunnelConfig,
)
from verifiers.v1.interception.tunnel.base import Tunnel
from verifiers.v1.interception.tunnel.custom import CustomTunnel
from verifiers.v1.interception.tunnel.prime import PrimeTunnel


def make_tunnel(config: TunnelConfig) -> Tunnel:
    """The tunnel for a config, picked by type."""
    if isinstance(config, CustomTunnelConfig):
        return CustomTunnel(config)
    return PrimeTunnel(config)


__all__ = [
    "BaseTunnelConfig",
    "CustomTunnel",
    "CustomTunnelConfig",
    "PrimeTunnel",
    "PrimeTunnelConfig",
    "Tunnel",
    "TunnelConfig",
    "make_tunnel",
]
