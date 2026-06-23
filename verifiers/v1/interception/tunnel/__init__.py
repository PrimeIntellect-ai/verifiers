"""Tunnels: make the host interception server reachable from a remote harness runtime.

`Tunnel` is the contract; the concrete tunnels match the `InterceptionConfig` types — `PrimeTunnel`
(prime_tunnel/frpc), `CustomTunnel` (a bring-your-own endpoint: reverse proxy or direct bind).
`tunnel_cls` picks the class for a config; the interception pool holds that class and instantiates one
tunnel per server (each server owns its own).
"""

from verifiers.v1.interception.config import (
    CustomInterceptionConfig,
    InterceptionConfig,
)
from verifiers.v1.interception.tunnel.base import Tunnel
from verifiers.v1.interception.tunnel.custom import CustomTunnel
from verifiers.v1.interception.tunnel.prime import PrimeTunnel


def tunnel_cls(config: InterceptionConfig) -> type[Tunnel]:
    """The tunnel class matching a config's type — the host-side counterpart to `_runtime_cls`."""
    if isinstance(config, CustomInterceptionConfig):
        return CustomTunnel
    return PrimeTunnel


__all__ = [
    "Tunnel",
    "PrimeTunnel",
    "CustomTunnel",
    "tunnel_cls",
]
