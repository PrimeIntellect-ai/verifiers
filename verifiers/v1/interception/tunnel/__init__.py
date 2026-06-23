"""Tunnels: make the host interception server reachable from a remote harness runtime.

`Tunnel` is the contract; the concrete tunnels match the `InterceptionConfig` types — `PrimeTunnel`
(prime_tunnel/frpc), `CustomTunnel` (a bring-your-own reverse proxy), `DirectTunnel` (bind a reachable
host interface directly). `tunnel_cls` picks the class for a config; the interception pool holds that
class and instantiates one tunnel per server (each server owns its own).
"""

from verifiers.v1.interception.config import (
    BaseInterceptionConfig,
    CustomInterceptionConfig,
    DirectInterceptionConfig,
)
from verifiers.v1.interception.tunnel.base import Tunnel
from verifiers.v1.interception.tunnel.custom import CustomTunnel
from verifiers.v1.interception.tunnel.direct import DirectTunnel
from verifiers.v1.interception.tunnel.prime import PrimeTunnel


def tunnel_cls(config: BaseInterceptionConfig) -> type[Tunnel]:
    """The tunnel class matching a config's type — the host-side counterpart to `_runtime_cls`."""
    if isinstance(config, CustomInterceptionConfig):
        return CustomTunnel
    if isinstance(config, DirectInterceptionConfig):
        return DirectTunnel
    return PrimeTunnel


__all__ = [
    "Tunnel",
    "PrimeTunnel",
    "CustomTunnel",
    "DirectTunnel",
    "tunnel_cls",
]
