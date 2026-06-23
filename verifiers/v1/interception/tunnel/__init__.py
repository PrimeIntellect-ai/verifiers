"""Tunnels: make the host interception server reachable from a remote harness runtime.

`Tunnel` is the contract; the concrete tunnels match the `InterceptionConfig` types — `PrimeTunnel`
(prime_tunnel/frpc), `ModalTunnel` (`modal.forward`), `CustomTunnel` (a bring-your-own reverse
proxy). `make_tunnel` builds the tunnel for a config — the host-side counterpart to `make_runtime`.
"""

from verifiers.v1.interception.config import (
    BaseInterceptionConfig,
    CustomInterceptionConfig,
    ModalInterceptionConfig,
)
from verifiers.v1.interception.tunnel.base import Tunnel
from verifiers.v1.interception.tunnel.custom import CustomTunnel
from verifiers.v1.interception.tunnel.modal import ModalTunnel
from verifiers.v1.interception.tunnel.prime import PrimeTunnel


def _tunnel_cls(config: BaseInterceptionConfig) -> type[Tunnel]:
    if isinstance(config, ModalInterceptionConfig):
        return ModalTunnel
    if isinstance(config, CustomInterceptionConfig):
        return CustomTunnel
    return PrimeTunnel


def make_tunnel(config: BaseInterceptionConfig) -> Tunnel:
    """Build the tunnel matching a config — the host-side counterpart to `make_runtime`."""
    return _tunnel_cls(config)(config)


__all__ = [
    "Tunnel",
    "PrimeTunnel",
    "ModalTunnel",
    "CustomTunnel",
    "make_tunnel",
]
