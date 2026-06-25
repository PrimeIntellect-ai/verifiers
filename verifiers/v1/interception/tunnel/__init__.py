"""Tunnels: make the host interception server reachable from a remote harness runtime.

`Tunnel` is the contract; the concrete tunnels match the `InterceptionConfig` types — `PrimeTunnel`
(prime_tunnel/frpc), `CustomTunnel` (a bring-your-own endpoint: reverse proxy or direct bind). The
interception pool picks the class matching its config and instantiates one tunnel per server (each
server owns its own).
"""

from verifiers.v1.interception.tunnel.base import Tunnel
from verifiers.v1.interception.tunnel.custom import CustomTunnel
from verifiers.v1.interception.tunnel.prime import PrimeTunnel

__all__ = [
    "Tunnel",
    "PrimeTunnel",
    "CustomTunnel",
]
