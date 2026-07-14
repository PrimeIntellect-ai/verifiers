"""Internal execution-network policy shared by runtime backends."""

from pydantic_config import BaseConfig


class NetworkPolicy(BaseConfig):
    """Agent egress rules applied at the trusted setup-to-execution boundary.

    Framework routes such as model and MCP endpoints are passed separately to the
    runtime. They remain reachable regardless of these rules and may be rewritten
    through backend-specific relays.

    This stays internal until network target syntax and backend support are defined.
    """

    allow: list[str] | None = None
    """Allowed network targets. None leaves arbitrary egress enabled; [] denies it."""
    block: list[str] = []
    """Blocked arbitrary-egress targets. These take precedence over allow rules."""

    @property
    def restricted(self) -> bool:
        return self.allow is not None or bool(self.block)
