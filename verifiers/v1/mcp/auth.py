"""Private credentials for framework-managed MCP endpoints."""

import hmac
from collections.abc import Iterable


class MCPURL(str):
    """A URL with a separate bearer credential for harness MCP clients.

    String behavior keeps the existing harness URL interface; serialization of the URL
    alone never grants access to the endpoint.
    """

    mcp_secret: str

    def __new__(cls, url: str, secret: str):
        value = super().__new__(cls, url)
        value.mcp_secret = secret
        return value


def authorized(headers: Iterable[tuple[bytes, bytes]], secret: str) -> bool:
    """Accept one Authorization header containing this server's bearer credential."""
    credentials = [value for name, value in headers if name.lower() == b"authorization"]
    return (
        bool(secret)
        and len(credentials) == 1
        and hmac.compare_digest(credentials[0], f"Bearer {secret}".encode())
    )
