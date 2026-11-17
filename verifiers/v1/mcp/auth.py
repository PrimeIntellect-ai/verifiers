"""Private credentials for framework-managed MCP endpoints."""

import hmac
from collections.abc import Mapping


class MCPURL(str):
    """A URL with a separate bearer credential for harness MCP clients.

    String behavior keeps the existing harness URL interface; serialization of the URL
    alone never grants access to the endpoint.
    """

    def __new__(cls, url: str, secret: str):
        value = super().__new__(cls, url)
        value.mcp_secret = secret
        return value


def authorized(headers: Mapping[str, str], secret: str) -> bool:
    """Accept exactly the bearer minted for this managed server."""
    value = headers.get("authorization", "")
    return bool(secret) and hmac.compare_digest(
        value.encode(), f"Bearer {secret}".encode()
    )
