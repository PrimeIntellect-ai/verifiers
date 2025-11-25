from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server.
    
    Args:
        name: Unique identifier for this server
        command: Command to run the server (required for stdio transport, optional for http)
        args: Arguments to pass to the command
        env: Environment variables for the server process
        url: URL for HTTP/Streamable HTTP transport (alternative to command)
    """
    name: str
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None  # For HTTP transport - can specify here instead of http_urls dict
