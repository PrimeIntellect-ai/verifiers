from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


@dataclass
class MCPServerConfig:
    name: str
    transport: Literal["stdio", "http"] = "stdio"
    description: str = ""
    # stdio params
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    # http params
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
