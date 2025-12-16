from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


Transport = Literal["stdio", "streamablehttp"]


@dataclass
class MCPServerConfig:
    name: str
    transport: Transport = "stdio"
    command: Optional[str] = None
    args: List[str] | None = None
    env: Dict[str, str] | None = None
    headers: Dict[str, str] | None = None
    url: str | None = None
    description: str = ""
