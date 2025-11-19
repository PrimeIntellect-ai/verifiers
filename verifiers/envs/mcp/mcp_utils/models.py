from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
