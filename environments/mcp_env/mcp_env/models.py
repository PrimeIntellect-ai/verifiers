from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str] | None = None
    env: Dict[str, str] | None = None
    description: str = ""
