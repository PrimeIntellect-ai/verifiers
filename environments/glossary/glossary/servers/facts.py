import json
from pathlib import Path

import verifiers.v1 as vf

FACTS: dict[str, str] = json.loads((Path(__file__).parent / "facts.json").read_text())


class GlossaryToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "facts"

    @vf.tool
    def lookup(self, name: str) -> str:
        """Look up what a person or thing is known for."""
        return FACTS.get(name.strip().lower(), "no entry found")


if __name__ == "__main__":
    GlossaryToolset.run()
