import verifiers.v1 as vf

from .config import NestedToolsetConfig


class NestedToolset(vf.Toolset[NestedToolsetConfig]):
    @vf.tool
    def call_harness(self, prompt: str) -> str:
        return prompt.upper()
