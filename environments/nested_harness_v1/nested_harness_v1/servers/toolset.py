import verifiers.v1 as vf


class NestedToolsetConfig(vf.ToolsetConfig):
    loader: str = "nested_harness_v1.servers.toolset:NestedToolset"
    name: str | None = "nested"


class NestedToolset(vf.Toolset[NestedToolsetConfig]):
    @vf.tool
    def call_harness(self, prompt: str) -> str:
        return prompt.upper()
