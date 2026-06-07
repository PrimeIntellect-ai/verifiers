import verifiers.v1 as vf


class SubagentToolsetConfig(vf.ToolsetConfig):
    loader: str = "hello_subagent_v1.servers.toolset:SubagentToolset"
    name: str | None = "subagent"


class SubagentToolset(vf.Toolset[SubagentToolsetConfig]):
    @vf.tool(extends={"subagent_calls": "state.extras.subagent_calls"})
    def ask_subagent(self, name: str) -> dict:
        answer = f"hello {name}"
        return {
            "content": answer,
            "subagent_calls": [{"name": name, "answer": answer}],
        }
