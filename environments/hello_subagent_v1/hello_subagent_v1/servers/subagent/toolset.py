import verifiers.v1 as vf

from .config import SubagentToolsetConfig


class SubagentToolset(vf.Toolset[SubagentToolsetConfig]):
    @vf.tool(extends={"subagent_calls": "state.extras.subagent_calls"})
    def ask_subagent(self, name: str) -> dict:
        answer = f"hello {name}"
        return {
            "content": answer,
            "subagent_calls": [{"name": name, "answer": answer}],
        }
