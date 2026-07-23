"""Two-segment ACP continuation with an MCP call after resume."""

import verifiers.v1 as vf

CODEWORD = "violet-cascade-731"
TOOL_STAMP = "resume-ok-9d2"


def _key(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


class ResumeToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "resume"

    @vf.tool
    def recall(self, codeword: str) -> str:
        """Return the supplied codeword with a private verification stamp."""
        return f"{codeword} [{TOOL_STAMP}]"


class ACPResumeTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(colocated=True)


class ACPResumeConfig(vf.TasksetConfig):
    task: ACPResumeTaskConfig = ACPResumeTaskConfig()


class ACPResumeTask(vf.Task[vf.TaskData, vf.State, ACPResumeTaskConfig]):
    tools = (ResumeToolset,)

    @vf.reward(weight=1.0)
    async def resumed(self, trace: vf.Trace) -> float:
        segments = trace.info.get("acp_segments", [])
        if len(segments) != 2:
            return 0.0
        first, second = segments
        resumed_tool_output = "\n".join(second["tool_outputs"])
        return float(
            bool(first["last_reply"].strip())
            and _key(CODEWORD) in _key(second["last_reply"])
            and _key(TOOL_STAMP) in _key(second["last_reply"])
            and _key(CODEWORD) in _key(resumed_tool_output)
            and _key(TOOL_STAMP) in _key(resumed_tool_output)
            and "tool" in second["roles"]
        )


class ACPResumeEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            first = await interaction.turn(
                f"Remember the codeword {CODEWORD}. Reply with exactly READY."
            )
            segments = [first]
            if not first.terminated:
                segments.append(
                    await interaction.turn(
                        "Call `resume_recall` with the codeword from my previous "
                        "message, then reply with exactly the tool result."
                    )
                )
            interaction.trace.info["acp_segments"] = [
                {
                    "roles": [message.role for message in segment.messages],
                    "tool_outputs": [
                        str(message.content)
                        for message in segment.messages
                        if message.role == "tool"
                    ],
                    "last_reply": segment.last_reply,
                    "terminated": segment.terminated,
                }
                for segment in segments
            ]


class ACPResumeTaskset(vf.Taskset[ACPResumeTask, ACPResumeConfig]):
    def load(self) -> list[ACPResumeTask]:
        return [
            ACPResumeTask(
                vf.TaskData(
                    idx=0,
                    prompt=None,
                    system_prompt=(
                        "Follow each user instruction exactly. Preserve conversational "
                        "context between turns and use requested tools."
                    ),
                ),
                self.config.task,
            )
        ]


__all__ = ["ACPResumeTaskset", "ACPResumeEnv"]


if __name__ == "__main__":
    ResumeToolset.run()
