"""Two-segment ACP continuation with an MCP call after resume."""

import verifiers.v1 as vf

CODEWORD = "violet-cascade-731"
TOOL_STAMP = "resume-ok-9d2"
SEEDED_CALL_ID = "seed_recall"
SEEDED_REPLY = "imported-assistant-4f8"


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
    seeded: bool = False


class ACPResumeConfig(vf.TasksetConfig):
    task: ACPResumeTaskConfig = ACPResumeTaskConfig()


class ACPResumeTask(vf.Task[vf.TaskData, vf.State, ACPResumeTaskConfig]):
    @classmethod
    def toolsets(cls, config: ACPResumeTaskConfig) -> list[vf.Toolset]:
        return [ResumeToolset(config.tools)]

    @vf.reward(weight=1.0)
    async def resumed(self, trace: vf.Trace) -> float:
        segments = trace.info.get("acp_segments", [])
        if len(segments) != 2:
            return 0.0
        first, second = segments
        expected_first = SEEDED_REPLY if trace.info["acp_seeded"] else "READY"
        resumed_tool_output = "\n".join(second["tool_outputs"])
        return float(
            _key(first["last_reply"]) == _key(expected_first)
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
                None
                if task.config.seeded
                else f"Remember the codeword {CODEWORD}. Reply with exactly READY."
            )
            segments = [first]
            if not first.terminated:
                segments.append(
                    await interaction.turn(
                        "Call `resume_recall` with the codeword from our conversation "
                        "history, then reply with exactly the tool result."
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
            interaction.trace.info["acp_seeded"] = task.config.seeded


class ACPResumeTaskset(vf.Taskset[ACPResumeTask, ACPResumeConfig]):
    def load(self) -> list[ACPResumeTask]:
        prompt = None
        if self.config.task.seeded:
            prompt = [
                vf.UserMessage(content=f"Remember the codeword {CODEWORD}."),
                vf.AssistantMessage(
                    tool_calls=[
                        vf.ToolCall(
                            id=SEEDED_CALL_ID,
                            name="resume_recall",
                            arguments=f'{{"codeword":"{CODEWORD}"}}',
                        )
                    ]
                ),
                vf.ToolMessage(
                    content=f"{CODEWORD} [{TOOL_STAMP}]",
                    tool_call_id=SEEDED_CALL_ID,
                    name="resume_recall",
                ),
                vf.AssistantMessage(content=SEEDED_REPLY),
                vf.UserMessage(
                    content="Reply with exactly the previous assistant reply."
                ),
            ]
        return [
            ACPResumeTask(
                vf.TaskData(
                    idx=0,
                    prompt=prompt,
                    system_prompt=None
                    if self.config.task.seeded
                    else (
                        "Follow each user instruction exactly. Preserve conversational "
                        "context between turns and use requested tools."
                    ),
                ),
                self.config.task,
            )
        ]


__all__ = ["ACPResumeEnv", "ACPResumeTaskset"]


if __name__ == "__main__":
    ResumeToolset.run()
