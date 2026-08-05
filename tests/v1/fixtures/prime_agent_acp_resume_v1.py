"""Two-segment Prime Agent exchange that requires live IPython state."""

import verifiers.v1 as vf


class PrimeAgentResumeTask(vf.Task):
    @vf.reward(weight=1.0)
    async def resumed(self, trace: vf.Trace) -> float:
        segments = trace.info.get("prime_agent_segments", [])
        if len(segments) != 2:
            return 0.0
        first, second = segments
        return float(
            "ready" in first["last_reply"].casefold()
            and second["last_reply"].strip() == "42"
        )


class PrimeAgentResumeEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            first = await interaction.turn(
                "Use IPython to set vf_acp_counter = 41. Reply with exactly READY."
            )
            segments = [first]
            if not first.terminated:
                segments.append(
                    await interaction.turn(
                        "Use IPython to increment the existing vf_acp_counter by one. "
                        "Reply with exactly its new integer value."
                    )
                )
            interaction.trace.info["prime_agent_segments"] = [
                {
                    "last_reply": segment.last_reply,
                    "terminated": segment.terminated,
                }
                for segment in segments
            ]


class PrimeAgentResumeTaskset(vf.Taskset):
    def load(self) -> list[PrimeAgentResumeTask]:
        return [
            PrimeAgentResumeTask(
                vf.TaskData(
                    idx=0,
                    prompt=None,
                    system_prompt=(
                        "Follow each user instruction exactly. Always perform requested "
                        "IPython operations before replying."
                    ),
                )
            )
        ]


__all__ = ["PrimeAgentResumeEnv", "PrimeAgentResumeTaskset"]
