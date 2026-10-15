"""Multi-turn echo driven by a scripted user — the single user-sim mechanism.

The env's `run()` scripts the user through an interaction. The task is
prompt-less, so the first `turn(phrase)` opens the conversation; each later turn
resumes the harness onto the accreted conversation, and leaving the `async with`
closes the interaction (the trace stops as `user_closed`).
"""

import verifiers.v1 as vf

PHRASES = ["hello world", "goodbye world"]


def _key(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isalnum())


class EchoUserSimConfig(vf.TasksetConfig):
    phrases: list[str] = PHRASES


class EchoUserSimData(vf.TaskData):
    phrases: list[str]


class EchoUserSimTask(vf.Task[EchoUserSimData, vf.State, vf.TaskConfig]):
    @vf.reward(weight=1.0)
    async def echoed(self, trace: vf.Trace) -> float:
        # Match each phrase to a later reply, in order: a harness may spend turns on
        # tool calls, so replies and phrases need not pair up one to one.
        replies = iter(_key(m.content or "") for m in trace.assistant_messages)
        matched = sum(
            any(_key(phrase) in reply for reply in replies)
            for phrase in self.data.phrases
        )
        return matched / len(self.data.phrases)


class EchoUserSimEnv(vf.SingleAgentEnv):
    """Scripts the user side: opens with the first phrase, follows with the rest."""

    async def run(self, task, agents):
        # An interaction scripting the user: the task carries no prompt, so the
        # first turn opens the conversation.
        async with agents.agent.interaction(task) as interaction:
            for phrase in task.data.phrases:
                instruction = (
                    "Answer directly without using tools or running code. Your "
                    f"reply must contain this exact phrase: {phrase}"
                )
                if (await interaction.turn(instruction)).terminated:
                    break


class EchoUserSimTaskset(vf.Taskset[EchoUserSimTask, EchoUserSimConfig]):
    def load(self) -> list[EchoUserSimTask]:
        return [
            EchoUserSimTask(
                EchoUserSimData(
                    idx=0,
                    # No prompt: the scripted user opens the conversation.
                    prompt=None,
                    phrases=self.config.phrases,
                )
            )
        ]


__all__ = ["EchoUserSimEnv", "EchoUserSimTaskset"]
