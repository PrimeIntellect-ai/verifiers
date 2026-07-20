"""Multi-turn echo driven by a scripted user — the single user-sim mechanism.

The env's `run()` scripts the user through a chat session. The task is
prompt-less, so the first `turn(phrase)` opens the conversation; each later turn
resumes the harness onto the accreted conversation, and leaving the `async with`
closes the chat (the trace stops as `user_closed`).
"""

import verifiers.v1 as vf

PHRASES = ["hello world", "goodbye world"]
SYSTEM = "Repeat the user's message back to them exactly, with no extra words."


def _key(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isalnum())


class EchoUserSimConfig(vf.TasksetConfig):
    phrases: list[str] = PHRASES


class EchoUserSimData(vf.TaskData):
    phrases: list[str]


class EchoUserSimTask(vf.Task[EchoUserSimData, vf.State, vf.TaskConfig]):
    @vf.reward(weight=1.0)
    async def echoed(self, trace: vf.Trace) -> float:
        replies = [m.content for m in trace.assistant_messages]
        phrases = self.data.phrases
        if len(replies) < len(phrases):
            return 0.0
        matched = sum(_key(p) in _key(r or "") for r, p in zip(replies, phrases))
        return matched / len(phrases)


class EchoUserSimEnv(vf.SingleAgentEnv):
    """Scripts the user side: opens with the first phrase, follows with the rest."""

    async def run(self, task, agents):
        # A chat session scripting the user: the task carries no prompt, so the
        # first turn opens the conversation.
        async with agents.agent.chat(task) as session:
            for phrase in task.data.phrases:
                if (await session.turn(phrase)).stopped:
                    break


class EchoUserSimTaskset(vf.Taskset[EchoUserSimTask, EchoUserSimConfig]):
    def load(self) -> list[EchoUserSimTask]:
        return [
            EchoUserSimTask(
                EchoUserSimData(
                    idx=0,
                    # No prompt: the scripted user opens the conversation.
                    prompt=None,
                    system_prompt=SYSTEM,
                    phrases=self.config.phrases,
                )
            )
        ]


__all__ = ["EchoUserSimTaskset", "EchoUserSimEnv"]
