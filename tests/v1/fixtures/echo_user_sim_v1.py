"""Multi-turn echo driven by a scripted user — the single user-sim mechanism.

The env's `rollout()` supplies the user as a plain closure via `user=`. The task is
prompt-less, so the closure also opens the conversation (the interception's opening
ping), then sends each phrase as the next user turn, and ends the exchange by
returning no messages (the trace stops as `user_closed`).
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


class EchoUserSimEnv(vf.Environment):
    """Scripts the user side: opens with the first phrase, follows with the rest."""

    async def rollout(self, task, agents):
        phrases = task.data.phrases
        sent = 0

        async def script(message: str) -> vf.Messages:
            nonlocal sent
            if sent == len(phrases):
                return []  # out of phrases — end the exchange
            sent += 1
            return [vf.UserMessage(content=phrases[sent - 1])]

        return [await agents["main"].run(task, user=script)]


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
