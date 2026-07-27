"""Multi-turn echo driven by a scripted user with imported conversation history.

The env's `run()` scripts the user through an interaction. The task is
prompt-less, so a seeded assistant greeting precedes the first `turn(phrase)`;
each later turn resumes the harness onto the accreted conversation, and leaving
the `async with` closes the interaction (the trace stops as `user_closed`).
"""

import verifiers.v1 as vf

PHRASES = ["hello world", "goodbye world"]
GREETING = "Hi! How can I help you today?"
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
        imported = [
            node
            for node in trace.nodes
            if not node.sampled
            and isinstance(node.message, vf.AssistantMessage)
            and node.message.content == GREETING
        ]
        if len(imported) != 1:
            return 0.0
        replies = [m.content for m in trace.assistant_messages]
        phrases = self.data.phrases
        if len(replies) < len(phrases):
            return 0.0
        matched = sum(_key(p) in _key(r or "") for r, p in zip(replies, phrases))
        return matched / len(phrases)


class EchoUserSimEnv(vf.SingleAgentEnv):
    """Scripts the user side: opens with the first phrase, follows with the rest."""

    async def run(self, task, agents):
        # Imported history is native context, while the first phrase remains the
        # caller's live opening turn.
        async with agents.agent.interaction(
            task,
            history=[
                vf.SystemMessage(content=SYSTEM),
                vf.AssistantMessage(content=GREETING),
            ],
        ) as interaction:
            for phrase in task.data.phrases:
                if (await interaction.turn(phrase)).terminated:
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


__all__ = ["EchoUserSimEnv", "EchoUserSimTaskset"]
