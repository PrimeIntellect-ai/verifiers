"""echo (v1, multi-turn): echo a phrase per turn, driven by a `vf.User` simulator.

The v1 multi-turn fixture for the e2e matrix. The user simulator is a `vf.User` class: it runs
host-side (its own subprocess runtime, the default) and the framework drives it, so it's
reachable whatever runtime the harness runs in. A user-sim is a task tool, so this needs a
tool-supporting harness.
"""

import verifiers.v1 as vf

PHRASES = ["hello world", "goodbye world"]
SYSTEM = "Repeat the user's message back to them exactly, with no extra words."


def _key(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isalnum())


class EchoMultiTask(vf.Task):
    phrases: list[str]


class EchoMultiConfig(vf.TasksetConfig):
    phrases: list[str] = PHRASES
    user: vf.UserConfig = vf.UserConfig()


class EchoMultiUser(vf.User[vf.UserConfig]):
    """Injects the next phrase as a user turn until the episode's phrases run out."""

    async def setup(self, task) -> None:
        self.phrases = task.phrases  # per-task input, from the task
        self.turns = 0  # per-rollout mutable state

    async def respond(self, message: str) -> tuple[vf.Messages, bool]:
        self.turns += 1
        if self.turns >= len(self.phrases):
            return [], True
        return [{"role": "user", "content": self.phrases[self.turns]}], False


class EchoMultiTaskset(vf.Taskset[EchoMultiTask, EchoMultiConfig]):
    def load_tasks(self) -> list[EchoMultiTask]:
        return [
            EchoMultiTask(
                idx=0,
                instruction=self.config.phrases[0],
                system_prompt=SYSTEM,
                phrases=self.config.phrases,
            )
        ]

    def user(self, task: EchoMultiTask) -> vf.User:
        return EchoMultiUser(self.config.user)

    @vf.reward(weight=1.0)
    async def echoed(self, task: EchoMultiTask, trace: vf.Trace) -> float:
        replies = [m.content for m in trace.assistant_messages]
        phrases = task.phrases
        if len(replies) < len(phrases):
            return 0.0
        matched = sum(_key(p) in _key(r or "") for r, p in zip(replies, phrases))
        return matched / len(phrases)


def load_taskset(config: EchoMultiConfig) -> EchoMultiTaskset:
    return EchoMultiTaskset(config)
