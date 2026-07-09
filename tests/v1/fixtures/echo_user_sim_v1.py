"""echo (v1, user simulator): echo a phrase per turn, driven by a `vf.User` simulator.

The v1 user-sim fixture for the e2e matrix. The user simulator is a `vf.User` class whose
placement is CLI-tunable (`--taskset.task.user.colocated`, `--taskset.task.user.runtime.type`): it runs
either inside the harness's runtime (`colocated`) or in its own runtime (the default), and
either way the framework drives it and reaches it from wherever the harness runs. A user-sim is
a task tool, so this needs a tool-supporting harness.
"""

import verifiers.v1 as vf

PHRASES = ["hello world", "goodbye world"]
SYSTEM = "Repeat the user's message back to them exactly, with no extra words."


def _key(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isalnum())


class EchoUserSimTaskConfig(vf.TaskConfig):
    user: vf.UserConfig = vf.UserConfig()


class EchoUserSimConfig(vf.TasksetConfig):
    phrases: list[str] = PHRASES
    task: EchoUserSimTaskConfig = EchoUserSimTaskConfig()


class EchoUserSimState(vf.State):
    user_finished: bool = False


class EchoUserSimUser(vf.User[vf.UserConfig, EchoUserSimState]):
    """Injects the next phrase as a user turn until the episode's phrases run out."""

    async def setup_task(self, task) -> None:
        self.phrases = task.phrases  # per-task input, from the task
        self.turns = 0  # per-rollout mutable state

    async def respond(self, message: str) -> vf.Messages:
        self.turns += 1
        if self.turns >= len(self.phrases):
            self.state.user_finished = True
            return []
        return [{"role": "user", "content": self.phrases[self.turns]}]


class EchoUserSimTask(vf.Task[EchoUserSimState, EchoUserSimTaskConfig]):
    phrases: list[str]
    user = EchoUserSimUser
    # Built with the task config's `user` field (placement stays CLI-tunable via
    # --taskset.task.user.*), resolved by `Task.server_config`.

    @vf.stop
    async def user_finished(self, trace: vf.Trace) -> bool:
        return trace.state.user_finished

    @vf.reward(weight=1.0)
    async def echoed(self, trace: vf.Trace) -> float:
        replies = [m.content for m in trace.assistant_messages]
        phrases = self.phrases
        if len(replies) < len(phrases):
            return 0.0
        matched = sum(_key(p) in _key(r or "") for r, p in zip(replies, phrases))
        return matched / len(phrases)


class EchoUserSimTaskset(vf.Taskset[EchoUserSimTask, EchoUserSimConfig]):
    def load(self) -> list[EchoUserSimTask]:
        return [
            EchoUserSimTask(
                idx=0,
                prompt=self.config.phrases[0],
                system_prompt=SYSTEM,
                phrases=self.config.phrases,
            )
        ]


__all__ = ["EchoUserSimTaskset"]


if __name__ == "__main__":
    EchoUserSimUser.run()
