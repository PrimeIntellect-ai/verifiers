import verifiers.v1 as vf


class AlphabetSortUser(vf.User[vf.UserConfig]):
    """Drives the whole conversation by replaying the episode's pre-generated user turns: each
    `respond` delivers the next queued turn as a user message, until the queue is exhausted (then
    `done`). The task carries no prompt, so the first turn (the opening `respond("")`) delivers
    the initial sort prompt; the rest are the follow-ups."""

    async def setup_task(self, task) -> None:
        self.queue = task.info["user_turns"]
        self.i = 0

    async def respond(self, message: str) -> vf.Messages:
        if self.i >= len(self.queue):
            self.state.done = True
            return []
        content = self.queue[self.i]
        self.i += 1
        return [{"role": "user", "content": content}]


if __name__ == "__main__":
    AlphabetSortUser.run()
