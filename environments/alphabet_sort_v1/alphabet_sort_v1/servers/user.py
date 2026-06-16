import verifiers.v1 as vf


class AlphabetSortUser(vf.User[vf.UserConfig]):
    """Replays the episode's pre-generated user turns: one `respond` delivers the next queued
    turn as a user message, until the queue is exhausted (then `done`). The queue is the
    follow-ups; when the task opens from the user (`info["instruction"]` is set, i.e. the task
    carries no prompt), the initial sort prompt is delivered first as the opening turn."""

    async def setup_task(self, task) -> None:
        opening = task.info.get("instruction")
        follow_ups = task.info["follow_ups"]
        self.queue = [opening, *follow_ups] if opening is not None else list(follow_ups)
        self.i = 0

    async def respond(self, message: str) -> tuple[vf.Messages, bool]:
        if self.i >= len(self.queue):
            return [], True
        content = self.queue[self.i]
        self.i += 1
        return [{"role": "user", "content": content}], False


if __name__ == "__main__":
    AlphabetSortUser.run()
