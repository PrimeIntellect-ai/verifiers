import verifiers.v1 as vf


class AlphabetSortUser(vf.User[vf.UserConfig]):
    """Replays the episode's pre-generated follow-up turns: one `respond` per assistant turn,
    injecting the next follow-up as a user message until all turns are done."""

    async def setup_task(self, task) -> None:
        self.follow_ups = task.info["follow_ups"]
        self.num_turns = task.info["num_turns"]
        self.turns = 0

    async def respond(self, message: str) -> tuple[vf.Messages, bool]:
        self.turns += 1
        if self.turns >= self.num_turns:
            return [], True
        return [{"role": "user", "content": self.follow_ups[self.turns - 1]}], False


if __name__ == "__main__":
    AlphabetSortUser.run()
