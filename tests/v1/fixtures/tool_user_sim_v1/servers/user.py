import verifiers.v1 as vf


class CalcState(vf.State):
    user_finished: bool = False


class CalcUser(vf.User[vf.UserConfig, CalcState]):
    """Poses the task's addition problems one per turn; after the last, flags `user_finished`
    (the taskset's `@vf.stop` ends the rollout on it). The task carries no prompt, so the opening
    `respond("")` delivers the first problem."""

    async def setup_task(self, task) -> None:
        self.problems = task.problems
        self.i = 0

    async def respond(self, message: str) -> vf.Messages:
        if self.i >= len(self.problems):
            self.state.user_finished = True
            return []
        a, b = self.problems[self.i]
        self.i += 1
        return [{"role": "user", "content": f"What is {a} + {b}?"}]


if __name__ == "__main__":
    CalcUser.run()
