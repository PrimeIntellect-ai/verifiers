from verifiers.types import Messages, SandboxConfig, State


class Task:
    def __init__(self, sandbox: SandboxConfig):
        pass

    async def setup(self):
        pass

    async def prompt(self):
        pass

    async def post_rollout(self):
        pass

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages | str:
        """Optional environment/user response hook for task-driven multi-turn flows."""
        return []


class TaskSet:
    def get_task(self) -> Task:
        raise NotImplementedError

    async def build_rubric(self) -> Rubric:
        pass
