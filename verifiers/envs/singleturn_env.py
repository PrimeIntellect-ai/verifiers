from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, State


class SingleTurnEnv(MultiTurnEnv):
    """
    Environment for single-turn tasks (chat or completion).
    """

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        return len(state["responses"]) > 0

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        # never called in MultiTurnEnv.rollout
        return [{"role": "user", "content": ""}], state
