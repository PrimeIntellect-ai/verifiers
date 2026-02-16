from abc import abstractmethod

import verifiers as vf
from verifiers.types import Messages, State, TrajectoryStep
from verifiers.utils.message_utils import concat_messages, normalize_messages


class MultiAgentEnv(vf.StatefulToolEnv):
    """
    Multi-agent environment on top of StatefulToolEnv.

    `state["trajectory_id"]` is the active actor id.
    """

    @abstractmethod
    def get_initial_actor_id(self, actors: dict[str, str], state: State) -> str:
        pass

    @abstractmethod
    def get_next_actor_id(self, state: State) -> str:
        pass

    @abstractmethod
    def get_all_actors(self, state: State) -> dict[str, str]:
        pass

    @abstractmethod
    def get_prompt_for_actor(self, messages: Messages, state: State) -> Messages:
        pass

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        actors = self.get_all_actors(state)
        initial_actor_id = self.get_initial_actor_id(actors, state)
        if initial_actor_id not in actors:
            raise ValueError(
                f"Initial actor ID '{initial_actor_id}' not found in actors"
            )
        state["trajectory_id"] = initial_actor_id
        state["tool_responses"] = {actor_id: None for actor_id in actors}
        state["system_prompts"] = actors
        self.logger.debug(
            "multiagent.setup actors=%s initial_actor=%s",
            list(actors.keys()),
            initial_actor_id,
        )
        return state

    def last_step_for_trajectory_id(
        self, state: State, trajectory_id: str
    ) -> TrajectoryStep | None:
        for step in reversed(state["trajectory"]):
            if step["trajectory_id"] == trajectory_id:
                return step
        return None

    def messages_for_trajectory_id(
        self, state: State, trajectory_id: str
    ) -> Messages | None:
        step = self.last_step_for_trajectory_id(state, trajectory_id)
        if step is None:
            return None
        step_prompt = normalize_messages(step["prompt"], field_name="trajectory.prompt")
        step_completion = normalize_messages(
            step["completion"], field_name="trajectory.completion"
        )
        return concat_messages([step_prompt, step_completion])

    async def get_prompt_messages(self, state: State) -> Messages:
        if len(state["trajectory"]) == 0:
            self.logger.debug(
                "multiagent.turn initial actor=%s",
                state["trajectory_id"],
            )
            return normalize_messages(
                self.get_prompt_for_actor([], state),
                field_name="get_prompt_for_actor",
            )

        self.logger.debug(
            "multiagent.turn resume trajectory_len=%s last_actor=%s",
            len(state["trajectory"]),
            state["trajectory"][-1]["trajectory_id"],
        )
        prev_turn_prompt = normalize_messages(
            state["trajectory"][-1]["prompt"], field_name="trajectory.prompt"
        )
        prev_turn_completion = normalize_messages(
            state["trajectory"][-1]["completion"], field_name="trajectory.completion"
        )
        prev_messages = concat_messages([prev_turn_prompt, prev_turn_completion])
        env_response = await self.env_response(prev_messages, state)
        env_response_messages = normalize_messages(
            env_response, field_name="env_response"
        )
        prev_trajectory_id = state["trajectory"][-1]["trajectory_id"]
        state["tool_responses"][prev_trajectory_id] = env_response_messages
        state["trajectory"][-1]["extras"]["env_response"] = env_response_messages
        self.logger.debug(
            "multiagent.env_response stored actor=%s message_count=%s",
            prev_trajectory_id,
            len(env_response_messages),
        )
        state["trajectory_id"] = self.get_next_actor_id(state)
        self.logger.debug("multiagent.turn switched actor=%s", state["trajectory_id"])
        messages = self.messages_for_trajectory_id(state, state["trajectory_id"])
        if messages is None:
            self.logger.debug(
                "multiagent.prompt actor=%s history=none",
                state["trajectory_id"],
            )
            return normalize_messages(
                self.get_prompt_for_actor([], state),
                field_name="get_prompt_for_actor",
            )
        actor_messages = messages
        prompt_messages = normalize_messages(
            self.get_prompt_for_actor(actor_messages, state),
            field_name="get_prompt_for_actor",
        )
        tool_response = state["tool_responses"][state["trajectory_id"]]
        if tool_response is None:
            self.logger.warning(
                "multiagent.prompt missing_tool_response actor=%s",
                state["trajectory_id"],
            )
            raise ValueError(
                f"Missing tool response for actor '{state['trajectory_id']}'"
            )

        tool_response_messages = normalize_messages(
            tool_response, field_name="tool_responses"
        )
        self.logger.debug(
            "multiagent.prompt actor=%s history=present tool_response_count=%s",
            state["trajectory_id"],
            len(tool_response_messages),
        )
        return concat_messages(
            [actor_messages, tool_response_messages, prompt_messages]
        )

    async def render_completion(self, state: State):
        """Render latest prompt/completion pair per actor from newest to oldest."""
        if len(state["trajectory"]) == 0:
            state["completion"] = []
            return

        unique_steps: list[TrajectoryStep] = []
        seen_trajectory_ids: set[str] = set()
        for step in reversed(state["trajectory"]):
            trajectory_id = step["trajectory_id"]
            if trajectory_id in seen_trajectory_ids:
                continue
            seen_trajectory_ids.add(trajectory_id)
            unique_steps.append(step)

        completion: Messages = []
        for i, step in enumerate(unique_steps):
            step_prompt = normalize_messages(
                step["prompt"], field_name=f"trajectory[{i}].prompt"
            )
            step_completion = normalize_messages(
                step["completion"], field_name=f"trajectory[{i}].completion"
            )
            completion = concat_messages([completion, step_prompt, step_completion])

        if state.get("final_env_response") is not None:
            completion = concat_messages(
                [
                    completion,
                    normalize_messages(
                        state["final_env_response"], field_name="final_env_response"
                    ),
                ]
            )

        state["completion"] = completion
