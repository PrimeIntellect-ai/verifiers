import time
from typing import cast

import verifiers as vf
from verifiers.types import TrajectoryStep


class ReplayHarnessConfig(vf.HarnessConfig):
    max_turns: int = 0


class ReplayHarness(vf.Harness[ReplayHarnessConfig]):
    async def base_program(self, task: vf.Task, state: vf.State) -> vf.State:
        await self.runtime.setup_rollout(task, state)
        messages = replay_messages(task)
        max_turns = state.get_max_turns(self.config.max_turns)
        assistant_indices = [
            index
            for index, message in enumerate(messages)
            if message.role == "assistant"
        ]
        if not assistant_indices:
            raise ValueError("task.messages has no assistant messages.")
        max_turns_reached = max_turns > 0 and max_turns < len(assistant_indices)
        if max_turns > 0:
            assistant_indices = assistant_indices[:max_turns]

        state["trajectory"] = []
        model = state.runtime_state().get("model")
        model_name = model if isinstance(model, str) and model else "replay"
        created = int(time.time())
        final_turn = len(assistant_indices) - 1
        for turn, message_index in enumerate(assistant_indices):
            message = messages[message_index]
            prompt = replay_messages_data(messages[:message_index])
            completion = [replay_message_data(message)]
            is_truncated = (max_turns_reached and turn == final_turn) or bool(
                completion[0].get("is_truncated", False)
            )
            response = replay_response(
                message=message,
                model=model_name,
                created=created,
                turn=turn,
                trajectory_id=str(state["trajectory_id"]),
                is_truncated=is_truncated,
            )
            state["trajectory"].append(
                replay_trajectory_step(
                    prompt=prompt,
                    completion=completion,
                    response=response,
                    trajectory_id=str(state["trajectory_id"]),
                    message_index=message_index,
                    is_truncated=is_truncated,
                )
            )
        if max_turns_reached:
            state._set_stop_condition("max_turns_reached")
        else:
            state._set_stop_condition("replayed_messages")
        return state


def replay_messages(task: vf.Task) -> vf.Messages:
    value = task.get("messages")
    if not isinstance(value, list):
        raise TypeError("task.messages must be a list.")
    return vf.get_messages(value)


def replay_message_data(message: vf.Message) -> vf.JsonData:
    return cast(vf.JsonData, message.model_dump(mode="json", exclude_none=True))


def replay_messages_data(messages: list[vf.Message]) -> list[vf.JsonData]:
    return [replay_message_data(message) for message in messages]


def replay_trajectory_step(
    *,
    prompt: list[vf.JsonData],
    completion: list[vf.JsonData],
    response: vf.JsonData,
    trajectory_id: str,
    message_index: int,
    is_truncated: bool,
) -> TrajectoryStep:
    return cast(
        TrajectoryStep,
        {
            "prompt": prompt,
            "completion": completion,
            "response": response,
            "tokens": None,
            "reward": None,
            "advantage": None,
            "is_truncated": is_truncated,
            "trajectory_id": trajectory_id,
            "extras": {"replay": True, "message_index": message_index},
        },
    )


def replay_response(
    *,
    message: vf.Message,
    model: str,
    created: int,
    turn: int,
    trajectory_id: str,
    is_truncated: bool,
) -> vf.JsonData:
    message_data = replay_message_data(message)
    message_data.setdefault(
        "finish_reason", "tool_calls" if message_data.get("tool_calls") else "stop"
    )
    message_data["is_truncated"] = is_truncated or bool(
        message_data.get("is_truncated", False)
    )
    return {
        "id": f"replay-{trajectory_id}-{turn}",
        "created": created,
        "model": model,
        "usage": None,
        "message": cast(vf.JsonData, message_data),
    }


def load_harness(config: ReplayHarnessConfig) -> ReplayHarness:
    return ReplayHarness(config=config)
