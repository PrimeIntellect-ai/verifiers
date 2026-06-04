import time
from typing import cast

import verifiers as vf
from verifiers.types import TrajectoryStep


class ReplayHarness(vf.Harness[vf.HarnessConfig]):
    async def base_program(self, task: vf.Task, state: vf.State) -> vf.State:
        await self.runtime.setup_rollout(task, state)
        messages = replay_messages(task)
        max_turns = state.get_max_turns(self.config.max_turns)
        assistant_indices = [
            index
            for index, message in enumerate(messages)
            if message["role"] == "assistant"
        ]
        if not assistant_indices:
            raise ValueError("task.messages has no assistant messages.")
        if max_turns > 0:
            assistant_indices = assistant_indices[:max_turns]

        state["trajectory"] = []
        model = state.runtime_state().get("model")
        model_name = model if isinstance(model, str) and model else "replay"
        created = int(time.time())
        for turn, message_index in enumerate(assistant_indices):
            message = messages[message_index]
            prompt = messages[:message_index]
            completion = [message]
            response = replay_response(
                message=message,
                model=model_name,
                created=created,
                turn=turn,
                trajectory_id=str(state["trajectory_id"]),
            )
            state["trajectory"].append(
                replay_trajectory_step(
                    prompt=prompt,
                    completion=completion,
                    response=response,
                    trajectory_id=str(state["trajectory_id"]),
                    message_index=message_index,
                )
            )
        state._set_stop_condition("replayed_messages")
        return state


def replay_messages(task: vf.Task) -> list[vf.JsonData]:
    value = task.get("messages")
    if not isinstance(value, list):
        raise TypeError("task.messages must be a list.")

    messages: list[vf.JsonData] = []
    for message in value:
        if not isinstance(message, dict):
            raise TypeError("task.messages must contain JSON objects.")
        role = message.get("role")
        if not isinstance(role, str):
            raise TypeError("task.messages message role must be a string.")
        messages.append(cast(vf.JsonData, dict(message)))
    return messages


def replay_trajectory_step(
    *,
    prompt: list[vf.JsonData],
    completion: list[vf.JsonData],
    response: vf.JsonData,
    trajectory_id: str,
    message_index: int,
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
            "is_truncated": False,
            "trajectory_id": trajectory_id,
            "extras": {"replay": True, "message_index": message_index},
        },
    )


def replay_response(
    *,
    message: vf.JsonData,
    model: str,
    created: int,
    turn: int,
    trajectory_id: str,
) -> vf.JsonData:
    message_data = dict(message)
    message_data.setdefault(
        "finish_reason", "tool_calls" if message_data.get("tool_calls") else "stop"
    )
    message_data.setdefault("is_truncated", False)
    return {
        "id": f"replay-{trajectory_id}-{turn}",
        "created": created,
        "model": model,
        "usage": None,
        "message": cast(vf.JsonData, message_data),
    }


def load_harness(config: vf.HarnessConfig) -> ReplayHarness:
    return ReplayHarness(config=config)
