import time

import verifiers.v1 as vf
from verifiers.types import Response, ResponseMessage


class ReplayHarness(vf.Harness[vf.HarnessConfig]):
    async def run_with_context(self, context: vf.Context) -> None:
        task = context.task
        state = context.state
        messages = replay_messages(task)
        assistant_indices = [
            index
            for index, message in enumerate(messages)
            if message.role == "assistant"
        ]
        if not assistant_indices:
            raise ValueError("task.messages has no assistant messages.")
        max_turns = self.config.max_turns
        max_turns_reached = max_turns > 0 and max_turns < len(assistant_indices)
        if max_turns > 0:
            assistant_indices = assistant_indices[:max_turns]
        created = int(time.time())
        final_turn = len(assistant_indices) - 1
        completion: vf.Messages = []
        for turn_index, message_index in enumerate(assistant_indices):
            message = messages[message_index]
            prompt = messages[:message_index]
            is_truncated = max_turns_reached and turn_index == final_turn
            response = replay_response(
                message=message,
                model=context.model or "replay",
                created=created,
                turn=turn_index,
                state_id=state.id,
                is_truncated=is_truncated,
            )
            await state.add_response_turn(prompt, response)
            completion.append(message)
        state.stop("max_turns" if max_turns_reached else "replayed_messages")


def replay_messages(task: vf.Task) -> vf.Messages:
    value = getattr(task, "messages", None)
    if not isinstance(value, list):
        raise TypeError("task.messages must be a list.")
    return vf.get_messages(value)


def replay_response(
    *,
    message: vf.Message,
    model: str,
    created: int,
    turn: int,
    state_id: str,
    is_truncated: bool,
) -> Response:
    message_data = message.model_dump(mode="json", exclude_none=True)
    message_data.setdefault(
        "finish_reason", "tool_calls" if message_data.get("tool_calls") else "stop"
    )
    message_data["is_truncated"] = is_truncated or bool(
        message_data.get("is_truncated", False)
    )
    return Response(
        id=f"replay-{state_id}-{turn}",
        created=created,
        model=model,
        usage=None,
        message=ResponseMessage.model_validate(message_data),
    )


def load_harness(config: vf.HarnessConfig) -> ReplayHarness:
    return ReplayHarness(config=config)
