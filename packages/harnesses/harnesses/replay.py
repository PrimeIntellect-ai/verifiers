import time

from pydantic import TypeAdapter
import verifiers.v1 as vf

_MESSAGES_ADAPTER = TypeAdapter(vf.Messages)


class ReplayHarness(vf.Harness[vf.HarnessConfig]):
    async def run_with_context(self, context: vf.Context) -> None:
        task = context.task
        state = context.state
        task_messages = getattr(task, "messages", None)
        if not isinstance(task_messages, list):
            raise TypeError("task.messages must be a list.")
        messages = _MESSAGES_ADAPTER.validate_python(task_messages)
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
        for turn_index, message_index in enumerate(assistant_indices):
            message = messages[message_index]
            prompt = messages[:message_index]
            is_truncated = max_turns_reached and turn_index == final_turn
            turn = vf.Turn(
                prompt=prompt,
                completion=[message],
                tool_calls=list(message.tool_calls or []),
                response_id=f"replay-{state.id}-{turn_index}",
                model=context.model or "replay",
                created=created,
                finish_reason=message.finish_reason
                or ("tool_calls" if message.tool_calls else "stop"),
                is_truncated=is_truncated or bool(message.is_truncated),
            )
            state.transcript.append(turn)
            if turn.is_truncated:
                state.is_truncated = True
        state.stop("max_turns" if max_turns_reached else "replayed_messages")


def load_harness(config: vf.HarnessConfig) -> ReplayHarness:
    return ReplayHarness(config=config)
