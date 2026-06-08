import verifiers.v1 as vf

from .config import UserConfig


class User(vf.User[UserConfig]):
    @vf.user(
        args={
            "info": "task.info",
            "transcript": "state.transcript",
        }
    )
    def respond(self, info: dict, transcript: list[dict]) -> dict:
        follow_ups = info.get("follow_ups") or []
        assistant_count = 0
        for turn in transcript:
            completion = turn.get("completion") or []
            assistant_count += sum(
                1 for message in completion if message.get("role") == "assistant"
            )
        if assistant_count <= 0 or assistant_count > len(follow_ups):
            return {"messages": []}
        return {
            "messages": [
                {"role": "user", "content": str(follow_ups[assistant_count - 1])}
            ],
        }
