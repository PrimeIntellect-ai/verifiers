import verifiers as vf


class SingleTurnEnv(vf.MultiTurnEnv):
    """
    Environment for single-turn tasks (chat or completion).
    """

    def __init__(self, **kwargs):
        super().__init__(max_turns=1, **kwargs)

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        raise NotImplementedError("env_response is not implemented for SingleTurnEnv")

    async def render_completion(self, state: vf.State):
        await super().render_completion(state)
        if self.message_type != "completion":
            return

        completion = state.get("completion")
        if not isinstance(completion, list):
            return

        chunks: list[str] = []
        for message in completion:
            content = (
                message.get("content", "")
                if isinstance(message, dict)
                else getattr(message, "content", "")
            )
            if isinstance(content, str):
                chunks.append(content)
                continue
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text")
                        if isinstance(text, str):
                            chunks.append(text)
                    elif getattr(part, "type", None) == "text":
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            chunks.append(text)

        state["completion"] = "".join(chunks)
