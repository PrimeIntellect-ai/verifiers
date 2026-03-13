from verifiers.types import Client, Messages, Response, SamplingArgs, State, Tool


class Harness:
    def __init__(self, tools: list[str] = [], system_prompt: str = ""):
        pass

    def setup(self):
        pass

    def run(self):
        pass

    async def get_model_response(
        self,
        env: TaskAgentEnv,
        state: State,
        prompt: Messages | str,
        client: Client | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> Response:
        return await env.request_model_response(
            state=state,
            prompt=prompt,
            client=client,
            model=model,
            tool_defs=tool_defs,
            sampling_args=sampling_args,
        )
