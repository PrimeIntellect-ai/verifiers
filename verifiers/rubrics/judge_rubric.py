from typing import Any

from verifiers.clients import Client
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import ClientConfig, Messages, State, UserMessage

DEFAULT_JUDGE_PROMPT = """Given a ground truth answer \
and a response, determine if the response is correct.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only."""


class JudgeRubric(Rubric):
    def __init__(
        self,
        parser: Parser | None = None,
        parallelize_scoring: bool = False,
        judge_client: Client | None = None,
        judge_model: str = "gpt-4.1-nano",
        judge_sampling_args: dict[str, Any] | None = None,
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
    ):
        super().__init__(parser=parser)
        self.judge_client: Client = (
            judge_client
            if judge_client is not None
            else OpenAIChatCompletionsClient(ClientConfig())
        )
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.judge_sampling_args = judge_sampling_args or {}
        self.class_objects = {
            "parser": self.parser,
            "judge": self.judge,
            "judge_client": self.judge_client,
            "judge_model": self.judge_model,
            "judge_prompt": self.judge_prompt,
            "judge_sampling_args": self.judge_sampling_args,
        }

    async def judge(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State | None = None,
    ) -> str:
        if isinstance(prompt, list):
            last_msg = prompt[-1]
            if isinstance(last_msg, dict) and "content" in last_msg:
                question = str(last_msg["content"])
            else:
                question = ""
        else:
            question = str(prompt)
        response = self.parser.parse_answer(completion)
        judge_prompt = self.judge_prompt.format(
            question=question, answer=answer, response=response
        )
        cached = state.get("judge_response") if state else None
        if isinstance(cached, dict) and judge_prompt in cached:
            return cached[judge_prompt]
        # Normalize judge sampling args for chat API
        judge_args = dict(self.judge_sampling_args or {})
        if "max_tokens" in judge_args:
            if judge_args["max_tokens"] is None:
                judge_args.pop("max_tokens")
            else:
                judge_args["max_completion_tokens"] = judge_args.pop("max_tokens")
        if (
            "max_completion_tokens" in judge_args
            and judge_args["max_completion_tokens"] is None
        ):
            judge_args.pop("max_completion_tokens")
        judge_args = {k: v for k, v in judge_args.items() if v is not None}

        judge_messages: Messages = [UserMessage(content=judge_prompt)]
        response_obj = await self.judge_client.get_response(
            prompt=judge_messages,
            model=self.judge_model,
            sampling_args=judge_args,
            state=state,
        )
        judge_response = str(response_obj.message.content or "")

        if state:
            if not isinstance(cached, dict):
                cached = {}
            cached[judge_prompt] = judge_response
            state["judge_response"] = cached
        return judge_response
