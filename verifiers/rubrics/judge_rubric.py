from typing import Any

from openai import OpenAI

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, State

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
        judge_client: OpenAI | None = None,
        judge_model: str = "gpt-4.1-nano",
        judge_sampling_args: dict[str, Any] | None = None,
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
        **kwargs,
    ):
        super().__init__(
            parser=parser, parallelize_scoring=parallelize_scoring, **kwargs
        )
        self.judge_client = judge_client if judge_client is not None else OpenAI()
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.judge_sampling_args = judge_sampling_args or {}

    def judge(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        **kwargs,
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
        cached = state.get("judge_response")
        if isinstance(cached, dict) and judge_prompt in cached:
            return cached[judge_prompt]
        judge_response = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            **self.judge_sampling_args,
        )
        judge_response = str(judge_response.choices[0].message.content)
        if not isinstance(cached, dict):
            cached = {}
        cached[judge_prompt] = judge_response
        state["judge_response"] = cached
        return judge_response
