from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.envs.experimental.modules.harnesses import EndpointHarness


class HelloEndpointHarness(EndpointHarness):
    async def execute(
        self,
        task: vf.Task,
        state: vf.State,
        resources: vf.Resources,
        client: AsyncOpenAI,
    ) -> object:
        response = await client.chat.completions.create(
            model=resources.model,
            messages=self.endpoint_message_payload(state["prompt"]),
            **resources.sampling_args,
        )
        message = response.choices[0].message.content or ""
        state["endpoint_answer"] = message
        return message


def load_taskset() -> vf.Taskset:
    parser = vf.XMLParser(fields=["answer"], answer_field="answer")

    def exact_answer(parser, completion, answer) -> float:
        response = parser.parse_answer(completion)
        return 1.0 if response == answer else 0.0

    def source() -> Dataset:
        return Dataset.from_list(
            [
                {
                    "question": (
                        "Reply with exactly "
                        "`<answer>hello endpoint</answer>` and no other text."
                    ),
                    "answer": "hello endpoint",
                }
            ]
        )

    return vf.Taskset(
        source=source,
        rubric=vf.Rubric(funcs=[exact_answer], weights=[1.0], parser=parser),
        name="hello-endpoint-th",
    )


def load_harness(
    system_prompt: str | None = "You are a precise endpoint-backed assistant.",
    max_turns: int = 4,
) -> vf.Harness:
    return HelloEndpointHarness(system_prompt=system_prompt, max_turns=max_turns)


def load_environment(
    system_prompt: str | None = "You are a precise endpoint-backed assistant.",
    max_turns: int = 4,
) -> vf.Environment:
    return vf.Env(
        taskset=load_taskset(),
        harness=load_harness(system_prompt=system_prompt, max_turns=max_turns),
    )
