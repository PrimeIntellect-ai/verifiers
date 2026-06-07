import re

import verifiers.v1 as vf
from verifiers.utils.data_utils import load_example_dataset


class DSPYRLMTasksetConfig(vf.TasksetConfig):
    id: str = "gsm8k-dspy-rlm"
    num_train_examples: int = 50
    num_eval_examples: int = 20


class DSPYRLMHarnessConfig(vf.HarnessConfig):
    max_iterations: int = 10


class DSPYRLMTask(vf.Task):
    question: str
    answer: str


def load_gsm8k_tasks(split: str, num_examples: int) -> vf.Tasks:
    n = num_examples if num_examples > 0 else None
    return [
        {
            **row,
            "row_id": index,
            "prompt": [{"role": "user", "content": str(row["question"])}],
        }
        for index, row in enumerate(load_example_dataset("gsm8k", split=split, n=n))
    ]


def extract_dspy_answer(text: str) -> str:
    match = re.search(r"SUBMIT\((.+?)\)", text)
    if match:
        return match.group(1).strip().strip("'\"")

    match = re.search(
        r"\[\[\s*##\s*answer\s*##\s*\]\]\s*(.+?)(?:\n|$)", text, re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if line and not line.startswith("[[ ##"):
            return line
    return ""


def answers_match(agent_answer: str, answer: str) -> float:
    try:
        parsed_agent_answer = float(agent_answer.replace(",", ""))
        parsed_answer = float(answer.replace(",", ""))
    except (ValueError, TypeError):
        return float(agent_answer.strip() == answer.strip())
    return float(abs(parsed_agent_answer - parsed_answer) < 0.01)


def final_text(state: vf.State) -> str:
    result = state.artifacts.get("agent_result")
    if isinstance(result, str):
        return result
    messages = [
        message for message in state.completion if message.role == "assistant"
    ] or state.completion
    return str(messages[-1].content or "") if messages else ""


class DSPYRLMTaskset(vf.Taskset[DSPYRLMTasksetConfig]):
    task_type = DSPYRLMTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        dataset_split = "train" if split == "train" else "test"
        num_examples = (
            self.config.num_train_examples
            if split == "train"
            else self.config.num_eval_examples
        )
        return load_gsm8k_tasks(dataset_split, num_examples)

    @vf.reward
    async def answer_reward(self, task: DSPYRLMTask, state: vf.State) -> float:
        answer = extract_dspy_answer(final_text(state))
        return answers_match(answer, task.answer) if answer else 0.0


class DSPYRLMHarness(vf.Harness[DSPYRLMHarnessConfig]):
    async def run_with_context(self, context: vf.Context) -> None:
        task = DSPYRLMTask.model_validate(context.task.model_dump())
        state = context.state
        runtime = context.runtime
        if runtime is None:
            raise ValueError("DSPYRLMHarness requires a runtime.")
        prompt = self.initial_messages(task)

        async def stop_check() -> str | None:
            if await self.is_completed(context):
                return state.stop_condition or "stop"
            return None

        async with vf.InterceptionServer(
            context,
            task,
            state,
            protocols=self.protocols,
            stop_check=stop_check,
        ) as endpoint:
            endpoint_url = await runtime.expose(endpoint.port)
            endpoint_env = endpoint.env(base_url=endpoint_url, model=context.model)
            final_output = await run_dspy_rlm(
                query=task.question,
                base_url=endpoint_env["OPENAI_BASE_URL"],
                api_key=endpoint_env["OPENAI_API_KEY"],
                model=endpoint_env["OPENAI_MODEL"],
                max_iterations=self.config.max_iterations,
            )

        state.artifacts["agent_result"] = final_output
        message = vf.AssistantMessage(content=final_output)
        if not state.transcript:
            state.transcript.append(vf.Turn(prompt=prompt, completion=[message]))
        state.stop("dspy_completed")


async def run_dspy_rlm(
    *,
    query: str,
    base_url: str,
    api_key: str,
    model: str,
    max_iterations: int,
) -> str:
    import dspy

    lm = dspy.LM(
        f"openai/{model}",
        api_base=base_url,
        api_key=api_key,
        cache=False,
    )
    with dspy.context(lm=lm):
        rlm = dspy.RLM("query -> answer", max_iterations=max_iterations)
        result = await rlm.aforward(query=query)
    return str(result.answer)


def load_taskset(config: DSPYRLMTasksetConfig) -> DSPYRLMTaskset:
    return DSPYRLMTaskset(config=config)


def load_harness(config: DSPYRLMHarnessConfig) -> DSPYRLMHarness:
    return DSPYRLMHarness(config=config)
