import re

import verifiers.v1 as vf
from verifiers.utils.data_utils import load_example_dataset

ANSWER_RE = re.compile(r"^\s*ANSWER\s*:?\s*(.+?)\s*$", re.IGNORECASE)


class OpenAIAgentsTasksetConfig(vf.TasksetConfig):
    id: str = "gsm8k-openai-agents"
    num_train_examples: int = 50
    num_eval_examples: int = 20


class OpenAIAgentsHarnessConfig(vf.HarnessConfig):
    max_turns: int = 10


class OpenAIAgentsTask(vf.Task):
    question: str
    answer: str


def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


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


def extract_answer(text: str) -> str:
    for line in reversed(text.splitlines()):
        match = ANSWER_RE.match(line)
        if match:
            return match.group(1).strip()
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


class OpenAIAgentsTaskset(vf.Taskset[OpenAIAgentsTasksetConfig]):
    task_type = OpenAIAgentsTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        dataset_split = "train" if split == "train" else "test"
        num_examples = (
            self.config.num_train_examples
            if split == "train"
            else self.config.num_eval_examples
        )
        return load_gsm8k_tasks(dataset_split, num_examples)

    @vf.reward
    async def answer_reward(self, task: OpenAIAgentsTask, state: vf.State) -> float:
        answer = extract_answer(final_text(state))
        return answers_match(answer, task.answer) if answer else 0.0


class OpenAIAgentsHarness(vf.Harness[OpenAIAgentsHarnessConfig]):
    async def run_with_context(self, context: vf.Context) -> None:
        task = OpenAIAgentsTask.model_validate(context.task.model_dump())
        state = context.state
        runtime = context.runtime
        if runtime is None:
            raise ValueError("OpenAIAgentsHarness requires a runtime.")
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
            final_output = await run_openai_agents(
                query=task.question,
                base_url=endpoint_env["OPENAI_BASE_URL"],
                api_key=endpoint_env["OPENAI_API_KEY"],
                model=endpoint_env["OPENAI_MODEL"],
            )

        state.artifacts["agent_result"] = final_output
        message = vf.AssistantMessage(content=final_output)
        if not state.transcript:
            state.transcript.append(vf.Turn(prompt=prompt, completion=[message]))
        state.stop("agent_completed")


async def run_openai_agents(
    *,
    query: str,
    base_url: str,
    api_key: str,
    model: str,
) -> str:
    from agents import (
        Agent,
        OpenAIChatCompletionsModel,
        Runner,
        function_tool,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    try:
        agent = Agent(
            name="MathSolver",
            instructions=(
                "You are a math problem solver. Use the calculate tool to evaluate "
                "expressions. Give your final numerical answer after the word ANSWER "
                "on its own line, e.g.:\nANSWER: 42"
            ),
            model=OpenAIChatCompletionsModel(model=model, openai_client=client),
            tools=[function_tool(calculate)],
        )
        result = await Runner.run(agent, input=query)
        return str(result.final_output)
    finally:
        await client.close()


def load_taskset(config: OpenAIAgentsTasksetConfig) -> OpenAIAgentsTaskset:
    return OpenAIAgentsTaskset(config=config)


def load_harness(config: OpenAIAgentsHarnessConfig) -> OpenAIAgentsHarness:
    return OpenAIAgentsHarness(config=config)
