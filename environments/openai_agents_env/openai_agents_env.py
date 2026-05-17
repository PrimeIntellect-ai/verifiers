import re

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset

ANSWER_RE = re.compile(r"^\s*ANSWER\s*:?\s*(.+?)\s*$", re.IGNORECASE)


class OpenAIAgentsTasksetConfig(vf.TasksetConfig):
    source: str = f"{__name__}:load_train_rows"
    eval_source: str = f"{__name__}:load_eval_rows"
    taskset_id: str = "gsm8k-openai-agents"
    rewards: list[vf.CallableConfig] = [
        vf.CallableConfig(fn=f"{__name__}:answer_reward")
    ]
    num_train_examples: int = 50
    num_eval_examples: int = 20


class OpenAIAgentsHarnessConfig(vf.HarnessConfig):
    program: vf.ProgramConfig = vf.ProgramConfig(
        fn=f"{__name__}:run_openai_agents_program"
    )


class OpenAIAgentsEnvConfig(vf.EnvConfig):
    taskset: OpenAIAgentsTasksetConfig = OpenAIAgentsTasksetConfig()
    harness: OpenAIAgentsHarnessConfig = OpenAIAgentsHarnessConfig()


def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


async def run_openai_agents_program(task: vf.Task, state: vf.State) -> vf.State:
    from agents import (
        Agent,
        OpenAIChatCompletionsModel,
        Runner,
        function_tool,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    set_tracing_disabled(True)
    endpoint_config = state.get_endpoint_config(api="chat")
    client = AsyncOpenAI(
        base_url=endpoint_config["api_base"],
        api_key=endpoint_config["api_key"],
    )
    model = OpenAIChatCompletionsModel(
        model=endpoint_config["model"],
        openai_client=client,
    )
    agent = Agent(
        name="MathSolver",
        instructions=(
            "You are a math problem solver. Use the calculate tool to evaluate "
            "expressions. Give your final numerical answer after the word ANSWER "
            "on its own line, e.g.:\nANSWER: 42"
        ),
        model=model,
        tools=[function_tool(calculate)],
    )

    question = task.get("question")
    if question is not None:
        query = str(question)
    else:
        query = ""
        prompt = task.get("prompt")
        if isinstance(prompt, list) and prompt:
            query = str(vf.get_messages(prompt)[-1].content or "")

    result = await Runner.run(agent, input=query)
    final_output = str(result.final_output)
    state["agent_result"] = final_output
    state["completion"] = [{"role": "assistant", "content": final_output}]
    return state


def load_rows(split: str, num_examples: int):
    n = num_examples if num_examples > 0 else None
    return load_example_dataset("gsm8k", split=split, n=n)


def load_train_rows(num_train_examples: int):
    return load_rows("train", num_train_examples)


def load_eval_rows(num_eval_examples: int):
    return load_rows("test", num_eval_examples)


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
        return 1.0 if agent_answer.strip() == answer.strip() else 0.0
    return 1.0 if abs(parsed_agent_answer - parsed_answer) < 0.01 else 0.0


def answer_reward(task: vf.Task, state: vf.State) -> float:
    """Check if the agent's final output contains the correct answer."""
    result = state.get("agent_result")
    if result is not None:
        text = str(result)
    else:
        completion = state.get("completion")
        messages = []
        if isinstance(completion, list):
            messages = vf.get_messages(completion, role="assistant") or vf.get_messages(
                completion
            )
        text = str(messages[-1].content or "") if messages else ""
    agent_answer = extract_answer(text)
    if not agent_answer:
        return 0.0
    return answers_match(agent_answer, str(task.get("answer", "")))


def load_taskset(
    config: OpenAIAgentsTasksetConfig = OpenAIAgentsTasksetConfig(),
) -> vf.Taskset:
    return vf.Taskset(config=config)


def load_harness(
    config: OpenAIAgentsHarnessConfig = OpenAIAgentsHarnessConfig(),
) -> vf.Harness:
    return vf.Harness(config=config)


def load_environment(
    config: OpenAIAgentsEnvConfig = OpenAIAgentsEnvConfig(),
) -> vf.Env:
    """Load the OpenAI Agents SDK V1 taskset/harness example environment."""
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
