import re
from typing import cast

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset


class DSPYRLMTasksetConfig(vf.TasksetConfig):
    rewards: list[str] = ["answer_reward"]
    taskset_id: str = "gsm8k-dspy-rlm"
    num_train_examples: int = 50
    num_eval_examples: int = 20


async def run_dspy_rlm_program(task: vf.Task, state: vf.State) -> vf.State:
    import dspy
    from openai import OpenAI

    endpoint_config = state.get_endpoint_config(api="chat")
    endpoint_client = cast(OpenAI, state.get_client(api="chat", sync=True))
    endpoint_api_key = endpoint_client.api_key
    endpoint_client.close()
    lm = dspy.LM(
        f"openai/{endpoint_config.model}",
        api_base=endpoint_config.base_url,
        api_key=endpoint_api_key,
        cache=False,
    )

    with dspy.context(lm=lm):
        question = task.get("question")
        if question is not None:
            query = str(question)
        else:
            query = ""
            prompt = task.get("prompt")
            if isinstance(prompt, list) and prompt:
                query = str(vf.get_messages(prompt)[-1].content or "")
        rlm = dspy.RLM("query -> answer", max_iterations=10)
        result = await rlm.aforward(query=query)

    final_output = str(result.answer)
    state["agent_result"] = final_output
    state["completion"] = [{"role": "assistant", "content": final_output}]
    return state


def load_gsm8k_tasks(split: str, num_examples: int):
    n = num_examples if num_examples > 0 else None
    return load_example_dataset("gsm8k", split=split, n=n)


def load_tasks(
    split: vf.TaskSplit = "train",
    num_train_examples: int = 50,
    num_eval_examples: int = 20,
):
    dataset_split = "train" if split == "train" else "test"
    num_examples = num_train_examples if split == "train" else num_eval_examples
    return load_gsm8k_tasks(dataset_split, num_examples)


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
    agent_answer = extract_dspy_answer(text)
    if not agent_answer:
        return 0.0
    return answers_match(agent_answer, str(task.get("answer", "")))


class DSPYRLMTaskset(vf.Taskset[DSPYRLMTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(
            split=split,
            num_train_examples=self.config.num_train_examples,
            num_eval_examples=self.config.num_eval_examples,
        )


class DSPYRLMHarnessConfig(vf.HarnessConfig):
    program: vf.ProgramConfig = vf.ProgramConfig(fn="run_dspy_rlm_program")


class DSPYRLMEnvConfig(vf.EnvConfig):
    taskset: DSPYRLMTasksetConfig = DSPYRLMTasksetConfig()
    harness: DSPYRLMHarnessConfig = DSPYRLMHarnessConfig()


def load_environment(config: DSPYRLMEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=DSPYRLMTaskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )
