import re

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset
from verifiers.v1.utils.prompt_utils import state_result_text, task_question_text


async def run_dspy_rlm_program(task: vf.Task, state: vf.State) -> vf.State:
    import dspy

    endpoint_config = state.get_endpoint_config(api="chat")
    lm = dspy.LM(
        f"openai/{endpoint_config['model']}",
        api_base=endpoint_config["api_base"],
        api_key=endpoint_config["api_key"],
        cache=False,
    )

    with dspy.context(lm=lm):
        rlm = dspy.RLM("query -> answer", max_iterations=10)
        result = await rlm.aforward(query=task_question_text(task))

    final_output = str(result.answer)
    state["agent_result"] = final_output
    state["completion"] = [{"role": "assistant", "content": final_output}]
    return state


def load_rows(split: str, num_examples: int):
    n = num_examples if num_examples > 0 else None
    return load_example_dataset("gsm8k", split=split, n=n)


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
    text = state_result_text(state)
    agent_answer = extract_dspy_answer(text)
    if not agent_answer:
        return 0.0
    return answers_match(agent_answer, str(task.get("answer", "")))


def load_taskset(
    num_train_examples: int = 50,
    num_eval_examples: int = 20,
    config: vf.TasksetConfig | None = None,
) -> vf.Taskset:
    return vf.Taskset(
        source=lambda: load_rows("train", num_train_examples),
        eval_source=lambda: load_rows("test", num_eval_examples),
        taskset_id="gsm8k-dspy-rlm",
        rewards=[answer_reward],
        config=config,
    )


def load_harness(config: vf.HarnessConfig | None = None) -> vf.Harness:
    return vf.Harness(program=run_dspy_rlm_program, config=config)


def load_environment(
    config: vf.EnvConfig,
    num_train_examples: int = 50,
    num_eval_examples: int = 20,
) -> vf.Env:
    """Load the DSPy RLM V1 taskset/harness example environment."""
    return vf.Env(
        taskset=load_taskset(
            num_train_examples=num_train_examples,
            num_eval_examples=num_eval_examples,
            config=config.taskset,
        ),
        harness=load_harness(config.harness),
    )
