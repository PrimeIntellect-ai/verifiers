"""
Smoke-test environment for OpenCodeRLMEnv.

Uses a tiny inline dataset of simple tasks that exercise the RLM plugin's
subagent, subagent_batch, and llm-subcall capabilities.
"""

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.opencode_rlm_env import OpenCodeRLMEnv


TASKS = [
    {
        "prompt": "Create a file called /app/hello.txt containing exactly the text 'hello world'. Then verify it exists and print its contents.",
        "answer": "hello world",
    },
    {
        "prompt": "Use llm-subcall to ask: 'What is 2+2? Reply with just the number.' Write the response to /app/result.txt, then print the file.",
        "answer": "4",
    },
    {
        "prompt": "Use subagent to create a file /app/colors.txt with three colors, one per line. Then print the file.",
        "answer": "colors",
    },
]


def _build_dataset() -> Dataset:
    prompts = []
    answers = []
    for task in TASKS:
        prompts.append([{"role": "user", "content": task["prompt"]}])
        answers.append(task["answer"])
    return Dataset.from_dict({"prompt": prompts, "answer": answers})


async def _answer_in_output(completion: vf.Messages, answer: str, **kwargs) -> float:
    """Check if the expected answer string appears anywhere in the completion."""
    text = ""
    for msg in completion:
        content = getattr(msg, "content", None) or ""
        if isinstance(content, str):
            text += content + "\n"
    return 1.0 if answer.lower() in text.lower() else 0.0


def load_environment(
    num_examples: int = -1,
    docker_image: str = "python:3.11-slim",
    timeout_seconds: float = 600.0,
    max_turns: int = -1,
    **kwargs,
) -> vf.Environment:
    dataset = _build_dataset()
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    rubric = vf.Rubric(funcs=[_answer_in_output], weights=[1.0])

    return OpenCodeRLMEnv(
        dataset=dataset,
        docker_image=docker_image,
        timeout_seconds=timeout_seconds,
        max_turns=max_turns,
        rubric=rubric,
        **kwargs,
    )
