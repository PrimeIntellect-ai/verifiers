import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import GenerateOutputs
from verifiers.utils.message_utils import messages_to_printable

logger = logging.getLogger("verifiers.utils.eval_runner")


def serialize_messages_for_hub(messages: Any) -> Any:
    """Serialize messages for Hub upload"""
    if not isinstance(messages, list):
        return messages

    intermediate = []
    for msg in messages:
        new_msg = dict(msg)
        content = msg.get("content", "")

        if hasattr(content, "__iter__") and not isinstance(content, (str, bytes, dict)):
            try:
                new_msg["content"] = list(content)
            except (TypeError, AttributeError):
                pass

        intermediate.append(new_msg)

    printable = messages_to_printable(intermediate)

    serialized = []
    for msg in printable:
        new_msg = {"role": msg["role"], "content": str(msg.get("content", ""))}

        if "tool_calls" in msg:
            new_msg["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in msg.get("tool_calls", [])
            ]

        serialized.append(new_msg)

    return serialized


async def eval_environment_async(
    env: str,
    env_args: dict,
    client: AsyncOpenAI,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    sampling_args: dict | None,
) -> tuple[str, GenerateOutputs]:
    """Evaluate a single environment asynchronously"""
    logger.info(f"Loading environment: {env}")
    vf_env = vf.load_environment(env_id=env, **env_args)

    if vf_env.eval_dataset is None:
        logger.debug(f"No eval dataset for {env}, using train dataset")
        dataset = vf_env.get_dataset(n=num_examples)
    else:
        dataset = vf_env.get_eval_dataset(n=num_examples)

    assert dataset is not None
    if rollouts_per_example > 1:
        dataset = dataset.repeat(rollouts_per_example)

    logger.info(f"Evaluating {env} with {len(dataset)} samples...")

    results = await vf_env.a_generate(
        inputs=dataset,
        client=client,
        model=model,
        sampling_args=sampling_args,
        score_rollouts=True,
        max_concurrent=max_concurrent,
    )

    return env, results


async def eval_environments_parallel(
    envs: list[str],
    env_args_dict: dict[str, dict],
    client: AsyncOpenAI,
    model: str,
    num_examples: list[int],
    rollouts_per_example: list[int],
    max_concurrent: list[int],
    sampling_args: dict | None = None,
    sampling_args_dict: dict[str, dict] | None = None,
) -> dict[str, GenerateOutputs]:
    """Evaluate multiple environments in parallel"""
    tasks = []
    for env, n, r, c in zip(envs, num_examples, rollouts_per_example, max_concurrent):
        env_sampling_args = sampling_args
        if sampling_args_dict and env in sampling_args_dict:
            env_sampling_args = sampling_args_dict[env]

        tasks.append(
            eval_environment_async(
                env=env,
                env_args=env_args_dict.get(env, {}),
                client=client,
                model=model,
                num_examples=n,
                rollouts_per_example=r,
                max_concurrent=c,
                sampling_args=env_sampling_args,
            )
        )

    results = await asyncio.gather(*tasks)

    return dict(results)
