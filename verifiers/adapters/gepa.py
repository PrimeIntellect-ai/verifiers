"""
GEPAAdapter: Bridge between Verifiers Environment API and GEPA optimization.

This adapter implements the GEPAAdapter protocol from the gepa package,
enabling automatic optimization of environment text components (system_prompt,
tool descriptions, etc.) through reflection-based evolution.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

from statistics import fmean
from gepa import EvaluationBatch, GEPAAdapter as BaseGEPAAdapter
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import Messages, RolloutInput

logger = logging.getLogger(__name__)


class GEPAAdapter(BaseGEPAAdapter):
    """
    Adapter bridging Verifiers Environment API to GEPA optimization.

    Key responsibilities:
    - Component management: Extract/inject text components (system_prompt, tool descriptions)
    - Evaluation: Run rollouts and collect scores
    - Feedback generation: Convert rubric scores + state to GEPA feedback
    - Dataset conversion: HF Dataset → GEPA format

    Args:
        env: Base Verifiers Environment to optimize
        client: AsyncOpenAI client for model inference
        model: Model name to optimize
        sampling_args: Sampling configuration (temperature, max_tokens, etc.)
        components_to_optimize: List of component names (e.g., ["system_prompt", "tool_descriptions"])
        num_rollouts_per_example: Number of rollouts per example for evaluation
        max_concurrent: Maximum concurrent rollout evaluations
    """

    def __init__(
        self,
        env: vf.Environment,
        client: AsyncOpenAI,
        model: str,
        sampling_args: dict[str, Any],
        components_to_optimize: list[str] | None = None,
        num_rollouts_per_example: int = 1,
        max_concurrent: int = 32,
    ):
        self.base_env = env
        self.client = client
        self.model = model
        self.sampling_args = sampling_args
        self.components_to_optimize = components_to_optimize or ["system_prompt"]
        self.num_rollouts_per_example = num_rollouts_per_example
        self.max_concurrent = max_concurrent
        self._candidate_build_count = 0  # Track candidate environment builds

        if self.num_rollouts_per_example < 1:
            raise ValueError("num_rollouts_per_example must be at least 1")
        if self.num_rollouts_per_example > 10:
            logger.warning(
                "num_rollouts_per_example=%s may be costly; "
                "expect roughly %sx more rollouts per batch",
                self.num_rollouts_per_example,
                self.num_rollouts_per_example,
            )

        # Validate components
        if "tool_descriptions" in self.components_to_optimize:
            if not hasattr(env, "oai_tools") or not env.oai_tools:
                raise ValueError(
                    "Cannot optimize tool_descriptions: environment has no tools"
                )

        for comp in self.components_to_optimize:
            if comp not in ["system_prompt", "tool_descriptions"]:
                if not hasattr(env, comp):
                    raise ValueError(
                        f"Environment does not have component '{comp}'. "
                        f"Available: system_prompt, tool_descriptions"
                    )

        logger.info(
            f"Initialized GEPAAdapter for {len(self.components_to_optimize)} components: "
            f"{self.components_to_optimize}"
        )

    def build_program(self, candidate: dict[str, str]) -> vf.Environment:
        """Create a candidate environment with updated components using shallow copy.

        Shallow copy shares heavy objects (dataset, rubric, parser) while
        allowing string attributes to be replaced. For oai_tools, we deep copy
        only if tool descriptions are being updated.
        """
        import copy

        self._candidate_build_count += 1
        logger.debug(
            f"Building candidate environment #{self._candidate_build_count} "
            f"with components: {list(candidate.keys())}"
        )

        # Create shallow copy - shares dataset, rubric, parser, etc.
        new_env = copy.copy(self.base_env)

        # Update system_prompt (assignment replaces reference, safe)
        if "system_prompt" in candidate:
            new_env.system_prompt = candidate["system_prompt"]

        # Update tool descriptions (need deep copy since we mutate nested dicts)
        if hasattr(self.base_env, "oai_tools") and self.base_env.oai_tools:
            tool_updates = {
                k: v
                for k, v in candidate.items()
                if k.startswith("tool_") and k.endswith("_description")
            }
            if tool_updates:
                new_env.oai_tools = copy.deepcopy(self.base_env.oai_tools)
                for i, tool in enumerate(new_env.oai_tools):
                    key = f"tool_{i}_description"
                    if key in tool_updates:
                        tool["function"]["description"] = tool_updates[key]

        logger.debug(
            f"Successfully built {new_env.__class__.__name__} candidate #{self._candidate_build_count}"
        )
        return new_env

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """
        Evaluate candidate on batch of examples.

        Args:
            batch: List of examples (dicts with 'question', 'answer', 'info', 'task')
            candidate: Dict of component values to evaluate
            capture_traces: Whether to capture detailed execution traces

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        # Build environment with candidate components
        env = self.build_program(candidate)

        logger.debug(
            f"Evaluating candidate on batch of {len(batch)} examples "
            f"({self.num_rollouts_per_example} rollouts/example = {len(batch) * self.num_rollouts_per_example} total rollouts)"
        )

        # Run evaluation using Environment's evaluate method
        evaluation = self._evaluate_async(env, batch, capture_traces)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - create one
            return asyncio.run(evaluation)

        # Already in an event loop - run in a thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, evaluation)
            return future.result()

    async def evaluate_async(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """
        Evaluate candidate asynchronously.

        Preferred when the caller already manages an asyncio loop (e.g., notebooks,
        services). Mirrors the synchronous evaluate() contract.
        """
        env = self.build_program(candidate)
        return await self._evaluate_async(env, batch, capture_traces)

    async def _evaluate_async(
        self, env: vf.Environment, batch: list[dict], capture_traces: bool
    ) -> EvaluationBatch:
        """Async helper for evaluation."""
        rollout_inputs = self._build_rollout_inputs(env, batch)
        if not rollout_inputs:
            raise ValueError(
                "Empty evaluation batch - no rollout inputs generated from batch"
            )

        generate_outputs = await env.generate(
            inputs=rollout_inputs,
            client=self.client,
            model=self.model,
            sampling_args=self.sampling_args,
            max_concurrent=self.max_concurrent,
            use_tqdm=False,
        )

        completions = generate_outputs["completion"]
        states = generate_outputs["state"]
        rewards = generate_outputs["reward"]

        if any(r is None for r in rewards):
            raise ValueError(
                "Received None reward from environment - check rubric configuration"
            )
        scores = [float(score) for score in rewards]
        trajectories = [] if capture_traces else None

        if capture_traces:
            for completion, state, score in zip(completions, states, scores):
                trajectories.append(
                    {
                        "completion": completion,
                        "state": state,
                        "score": score,
                    }
                )

        mean_score = fmean(scores) if scores else 0.0
        logger.debug(
            f"Evaluation complete: {len(scores)} rollouts, "
            f"mean={mean_score:.4f}, min={min(scores) if scores else 0:.4f}, "
            f"max={max(scores) if scores else 0:.4f}"
        )

        return EvaluationBatch(
            outputs=completions,
            scores=scores,
            trajectories=trajectories,
        )

    def _build_rollout_inputs(
        self, env: vf.Environment, batch: list[dict]
    ) -> list[RolloutInput]:
        """
        Convert GEPA batch examples into Verifiers RolloutInput objects.

        Handles prompt normalization, example/task bookkeeping, answer passthrough,
        and optional info payloads while duplicating entries according to
        num_rollouts_per_example so downstream generate() calls receive independent
        rollout inputs.
        """
        rollout_inputs: list[RolloutInput] = []

        for example_idx, example in enumerate(batch):
            raw_prompt = example.get("prompt") or example.get("question") or ""
            formatted_prompt = self._format_prompt(env, raw_prompt)
            task = str(example.get("task") or env.env_id or "default")

            example_id_value = example.get("example_id", example_idx)
            try:
                example_id = int(example_id_value)
            except (TypeError, ValueError):
                example_id = example_idx

            base_input: RolloutInput = {
                "prompt": formatted_prompt,
                "task": task,
                "example_id": example_id,
            }

            if "answer" in example and example["answer"] is not None:
                base_input["answer"] = example["answer"]

            info = example.get("info")
            if info is not None:
                base_input["info"] = deepcopy(info)

            for _ in range(self.num_rollouts_per_example):
                rollout_inputs.append(deepcopy(base_input))

        return rollout_inputs

    def _format_prompt(self, env: vf.Environment, prompt: str | Messages) -> Messages:
        """
        Ensure prompts match the environment's declared message_type.

        Completion environments expect raw strings, so chat-style prompts are
        flattened into a single string. Chat environments expect structured
        message lists, so bare strings are wrapped with system/few-shot context.
        """
        if env.message_type == "completion":
            if isinstance(prompt, str):
                return prompt
            if isinstance(prompt, list):
                content_parts: list[str] = []
                for message in prompt:
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            content_parts.append(content)
                return " ".join(content_parts) if content_parts else str(prompt)
            return str(prompt)

        if isinstance(prompt, list):
            return prompt

        messages: list[dict[str, str]] = []
        if env.system_prompt:
            messages.append({"role": "system", "content": env.system_prompt})
        if env.few_shot:
            messages.extend(deepcopy(env.few_shot))
        messages.append({"role": "user", "content": str(prompt)})
        return messages

    def _format_tool_calls_text(self, tool_calls: list[dict]) -> str:
        """Format tool calls as readable text for GEPA reflection."""
        parts = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args_str = func.get("arguments", "{}")
            parts.append(f"Tool Call: {name}({args_str})")
        return "\n".join(parts)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """
        Generate reflective dataset for GEPA's proposal phase.

        Each reflective example contains:
        - Inputs: Original prompt/task context
        - Generated_Outputs: Model completion
        - Feedback: Textual explanation of score

        Args:
            candidate: Current candidate being evaluated
            eval_batch: Results from evaluate()
            components_to_update: Which components to generate feedback for

        Returns:
            Dict mapping component_name → list[ReflectiveExample]
        """
        if not eval_batch.trajectories:
            raise ValueError(
                "make_reflective_dataset requires capture_traces=True in evaluate()"
            )

        reflective_data: dict[str, list[dict]] = {}
        _warned_no_get_feedback = False

        # For environment-level components (like system_prompt), all examples
        # reflect on the same component, so we aggregate feedback across examples
        for comp_name in components_to_update:
            # Check if component is in optimization list
            # Support both exact matches (e.g., "system_prompt") and group patterns
            # (e.g., "tool_0_description" matches "tool_descriptions")
            is_optimizable = comp_name in self.components_to_optimize

            # Check if this is a tool description (tool_N_description pattern)
            if (
                not is_optimizable
                and "tool_descriptions" in self.components_to_optimize
            ):
                # Match pattern: tool_0_description, tool_1_description, etc.
                if comp_name.startswith("tool_") and comp_name.endswith("_description"):
                    is_optimizable = True

            if not is_optimizable:
                logger.debug(
                    f"Skipping component '{comp_name}' - not in components_to_optimize: {self.components_to_optimize}"
                )
                continue

            examples = []

            for traj in eval_batch.trajectories:
                completion = traj["completion"]
                state = traj["state"]
                score = traj["score"]

                # Extract prompt for context
                prompt = state.get("prompt", "")
                if isinstance(prompt, list):
                    # Chat format - extract user message
                    user_msgs = [m for m in prompt if m.get("role") == "user"]
                    prompt_text = user_msgs[-1].get("content", "") if user_msgs else ""
                else:
                    prompt_text = prompt

                # Extract completion text - format entire conversation
                if isinstance(completion, list):
                    # Chat format - include all messages (assistant + tool responses)
                    completion_parts = []
                    for msg in completion:
                        role = msg.get("role", "")
                        content = msg.get("content", "")

                        if role == "assistant":
                            # Include content if present
                            if content:
                                completion_parts.append(f"Assistant: {content}")
                            # Include tool calls
                            tool_calls = msg.get("tool_calls", [])
                            if tool_calls:
                                completion_parts.append(
                                    self._format_tool_calls_text(tool_calls)
                                )
                        elif role == "tool":
                            # Include tool responses
                            completion_parts.append(f"Tool Result: {content}")

                    completion_text = (
                        "\n\n".join(completion_parts) if completion_parts else ""
                    )
                else:
                    completion_text = str(completion)

                # Build inputs dict
                inputs = {
                    "Task": prompt_text,
                }

                # Build outputs
                generated_outputs = completion_text

                # Generate feedback - use rubric's get_feedback if available
                if hasattr(self.base_env.rubric, "get_feedback"):
                    feedback = self.base_env.rubric.get_feedback(state)
                else:
                    # Default fallback for basic rubrics - warn once
                    if not _warned_no_get_feedback:
                        logger.warning(
                            "Rubric lacks get_feedback method - using generic feedback. "
                            "Consider implementing get_feedback for better GEPA reflection."
                        )
                        _warned_no_get_feedback = True
                    feedback = f"Reward: {score:.3f}"
                    if score < 0.5:
                        feedback += " (Low score - needs improvement)"
                    elif score >= 0.8:
                        feedback += " (Good performance)"

                examples.append(
                    {
                        "Inputs": inputs,
                        "Generated Outputs": generated_outputs,
                        "Feedback": feedback,
                    }
                )

            reflective_data[comp_name] = examples

        if not reflective_data:
            raise ValueError(
                f"No reflective data generated for components: {components_to_update}"
            )

        logger.info(
            f"Generated reflective dataset with {sum(len(v) for v in reflective_data.values())} examples "
            f"across {len(reflective_data)} components"
        )

        return reflective_data


__all__ = ["GEPAAdapter"]
