import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from gepa.core.adapter import EvaluationBatch
from openai import AsyncOpenAI, OpenAI

from verifiers.envs.environment import Environment
from verifiers.types import (
    ClientConfig,
    Messages,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
)
from verifiers.utils.message_utils import message_to_printable
from verifiers.utils.save_utils import make_serializable

if TYPE_CHECKING:
    from verifiers.gepa.display import GEPADisplay

logger = logging.getLogger(__name__)


def make_reflection_lm(
    client_config: ClientConfig,
    model: str,
    **kwargs: Any,
) -> Callable[[str], str]:
    """
    Create a synchronous reflection LM callable for GEPA.

    GEPA expects: reflection_lm(prompt: str) -> str
    """
    import os

    client = OpenAI(
        api_key=os.environ.get(client_config.api_key_var, ""),
        base_url=client_config.api_base_url,
        timeout=client_config.timeout,
        max_retries=client_config.max_retries,
    )

    def reflection_lm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response.choices[0].message.content
        return content or ""

    return reflection_lm


@dataclass
class VerifiersGEPAAdapter:
    """Bridges GEPA optimization loop with verifiers evaluation infrastructure."""

    env: Environment
    client: AsyncOpenAI
    model: str
    sampling_args: SamplingArgs | None = None
    max_concurrent: int = 32
    state_columns: list[str] = field(default_factory=list)

    # Optional display for progress updates
    display: "GEPADisplay | None" = None

    # GEPA adapter protocol: None means use default proposer with reflection_lm
    propose_new_texts: Callable[..., dict[str, str]] | None = None

    # Internal: track candidates by prompt hash
    _seen_prompts: dict[str, int] = field(default_factory=dict)

    def evaluate(
        self,
        batch: list[RolloutInput],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[RolloutOutput, RolloutOutput]:
        """
        Run verifiers evaluation with the candidate system prompt.
        """

        # Attach prompt components to info for envs that read them,
        # leaving prompt mutation to the environment.
        inputs = _attach_prompt_components(batch, candidate, self.env)

        def do_nothing(*args, **kwargs) -> None:
            pass

        results = asyncio.get_event_loop().run_until_complete(
            self.env.generate(
                inputs=inputs,
                client=self.client,
                model=self.model,
                sampling_args=self.sampling_args,
                max_concurrent=self.max_concurrent,
                state_columns=self.state_columns,
                on_start=do_nothing,
                on_progress=do_nothing,
            )
        )

        outputs = results["outputs"]
        example_ids = [o["example_id"] for o in outputs]
        rewards = [o["reward"] for o in outputs]

        # Update display if configured
        if self.display is not None:
            candidate_key = _candidate_key(candidate)
            if candidate_key not in self._seen_prompts:
                self._seen_prompts[candidate_key] = len(self._seen_prompts)
            candidate_idx = self._seen_prompts[candidate_key]

            self.display.update_eval(
                candidate_idx=candidate_idx,
                scores=rewards,
                example_ids=example_ids,
                capture_traces=capture_traces,
            )

        return EvaluationBatch(
            outputs=outputs,
            scores=rewards,
            trajectories=outputs if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],  # noqa: ARG002 - required by GEPA adapter protocol
        eval_batch: EvaluationBatch[RolloutOutput, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build reflective dataset for GEPA teacher LLM."""
        outputs = eval_batch.outputs
        trajectories = eval_batch.trajectories or []
        scores = eval_batch.scores

        records = []
        # outputs, trajectories, and scores should be the same length
        # Note: prompt/completion are already in printable format from state_to_output
        for output, trajectory, score in zip(outputs, trajectories, scores):
            record: dict[str, Any] = {
                "query": _extract_user_query(output["prompt"]),
                "completion": output["completion"],
                "expected_answer": output.get("answer", ""),
                "reward": score,
            }

            if trajectory.get("error"):
                record["error"] = trajectory["error"]

            if trajectory.get("stop_condition"):
                record["stop_condition"] = trajectory["stop_condition"]

            for col in self.state_columns:
                if col in trajectory:
                    record[col] = make_serializable(trajectory[col])

            records.append(record)

        # we might want this to become more sophisticated, only giving the relevant records for each component
        return {comp: records for comp in components_to_update}


def _attach_prompt_components(
    inputs: list[RolloutInput],
    candidate: dict[str, str],
    env: Environment,
) -> list[RolloutInput]:
    """Attach prompt components to info and update tool descriptions if provided."""
    if not candidate:
        return inputs

    tool_overrides = {
        key.split(":", 1)[1]: value
        for key, value in candidate.items()
        if key.startswith("tool:") and isinstance(value, str)
    }

    modified = []
    for inp in inputs:
        inp_copy = dict(inp)
        info = inp_copy.get("info")
        if not isinstance(info, dict):
            info = {}
        info = dict(info)
        info["prompt_components"] = dict(candidate)

        tool_source_env = _resolve_env_for_input(env, inp_copy)
        tool_source = getattr(tool_source_env, "oai_tools", None)
        if tool_overrides and tool_source:
            new_tools = []
            for tool in tool_source or []:
                tool_copy = dict(tool)
                func = tool_copy.get("function")
                if not isinstance(func, dict):
                    logger.warning("Skipping tool override: invalid tool function")
                    new_tools.append(tool_copy)
                    continue
                tool_name = func.get("name")
                if not tool_name:
                    logger.warning("Skipping tool override: missing tool name")
                    new_tools.append(tool_copy)
                    continue
                if tool_name in tool_overrides:
                    func_copy = dict(func)
                    func_copy["description"] = tool_overrides[tool_name]
                    tool_copy["function"] = func_copy
                new_tools.append(tool_copy)
            info["oai_tools"] = new_tools

        inp_copy["info"] = info
        modified.append(inp_copy)
    return modified


def _resolve_env_for_input(env: Environment, inp: RolloutInput) -> Environment:
    """Resolve per-task sub-environment for EnvGroup-like wrappers."""
    task = inp.get("task")
    if task is not None and hasattr(env, "get_env_for_task"):
        get_env_for_task = getattr(env, "get_env_for_task")
        if callable(get_env_for_task):
            try:
                resolved = get_env_for_task(task)
                if isinstance(resolved, Environment):
                    return resolved
            except Exception:
                pass
    return env


def _extract_user_query(prompt: Messages) -> str:
    """Extract user query from prompt, skipping system message."""
    if isinstance(prompt, str):
        return prompt
    for msg in prompt:
        if msg.get("role") == "user":
            content = message_to_printable(msg).get("content", "")
            if isinstance(content, str):
                return content
            return str(content) if content else ""
    return ""


def _candidate_key(candidate: dict[str, str]) -> str:
    """Stable key for multi-component candidates."""
    try:
        return json.dumps(candidate, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError):
        return str(sorted(candidate.items()))
