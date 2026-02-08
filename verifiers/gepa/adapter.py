import asyncio
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
from verifiers.utils.client_utils import resolve_client_config
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

    resolved_client_config = resolve_client_config(client_config)

    client = OpenAI(
        api_key=os.environ.get(resolved_client_config.api_key_var, ""),
        base_url=resolved_client_config.api_base_url,
        timeout=resolved_client_config.timeout,
        max_retries=resolved_client_config.max_retries,
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
        inputs = _inject_system_prompt(batch, candidate.get("system_prompt", ""))

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
            prompt_text = candidate.get("system_prompt", "")
            if prompt_text not in self._seen_prompts:
                self._seen_prompts[prompt_text] = len(self._seen_prompts)
            candidate_idx = self._seen_prompts[prompt_text]

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

        return {comp: records for comp in components_to_update}


def _inject_system_prompt(
    inputs: list[RolloutInput],
    system_prompt: str,
) -> list[RolloutInput]:
    """Inject or replace system prompt in each input's prompt."""
    if not system_prompt:
        return inputs

    modified = []
    for inp in inputs:
        inp_copy = dict(inp)
        prompt = inp_copy.get("prompt", [])

        if isinstance(prompt, str):
            inp_copy["prompt"] = f"{system_prompt}\n\n{prompt}"
        else:
            prompt = [dict(m) for m in prompt]
            if not prompt:
                # Empty prompt list - just add system message
                prompt = [{"role": "system", "content": system_prompt}]
            elif prompt[0].get("role") == "system":
                prompt[0] = {**prompt[0], "content": system_prompt}
            else:
                prompt = [{"role": "system", "content": system_prompt}] + prompt
            inp_copy["prompt"] = prompt

        modified.append(inp_copy)
    return modified


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
