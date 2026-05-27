import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, cast

import httpx
from gepa.core.adapter import EvaluationBatch
from openai import OpenAI

from verifiers.clients import Client
from verifiers.envs.environment import Environment
from verifiers.types import (
    ClientConfig,
    Messages,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
)
from verifiers.utils.client_utils import (
    build_headers_and_api_key,
    resolve_client_config,
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
    resolved_client_config = resolve_client_config(client_config)
    headers, api_key = build_headers_and_api_key(resolved_client_config)
    timeout = httpx.Timeout(
        resolved_client_config.timeout,
        connect=resolved_client_config.connect_timeout,
    )
    limits = httpx.Limits(
        max_connections=resolved_client_config.max_connections,
        max_keepalive_connections=resolved_client_config.max_keepalive_connections,
    )

    client = OpenAI(
        api_key=api_key or "EMPTY",
        base_url=resolved_client_config.api_base_url,
        max_retries=resolved_client_config.max_retries,
        http_client=httpx.Client(headers=headers, timeout=timeout, limits=limits),
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
    client: Client
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
                "completion": _completion_to_reflection_text(output),
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


MAX_REFLECTION_COMPLETION_CHARS = 12_000
MAX_REFLECTION_TOOL_EVENTS = 30


def _truncate_for_reflection(
    text: str,
    limit: int = MAX_REFLECTION_COMPLETION_CHARS,
) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n\n[truncated {len(text) - limit} chars]"


def _content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                part_map = cast(Mapping[str, Any], part)
                text = part_map.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                    continue
                part_type = part_map.get("type")
                if part_type in {"image_url", "input_audio", "audio"}:
                    chunks.append(f"[{part_type}]")
            else:
                chunks.append(str(part))
        return " ".join(chunks).strip()
    return str(content)


def _tool_call_summary(raw_tool_call: object) -> str:
    if isinstance(raw_tool_call, str):
        try:
            raw_tool_call = json.loads(raw_tool_call)
        except json.JSONDecodeError:
            return _truncate_for_reflection(raw_tool_call, limit=300)
    if not isinstance(raw_tool_call, Mapping):
        return _truncate_for_reflection(str(raw_tool_call), limit=300)

    raw_tool_call_map = cast(Mapping[str, Any], raw_tool_call)
    function = raw_tool_call_map.get("function")
    source: Mapping[str, Any]
    if isinstance(function, Mapping):
        source = cast(Mapping[str, Any], function)
    else:
        source = raw_tool_call_map

    name = str(source.get("name") or "tool")
    raw_args = source.get("arguments")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = raw_args
    else:
        args = raw_args

    if isinstance(args, Mapping):
        safe_args = {
            key: value
            for key, value in args.items()
            if key
            in {
                "file_path",
                "path",
                "pattern",
                "command",
                "old_text",
                "new_text",
            }
        }
        if "command" in safe_args:
            safe_args["command"] = _truncate_for_reflection(
                str(safe_args["command"]),
                limit=200,
            )
        if "old_text" in safe_args:
            safe_args["old_text"] = _truncate_for_reflection(
                str(safe_args["old_text"]),
                limit=120,
            )
        if "new_text" in safe_args:
            safe_args["new_text"] = _truncate_for_reflection(
                str(safe_args["new_text"]),
                limit=120,
            )
        args_text = json.dumps(safe_args, ensure_ascii=False)
    else:
        args_text = _truncate_for_reflection(str(args or ""), limit=300)

    return f"{name} {args_text}".strip()


def _completion_to_reflection_text(output: Mapping[str, Any]) -> str:
    completion = output.get("completion")
    if isinstance(completion, str):
        return _truncate_for_reflection(completion)
    if not isinstance(completion, Sequence):
        return _truncate_for_reflection(str(completion or ""))

    final_assistant_text = ""
    tool_events: list[str] = []
    for message in completion:
        printable = message_to_printable(message)
        if not isinstance(printable, Mapping):
            continue
        role = printable.get("role")
        if role == "assistant":
            content = _content_to_text(printable.get("content"))
            if content:
                final_assistant_text = content
            raw_tool_calls = printable.get("tool_calls")
            if isinstance(raw_tool_calls, Sequence) and not isinstance(
                raw_tool_calls, str | bytes
            ):
                for raw_tool_call in raw_tool_calls:
                    if len(tool_events) >= MAX_REFLECTION_TOOL_EVENTS:
                        break
                    tool_events.append(_tool_call_summary(raw_tool_call))

    parts: list[str] = []
    if final_assistant_text:
        parts.append(f"Final assistant message:\n{final_assistant_text}")
    if tool_events:
        omitted = ""
        if len(tool_events) >= MAX_REFLECTION_TOOL_EVENTS:
            omitted = "\n[additional tool calls omitted]"
        parts.append(
            "Tool-call summary:\n"
            + "\n".join(f"- {event}" for event in tool_events)
            + omitted
        )

    if not parts:
        return ""
    return _truncate_for_reflection("\n\n".join(parts))


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
