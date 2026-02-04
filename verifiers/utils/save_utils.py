import json
import logging
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

from datasets import Dataset
from openai import AsyncOpenAI
from pydantic import BaseModel

from verifiers.types import (
    ChatCompletionToolParam,
    ClientConfig,
    ErrorInfo,
    GenerateMetadata,
    GenerateOutputs,
    RolloutOutput,
    SamplingArgs,
    State,
    TokenUsage,
)
from verifiers.utils.error_utils import ErrorChain
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls
from verifiers.utils.path_utils import get_results_path

logger = logging.getLogger(__name__)


def is_json_serializable(value: object) -> bool:
    """Check if a value is JSON-serializable without conversion.

    Returns True for JSON primitives, lists/dicts of primitives,
    Pydantic models, datetime/date, Path, and exceptions.
    """
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_json_serializable(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and is_json_serializable(v) for k, v in value.items()
        )
    # Types that make_serializable can handle
    if isinstance(value, (BaseModel, datetime, date, Path, BaseException)):
        return True
    return False


def make_serializable(value: object) -> str | int | float | bool | list | dict | None:
    """Convert value to JSON-serializable types for non-standard types.

    Example:
    >>> json.dumps(value, default=make_serializable)
    """
    if isinstance(value, BaseModel):
        return value.model_dump()
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif isinstance(value, Path):
        return value.as_posix()
    elif isinstance(value, (BaseException)):
        return repr(value)
    else:
        return str(value)


def extract_usage_tokens(response: object) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    def get_usage_value(usage_obj: object, key: str) -> int | float:
        if isinstance(usage_obj, dict):
            return cast(dict[str, Any], usage_obj).get(key, 0)
        return getattr(usage_obj, key, 0)

    prompt_tokens = get_usage_value(usage, "prompt_tokens")
    completion_tokens = get_usage_value(usage, "completion_tokens")
    if not prompt_tokens and not completion_tokens:
        prompt_tokens = get_usage_value(usage, "input_tokens")
        completion_tokens = get_usage_value(usage, "output_tokens")
    return int(prompt_tokens or 0), int(completion_tokens or 0)


def get_hf_hub_dataset_name(outputs: GenerateOutputs) -> str:
    """Auto-generates a dataset name."""
    metadata = outputs["metadata"]
    dataset_name = (
        metadata["env_id"]
        + "_"
        + metadata["model"].replace("/", "_")
        + "_n"
        + str(metadata["num_examples"])
        + "_r"
        + str(metadata["rollouts_per_example"])
    )
    return dataset_name


def state_to_output(state: State, state_columns: list[str] = []) -> RolloutOutput:
    """Convert a State to a serializable RolloutOutput.

    Args:
        state: The State object to convert.
        state_columns: Additional State fields to include. Values must be
            JSON-serializable or an error will be raised.

    Returns:
        A RolloutOutput dict with all standard fields plus state_columns.

    Raises:
        ValueError: If a state_columns value is not JSON-serializable.
    """
    output = RolloutOutput(
        example_id=state.get("example_id", 0),
        prompt=state.get("prompt"),
        completion=state.get("completion"),
        answer=state.get("answer", ""),
        task=state.get("task", "default"),
        info=state.get("info", {}),
        reward=state.get("reward", 0.0),
        error=state.get("error", None),
        timing=state.get("timing", {}),
        is_completed=state.get("is_completed", False),
        is_truncated=state.get("is_truncated", False),
        stop_condition=state.get("stop_condition", None),
        metrics=state.get("metrics", {}),
        oai_tools=state.get("oai_tools", None),
    )
    trajectory = state.get("trajectory", [])
    input_tokens = 0
    output_tokens = 0
    usage_seen = False
    for step in trajectory:
        response = step.get("response")
        if response is None:
            continue
        if getattr(response, "usage", None) is not None:
            usage_seen = True
        step_input_tokens, step_output_tokens = extract_usage_tokens(response)
        input_tokens += step_input_tokens
        output_tokens += step_output_tokens
    if usage_seen:
        output["token_usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    # sanitize messages (handle None for error cases)
    prompt = state.get("prompt")
    if prompt is not None:
        output["prompt"] = sanitize_tool_calls(messages_to_printable(prompt))
    completion = state.get("completion")
    if completion is not None:
        output["completion"] = sanitize_tool_calls(messages_to_printable(completion))
    # use repr for error
    if state.get("error") is not None:
        error_chain = ErrorChain(state.get("error"))
        output["error"] = ErrorInfo(
            error=type(state.get("error")).__name__,
            error_chain_repr=repr(error_chain),
            error_chain_str=str(error_chain),
        )
        output["error_chain"] = repr(error_chain)
        output["long_error_chain"] = str(error_chain)
    # only include optional fields if non-empty
    if "answer" in output and not output["answer"]:
        output.pop("answer")
    if "info" in output and not output["info"]:
        output.pop("info")
    # flatten metrics to top-level keys (backwards compatibility)
    state_metrics = state.get("metrics", {})
    for k, v in state_metrics.items():
        output[k] = v
    # add state columns (must be serializable)
    for col in state_columns:
        value = state.get(col)
        if not is_json_serializable(value):
            raise ValueError(
                f"state_columns value for '{col}' is not JSON-serializable: "
                f"{type(value).__name__}. Only JSON-serializable types are allowed."
            )
        output[col] = value

    return output


def states_to_outputs(
    states: list[State], state_columns: list[str] = []
) -> list[RolloutOutput]:
    """Convert a list of States to serializable RolloutOutputs."""
    return [state_to_output(state, state_columns) for state in states]


class GenerateOutputsBuilder:
    """Incrementally builds GenerateOutputs, serializing states exactly once.

    This builder accumulates outputs as rollouts complete, avoiding redundant
    serialization for intermediate saves. Each state is serialized once via
    add_states(), and the accumulated outputs can be retrieved at any time
    via build().
    """

    def __init__(
        self,
        env_id: str,
        env_args: dict,
        model: str,
        client: AsyncOpenAI | ClientConfig,
        state_columns: list[str] | None,
        sampling_args: SamplingArgs,
        results_path: Path | None,
    ):
        self.env_id = env_id
        self.env_args = env_args
        self.model = model
        self.client = client
        self.state_columns = state_columns or []
        self.sampling_args = sampling_args
        self.results_path = results_path or get_results_path(env_id, model)
        self.start_time = time.time()

        # Accumulated outputs
        self.outputs: list[RolloutOutput] = []
        # Track tools from states for metadata
        self._tools_list: list[list[ChatCompletionToolParam] | None] = []

    def add_outputs(self, new_outputs: list[RolloutOutput]) -> list[RolloutOutput]:
        """Convert new outputs to outputs and accumulate. Returns the new outputs."""
        self.outputs.extend(new_outputs)
        # Track tools for metadata computation
        for output in new_outputs:
            self._tools_list.append(output.get("oai_tools"))
        return new_outputs

    def build(self, sort_by_example_id: bool = False) -> GenerateOutputs:
        """Build GenerateOutputs from accumulated outputs."""
        outputs = self.outputs
        if sort_by_example_id:
            outputs = sorted(outputs, key=lambda o: o.get("example_id", 0))
        metadata = self._compute_metadata(outputs)
        return GenerateOutputs(outputs=outputs, metadata=metadata)

    def _compute_metadata(self, outputs: list[RolloutOutput]) -> GenerateMetadata:
        """Compute metadata from accumulated outputs."""
        if isinstance(self.client, ClientConfig):
            base_url = self.client.api_base_url
        else:
            base_url = (
                str(self.client.base_url) if hasattr(self.client, "base_url") else ""
            )
        # Compute reward stats from outputs
        rewards = [o.get("reward", 0.0) for o in outputs]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Compute metrics stats from outputs
        metrics: dict[str, list[float]] = defaultdict(list)
        for output in outputs:
            output_metrics = output.get("metrics", {})
            if output_metrics:
                for metric_name, metric_value in output_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        metrics[metric_name].append(metric_value)
        avg_metrics = {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}

        input_tokens_total = 0.0
        output_tokens_total = 0.0
        usage_seen = False
        usage_count = 0
        for output in outputs:
            token_usage = output.get("token_usage")
            if not isinstance(token_usage, dict):
                continue
            usage_seen = True
            usage_count += 1
            input_tokens_total += float(token_usage.get("input_tokens", 0.0))
            output_tokens_total += float(token_usage.get("output_tokens", 0.0))
        usage: TokenUsage | None = None
        if usage_seen and usage_count > 0:
            usage = {
                "input_tokens": input_tokens_total / usage_count,
                "output_tokens": output_tokens_total / usage_count,
            }

        # Compute example counts
        example_ids = [o.get("example_id", 0) for o in outputs]
        num_examples = len(set(example_ids)) if example_ids else 0
        rollouts_per_example = len(outputs) // num_examples if num_examples > 0 else 1

        # Determine tools (use first non-None if all same)
        def tools_key(tools: list | None) -> str:
            if not tools:
                return ""
            return str(sorted([t.get("function", {}).get("name", "") for t in tools]))

        unique_tools = set(tools_key(t) for t in self._tools_list)
        tools = (
            next((t for t in self._tools_list if t), None)
            if len(unique_tools) == 1
            else None
        )

        return GenerateMetadata(
            env_id=self.env_id,
            env_args=self.env_args,
            model=self.model,
            base_url=base_url,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            sampling_args=self.sampling_args,
            date=datetime.now().isoformat(),
            time_ms=(time.time() - self.start_time) * 1000.0,
            avg_reward=avg_reward,
            avg_metrics=avg_metrics,
            usage=usage,
            state_columns=self.state_columns,
            path_to_save=self.results_path,
            tools=tools,
        )


def sanitize_metadata(metadata: GenerateMetadata) -> dict:
    """Sanitizes metadata before saving to disk."""

    metadata_dict = dict(metadata)
    metadata_dict.pop("path_to_save")
    metadata_dict.pop("date")

    return metadata_dict


def save_to_disk(outputs: list[RolloutOutput], metadata: dict, path: Path):
    """Saves rollout outputs and metadata to disk."""
    path.mkdir(parents=True, exist_ok=True)

    def save_outputs(outputs: list[RolloutOutput], results_path: Path):
        with open(results_path, "w") as f:
            for idx, output in enumerate(outputs):
                example_id = output.get("example_id") or "unknown"
                try:
                    json.dump(output, f, default=make_serializable)
                    f.write("\n")
                except Exception as e:
                    logger.error(
                        f"Failed to save result with index {idx} ({example_id=}): {e}"
                    )

    def save_metadata(metadata: dict, metadata_path: Path):
        with open(metadata_path, "w") as f:
            try:
                json.dump(metadata, f, default=make_serializable)
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")

    save_metadata(metadata, path / "metadata.json")
    save_outputs(outputs, path / "results.jsonl")


def make_dataset(results: GenerateOutputs) -> Dataset:
    """Create a Dataset from GenerateOutputs (outputs are already serialized)."""
    # RolloutOutput is a dict subclass, so this is type-safe at runtime
    return Dataset.from_list(list(results["outputs"]))


def save_generate_outputs(
    results: GenerateOutputs,
    push_to_hf_hub: bool = False,
    hf_hub_dataset_name: str | None = None,
):
    """Save GenerateOutputs to disk. Outputs are already serialized."""
    path_to_save = results["metadata"]["path_to_save"]
    sanitized_metadata = sanitize_metadata(results["metadata"])
    save_to_disk(results["outputs"], sanitized_metadata, path_to_save)
    logger.info(f"Results saved to {path_to_save}")
    if push_to_hf_hub:
        dataset_name = hf_hub_dataset_name or get_hf_hub_dataset_name(results)
        try:
            dataset = make_dataset(results)
            dataset.push_to_hub(dataset_name)
            logger.info(f"Dataset saved to Hugging Face Hub: {dataset_name}")
        except Exception as e:
            logger.error(f"Error pushing dataset to Hugging Face Hub: {e}")
