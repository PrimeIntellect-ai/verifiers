import json
import logging
import time
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

from datasets import Dataset
from openai import AsyncOpenAI
from pydantic import BaseModel

from verifiers.types import (
    ClientConfig,
    ErrorInfo,
    GenerateMetadata,
    GenerateOutputs,
    RolloutOutput,
    SamplingArgs,
    State,
    TokenUsage,
    Tool,
)
from verifiers.utils.error_utils import ErrorChain
from verifiers.utils.message_utils import (
    sanitize_tool_calls,
    serialize_messages_for_output,
)
from verifiers.utils.metric_utils import (
    EnvMetrics,
    ErrorRateMetric,
    InputTokensMetric,
    OutputTokensMetric,
    PassAtKMetric,
    RewardMetric,
)
from verifiers.utils.path_utils import get_results_path
from verifiers.utils.usage_utils import (
    StateUsageTracker,
)
from verifiers.utils.usage_utils import (
    extract_usage_tokens as extract_usage_tokens_from_response,
)
from verifiers.utils.version_utils import get_version_info

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
    if isinstance(value, Mapping):
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
        return value.model_dump(exclude_none=True)
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif isinstance(value, Path):
        return value.as_posix()
    elif isinstance(value, (BaseException)):
        return repr(value)
    elif isinstance(value, Mapping):
        return dict(value)
    else:
        return str(value)


def extract_usage_tokens(response: object) -> tuple[int, int]:
    return extract_usage_tokens_from_response(response)


def _coerce_token_usage(value: object) -> TokenUsage | None:
    if not isinstance(value, Mapping):
        return None
    mapping_value = cast(Mapping[str, Any], value)
    try:
        input_raw = mapping_value.get("input_tokens")
        output_raw = mapping_value.get("output_tokens")
        input_tokens = float(0.0 if input_raw is None else input_raw)
        output_tokens = float(0.0 if output_raw is None else output_raw)
    except (TypeError, ValueError):
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _extract_state_token_usage(state: State) -> TokenUsage | None:
    tracker = state.get("usage_tracker")
    if isinstance(tracker, StateUsageTracker):
        usage = tracker.snapshot()
        coerced = _coerce_token_usage(usage)
        if coerced is not None:
            return coerced
        # Tracker exists but has not seen usage yet. Avoid falling through to
        # state["usage"], which is a zeroed live tracker view.
        token_usage = _coerce_token_usage(state.get("token_usage"))
        if token_usage is not None:
            return token_usage
        return None

    for key in ("token_usage", "usage"):
        usage = _coerce_token_usage(state.get(key))
        if usage is not None:
            return usage

    return None


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


def state_to_output(
    state: State, state_columns: list[str] | None = None
) -> RolloutOutput:
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
        tool_defs=state.get("tool_defs"),
    )
    usage = _extract_state_token_usage(state)
    if usage is None:
        # Legacy fallback for states that do not use state-level usage tracking.
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
            usage = {
                "input_tokens": float(input_tokens),
                "output_tokens": float(output_tokens),
            }
    if usage is not None:
        output["token_usage"] = usage

    # sanitize messages (handle None for error cases)
    prompt = state.get("prompt")
    if prompt is not None:
        output_prompt = sanitize_tool_calls(serialize_messages_for_output(prompt))
        output["prompt"] = output_prompt
    completion = state.get("completion")
    if completion is not None:
        output_completion = sanitize_tool_calls(
            serialize_messages_for_output(completion)
        )
        output["completion"] = output_completion
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
    state_metrics = state.get("metrics") or {}
    for k, v in state_metrics.items():
        output[k] = v
    # add state columns (must be serializable)
    for col in state_columns or []:
        value = state.get(col)
        if not is_json_serializable(value):
            raise ValueError(
                f"state_columns value for '{col}' is not JSON-serializable: "
                f"{type(value).__name__}. Only JSON-serializable types are allowed."
            )
        output[col] = value

    return output


def states_to_outputs(
    states: list[State], state_columns: list[str] | None = None
) -> list[RolloutOutput]:
    """Convert a list of States to serializable RolloutOutputs."""
    return [state_to_output(state, state_columns) for state in states]


class GenerateOutputsBuilder:
    """Incrementally builds GenerateOutputs."""

    def __init__(
        self,
        env_id: str,
        env_args: dict,
        model: str,
        client: AsyncOpenAI | ClientConfig | object,
        num_examples: int,
        rollouts_per_example: int,
        state_columns: list[str] | None,
        sampling_args: SamplingArgs,
        results_path: Path | None,
        pass_threshold: float = 0.5,
    ):
        self.env_id = env_id
        self.env_args = env_args
        self.model = model
        self.client = client
        self.num_examples = num_examples
        self.rollouts_per_example = rollouts_per_example
        self.state_columns = state_columns or []
        self.sampling_args = sampling_args
        self.results_path = results_path or get_results_path(env_id, model)
        self.pass_threshold = pass_threshold
        self.start_time = time.time()
        self.base_url = self.compute_base_url(self.client)
        self.version_info = get_version_info(env_id=env_id)

        # Accumulated outputs
        self.outputs: list[RolloutOutput] = []

        # Incremental metric accumulators (avoid O(n) rescan in build_metadata)
        self.reward = RewardMetric()
        self.error_rate = ErrorRateMetric()
        self.env_metrics = EnvMetrics()
        self.input_tokens = InputTokensMetric()
        self.output_tokens = OutputTokensMetric()
        self.pass_at_k = PassAtKMetric(rollouts_per_example, threshold=pass_threshold)

        # Tools tracking
        self.unique_tools_keys: set[str] = set()
        self.first_tools: list[Tool] | None = None

    @staticmethod
    def format_base_url(url: str) -> str:
        return url

    def compute_base_url(self, client: AsyncOpenAI | ClientConfig | object) -> str:
        if isinstance(client, ClientConfig):
            if client.endpoint_configs:
                endpoint_urls = [cfg.api_base_url for cfg in client.endpoint_configs]
                if endpoint_urls:
                    return ",".join(endpoint_urls)
            return self.format_base_url(client.api_base_url)

        if hasattr(client, "base_url"):
            return str(getattr(client, "base_url"))
        return ""

    @staticmethod
    def tools_key(tools: list[Tool] | None) -> str:
        if not tools:
            return ""

        def _tool_name(tool: Tool | dict[str, Any]) -> str:
            if isinstance(tool, dict):
                function = tool.get("function")
                if isinstance(function, dict):
                    name = function.get("name")
                    if isinstance(name, str):
                        return name
                name = tool.get("name")
                return name if isinstance(name, str) else ""
            return tool.name

        return str(sorted(_tool_name(t) for t in tools))

    def add_outputs(self, new_outputs: list[RolloutOutput]) -> None:
        """Accumulate new outputs and update incremental accumulators."""
        self.outputs.extend(new_outputs)
        self.reward.add_outputs(new_outputs)
        self.error_rate.add_outputs(new_outputs)
        self.env_metrics.add_outputs(new_outputs)
        self.input_tokens.add_outputs(new_outputs)
        self.output_tokens.add_outputs(new_outputs)
        self.pass_at_k.add_outputs(new_outputs)

        for output in new_outputs:
            tool_defs = output.get("tool_defs")

            # Tools tracking
            tk = self.tools_key(tool_defs)
            self.unique_tools_keys.add(tk)
            if self.first_tools is None and tool_defs:
                self.first_tools = tool_defs

    def build_metadata(self) -> GenerateMetadata:
        """Build metadata from incremental accumulators. O(1) per call."""
        pass_at_k_result, pass_all_k_result = self.pass_at_k.compute()
        tools = self.first_tools if len(self.unique_tools_keys) == 1 else None

        usage: TokenUsage | None = None
        if self.input_tokens.count > 0:
            usage = {
                "input_tokens": self.input_tokens.compute(),
                "output_tokens": self.output_tokens.compute(),
            }

        return GenerateMetadata(
            env_id=self.env_id,
            env_args=self.env_args,
            model=self.model,
            base_url=self.base_url,
            num_examples=self.num_examples,
            rollouts_per_example=self.rollouts_per_example,
            sampling_args=self.sampling_args,
            date=datetime.now().isoformat(),
            time_ms=(time.time() - self.start_time) * 1000.0,
            avg_reward=self.reward.compute(),
            avg_metrics=self.env_metrics.compute(),
            avg_error=self.error_rate.compute(),
            pass_at_k=pass_at_k_result,
            pass_all_k=pass_all_k_result,
            pass_threshold=self.pass_threshold,
            usage=usage,
            version_info=self.version_info,
            state_columns=self.state_columns,
            path_to_save=self.results_path,
            tools=tools,
        )

    def build_outputs(self, sort_by_example_id: bool = False) -> list[RolloutOutput]:
        """Return (sorted) accumulated outputs"""
        if sort_by_example_id:
            return sorted(self.outputs, key=lambda o: o.get("example_id", 0))
        return self.outputs

    def build(self, sort_by_example_id: bool = False) -> GenerateOutputs:
        """Build GenerateOutputs from accumulated outputs."""
        return GenerateOutputs(
            outputs=self.build_outputs(sort_by_example_id),
            metadata=self.build_metadata(),
        )


def load_outputs(results_path: Path) -> list[RolloutOutput]:
    """Load outputs from disk."""
    outputs_path = results_path / "results.jsonl"
    outputs: list[RolloutOutput] = []

    with open(outputs_path, "r") as f:
        lines = f.readlines()

    for line_idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            outputs.append(RolloutOutput(**json.loads(line)))
        except json.JSONDecodeError:
            # A crash during append can leave the final JSONL line partially written.
            # Recover completed records, but keep raising for malformed non-trailing rows.
            has_nonempty_lines_after = any(
                remaining.strip() for remaining in lines[line_idx:]
            )
            if has_nonempty_lines_after:
                raise

            logger.warning(
                f"Ignoring malformed trailing line in {outputs_path} at line {line_idx}"
            )
            break

    return outputs


def validate_resume_metadata(
    results_path: Path,
    env_id: str,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
) -> None:
    """Validate saved metadata matches the current resume configuration.

    `num_examples` may increase between runs to request additional rollouts.
    """
    metadata_path = results_path / "metadata.json"

    try:
        with open(metadata_path, "r") as f:
            saved_metadata_raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Cannot resume from {results_path}: metadata at {metadata_path} is not valid JSON."
        ) from e

    if not isinstance(saved_metadata_raw, dict):
        raise ValueError(
            f"Cannot resume from {results_path}: metadata at {metadata_path} must be a JSON object."
        )

    saved_metadata = cast(dict[str, Any], saved_metadata_raw)
    expected = {
        "env_id": env_id,
        "model": model,
        "rollouts_per_example": rollouts_per_example,
    }

    mismatches: list[str] = []
    for field, expected_value in expected.items():
        saved_value = saved_metadata.get(field, "<missing>")
        if saved_value != expected_value:
            mismatches.append(
                f"{field}: saved={saved_value!r}, current={expected_value!r}"
            )

    saved_num_examples = saved_metadata.get("num_examples", "<missing>")
    if not isinstance(saved_num_examples, int):
        mismatches.append(
            f"num_examples: saved={saved_num_examples!r}, current={num_examples!r}"
        )
    elif num_examples < saved_num_examples:
        mismatches.append(
            f"num_examples: saved={saved_num_examples!r}, current={num_examples!r} (current must be >= saved)"
        )

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            f"Cannot resume from {results_path}: metadata mismatch ({mismatch_text}). "
            "Use matching evaluation settings or provide a new results path."
        )


def save_outputs(outputs: list[RolloutOutput], results_path: Path, mode: str = "w"):
    """Save outputs to disk."""
    results_path.mkdir(parents=True, exist_ok=True)
    outputs_path = results_path / "results.jsonl"
    with open(outputs_path, mode) as f:
        for idx, output in enumerate(outputs):
            example_id = output.get("example_id") or "unknown"
            try:
                json.dump(output, f, default=make_serializable)
                f.write("\n")
            except Exception as e:
                logger.error(
                    f"Failed to save result with index {idx} ({example_id=}): {e}"
                )


def _get_last_nonempty_line_bounds(file_obj: Any) -> tuple[int, bytes] | None:
    """Return byte offset + contents for the last non-empty line in a file."""
    file_obj.seek(0, 2)
    file_size = file_obj.tell()
    if file_size == 0:
        return None

    cursor = file_size

    # Skip trailing whitespace/newlines to locate the real end of the last row.
    while cursor > 0:
        cursor -= 1
        file_obj.seek(cursor)
        if file_obj.read(1) not in b" \t\r\n":
            break
    else:
        return None

    line_end = cursor + 1
    line_start = cursor
    while line_start > 0:
        file_obj.seek(line_start - 1)
        if file_obj.read(1) == b"\n":
            break
        line_start -= 1

    file_obj.seek(line_start)
    return line_start, file_obj.read(line_end - line_start)


def _truncate_malformed_trailing_line(outputs_path: Path) -> None:
    """Drop a malformed trailing JSONL row so future appends stay valid."""
    if not outputs_path.exists() or not outputs_path.is_file():
        return

    with open(outputs_path, "rb+") as f:
        last_line_info = _get_last_nonempty_line_bounds(f)
        if last_line_info is None:
            return

        line_start, line_bytes = last_line_info
        try:
            json.loads(line_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning(
                "Removing malformed trailing line in %s at byte offset %s",
                outputs_path,
                line_start,
            )
            f.truncate(line_start)


def save_new_outputs(new_outputs: list[RolloutOutput], results_path: Path):
    """Saves new rollout outputs to disk (in append mode)."""
    outputs_path = results_path / "results.jsonl"
    _truncate_malformed_trailing_line(outputs_path)
    save_outputs(new_outputs, results_path, mode="a")


def sanitize_metadata(metadata: GenerateMetadata) -> dict:
    """Sanitizes metadata before saving to disk."""

    metadata_dict = dict(metadata)
    metadata_dict.pop("path_to_save")
    metadata_dict.pop("date")

    return metadata_dict


def save_metadata(metadata: GenerateMetadata, result_path: Path):
    """Saves metadata to disk."""

    result_path.mkdir(parents=True, exist_ok=True)
    metadata_path = result_path / "metadata.json"
    metadata_dict = sanitize_metadata(metadata)
    with open(metadata_path, "w") as f:
        try:
            json.dump(metadata_dict, f, default=make_serializable)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")


def make_dataset(results: GenerateOutputs) -> Dataset:
    """Create a Dataset from GenerateOutputs (outputs are already serialized)."""
    return Dataset.from_list(list(results["outputs"]))


def push_results_to_hf_hub(results: GenerateOutputs, dataset_name: str | None = None):
    """Push results to Hugging Face Hub."""
    dataset_name = dataset_name or get_hf_hub_dataset_name(results)
    try:
        dataset = make_dataset(results)
        dataset.push_to_hub(dataset_name)
        logger.info(f"Results pushed to Hugging Face Hub: {dataset_name}")
    except Exception as e:
        logger.error(f"Error pushing results to Hugging Face Hub: {e}")
