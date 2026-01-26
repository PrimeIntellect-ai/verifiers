"""Tests for verifiers.utils.save_utils serialization behavior.

Covers:
- make_serializable: JSON serialization for non-standard types
- sanitize_states: state sanitization before saving
- sanitize_metadata: metadata sanitization before saving
- save_to_disk: disk saving with proper serialization
"""

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

from pydantic import BaseModel

from verifiers.types import GenerateMetadata, State
from verifiers.utils.save_utils import (
    make_serializable,
    sanitize_metadata,
    sanitize_states,
    save_to_disk,
)


# Test models for make_serializable tests
class SimpleModel(BaseModel):
    name: str
    value: int


class NestedModel(BaseModel):
    inner: SimpleModel
    tags: list[str]


# =============================================================================
# make_serializable tests
# =============================================================================


def test_make_serializable_pydantic_basemodel():
    """Test that Pydantic BaseModel is converted to dict via model_dump."""
    model = SimpleModel(name="test", value=42)
    result = make_serializable(model)

    assert result == {"name": "test", "value": 42}
    assert isinstance(result, dict)


def test_make_serializable_nested_pydantic_model():
    """Test nested Pydantic models are serialized correctly."""
    inner = SimpleModel(name="inner", value=10)
    outer = NestedModel(inner=inner, tags=["a", "b"])
    result = make_serializable(outer)

    assert result == {"inner": {"name": "inner", "value": 10}, "tags": ["a", "b"]}


def test_make_serializable_datetime():
    """Test that datetime is converted to ISO format string."""
    dt = datetime(2025, 1, 15, 10, 30, 45)
    result = make_serializable(dt)

    assert result == "2025-01-15T10:30:45"
    assert isinstance(result, str)


def test_make_serializable_datetime_with_microseconds():
    """Test datetime with microseconds is serialized correctly."""
    dt = datetime(2025, 6, 20, 14, 25, 30, 123456)
    result = make_serializable(dt)

    assert result == "2025-06-20T14:25:30.123456"


def test_make_serializable_date():
    """Test that date is converted to ISO format string."""
    d = date(2025, 12, 25)
    result = make_serializable(d)

    assert result == "2025-12-25"
    assert isinstance(result, str)


def test_make_serializable_path():
    """Test that Path is converted to POSIX string."""
    p = Path("/home/user/data/file.json")
    result = make_serializable(p)

    assert result == "/home/user/data/file.json"
    assert isinstance(result, str)


def test_make_serializable_path_relative():
    """Test relative Path is converted correctly."""
    p = Path("data/results/output.jsonl")
    result = make_serializable(p)

    assert result == "data/results/output.jsonl"


def test_make_serializable_path_with_spaces():
    """Test Path with spaces in name is handled correctly."""
    p = Path("/home/user/my data/file name.json")
    result = make_serializable(p)

    assert result == "/home/user/my data/file name.json"


def test_make_serializable_unknown_type_uses_repr():
    """Test that unknown types fall back to repr()."""

    class CustomClass:
        def __repr__(self):
            return "CustomClass(special)"

    obj = CustomClass()
    result = make_serializable(obj)

    assert result == "CustomClass(special)"
    assert isinstance(result, str)


def test_make_serializable_mock_object():
    """Test that mock objects are serialized using repr."""
    mock = MagicMock(name="test_mock")
    result = make_serializable(mock)

    assert isinstance(result, str)
    assert "test_mock" in result or "MagicMock" in result


def test_make_serializable_with_json_dumps():
    """Test make_serializable works correctly as json.dumps default."""
    data = {
        "model": SimpleModel(name="test", value=1),
        "timestamp": datetime(2025, 1, 1, 12, 0, 0),
        "path": Path("/data/file.json"),
    }

    result = json.dumps(data, default=make_serializable)
    parsed = json.loads(result)

    assert parsed["model"] == {"name": "test", "value": 1}
    assert parsed["timestamp"] == "2025-01-01T12:00:00"
    assert parsed["path"] == "/data/file.json"


def test_make_serializable_complex_nested_structure():
    """Test serialization of complex nested structures."""

    class CustomObj:
        def __repr__(self):
            return "<CustomObj>"

    data = {
        "items": [
            {"date": date(2025, 1, 1), "path": Path("/a")},
            {"date": date(2025, 2, 1), "path": Path("/b")},
        ],
        "config": SimpleModel(name="cfg", value=99),
        "custom": CustomObj(),
    }

    result = json.dumps(data, default=make_serializable)
    parsed = json.loads(result)

    assert len(parsed["items"]) == 2
    assert parsed["items"][0]["date"] == "2025-01-01"
    assert parsed["config"]["value"] == 99
    assert parsed["custom"] == "<CustomObj>"


# =============================================================================
# sanitize_states tests
# =============================================================================


def _make_state(
    prompt=None,
    completion=None,
    reward=0.5,
    error=None,
    answer=None,
    info=None,
    metrics=None,
    example_id=0,
) -> State:
    """Helper to create a State object for testing.

    Creates a State with the structure expected by sanitize_states:
    - prompt, completion, reward, error, metrics at top level
    - answer, info at top level (optional fields that get popped if empty)
    - input dict with prompt, example_id, task, and optionally answer/info

    Note: We use dict.__setitem__ to bypass State's forwarding behavior
    for INPUT_FIELDS, ensuring these are set at the top level.
    """
    state = State()

    # Build the input dict
    input_dict = {
        "prompt": prompt or [{"role": "user", "content": "test question"}],
        "example_id": example_id,
        "task": "default",
    }
    if answer is not None:
        input_dict["answer"] = answer
    if info is not None:
        input_dict["info"] = info

    state["input"] = input_dict

    # Set all fields at the top level using dict.__setitem__ to bypass
    # State's forwarding behavior for INPUT_FIELDS
    dict.__setitem__(
        state, "prompt", prompt or [{"role": "user", "content": "test question"}]
    )
    dict.__setitem__(
        state,
        "completion",
        completion or [{"role": "assistant", "content": "test answer"}],
    )
    dict.__setitem__(state, "reward", reward)
    dict.__setitem__(state, "error", error)
    dict.__setitem__(state, "metrics", metrics if metrics is not None else {})

    # answer and info must be at top level for sanitize_states to work
    # (it calls dict(state) then tries to pop these keys)
    dict.__setitem__(state, "answer", answer if answer is not None else "")
    dict.__setitem__(state, "info", info if info is not None else {})

    return state


def test_sanitize_states_basic():
    """Test basic state sanitization."""
    states = [_make_state(reward=0.8)]
    result = sanitize_states(states)

    assert len(result) == 1
    assert result[0]["reward"] == 0.8


def test_sanitize_states_converts_messages_to_printable():
    """Test that prompt and completion are converted to printable format."""
    prompt = [{"role": "user", "content": "Hello"}]
    completion = [{"role": "assistant", "content": "Hi there"}]
    states = [_make_state(prompt=prompt, completion=completion)]

    result = sanitize_states(states)

    # Messages should be converted (messages_to_printable is applied)
    assert "prompt" in result[0]
    assert "completion" in result[0]


def test_sanitize_states_error_converted_to_repr():
    """Test that error field is converted to string repr."""

    class CustomError(Exception):
        def __repr__(self):
            return "CustomError('test failure')"

    error = CustomError("test failure")
    states = [_make_state(error=error)]

    result = sanitize_states(states)

    assert result[0]["error"] == "CustomError('test failure')"
    assert isinstance(result[0]["error"], str)


def test_sanitize_states_none_error():
    """Test that None error is handled correctly."""
    states = [_make_state(error=None)]

    result = sanitize_states(states)

    assert result[0]["error"] == "None"


def test_sanitize_states_removes_empty_answer():
    """Test that empty answer field is removed."""
    states = [_make_state(answer="")]

    result = sanitize_states(states)

    assert "answer" not in result[0]


def test_sanitize_states_keeps_non_empty_answer():
    """Test that non-empty answer field is kept."""
    states = [_make_state(answer="42")]

    result = sanitize_states(states)

    assert result[0].get("answer") == "42"


def test_sanitize_states_removes_empty_info():
    """Test that empty info field is removed."""
    states = [_make_state(info={})]

    result = sanitize_states(states)

    # Empty dict is falsy, so info should be removed
    assert "info" not in result[0]


def test_sanitize_states_keeps_non_empty_info():
    """Test that non-empty info field is kept."""
    states = [_make_state(info={"key": "value"})]

    result = sanitize_states(states)

    assert result[0].get("info") == {"key": "value"}


def test_sanitize_states_flattens_metrics():
    """Test that metrics dict is flattened into the state."""
    metrics = {"accuracy": 0.95, "f1_score": 0.88}
    states = [_make_state(metrics=metrics)]

    result = sanitize_states(states)

    assert result[0]["accuracy"] == 0.95
    assert result[0]["f1_score"] == 0.88


def test_sanitize_states_with_state_columns():
    """Test that state_columns are included in sanitized output."""
    state = _make_state()
    state["custom_field"] = "custom_value"
    state["another_field"] = 123

    result = sanitize_states([state], state_columns=["custom_field", "another_field"])

    assert result[0]["custom_field"] == "custom_value"
    assert result[0]["another_field"] == 123


def test_sanitize_states_multiple_states():
    """Test sanitizing multiple states."""
    states = [
        _make_state(reward=0.1, example_id=0),
        _make_state(reward=0.5, example_id=1),
        _make_state(reward=0.9, example_id=2),
    ]

    result = sanitize_states(states)

    assert len(result) == 3
    assert [r["reward"] for r in result] == [0.1, 0.5, 0.9]


# =============================================================================
# sanitize_metadata tests
# =============================================================================


def _make_metadata(**overrides) -> GenerateMetadata:
    """Helper to create GenerateMetadata for testing."""
    defaults = {
        "env_id": "test-env",
        "env_args": {"arg1": "value1"},
        "model": "test-model",
        "base_url": "http://localhost:8000",
        "num_examples": 100,
        "rollouts_per_example": 2,
        "sampling_args": {"temperature": 0.7},
        "date": "2025-01-15T10:30:00",
        "time_ms": 5000.0,
        "avg_reward": 0.75,
        "avg_metrics": {"accuracy": 0.85},
        "state_columns": ["custom_col"],
        "path_to_save": Path("/results/test"),
        "tools": None,
    }
    defaults.update(overrides)
    return GenerateMetadata(**defaults)


def test_sanitize_metadata_removes_path_to_save():
    """Test that path_to_save is removed from metadata."""
    metadata = _make_metadata(path_to_save=Path("/some/path"))

    result = sanitize_metadata(metadata)

    assert "path_to_save" not in result


def test_sanitize_metadata_removes_date():
    """Test that date field is removed from metadata."""
    metadata = _make_metadata(date="2025-06-15T12:00:00")

    result = sanitize_metadata(metadata)

    assert "date" not in result


def test_sanitize_metadata_keeps_other_fields():
    """Test that other metadata fields are preserved."""
    metadata = _make_metadata(
        env_id="my-env",
        model="gpt-4",
        num_examples=50,
        avg_reward=0.9,
    )

    result = sanitize_metadata(metadata)

    assert result["env_id"] == "my-env"
    assert result["model"] == "gpt-4"
    assert result["num_examples"] == 50
    assert result["avg_reward"] == 0.9


def test_sanitize_metadata_preserves_env_args():
    """Test that env_args dict is preserved."""
    metadata = _make_metadata(env_args={"dataset": "gsm8k", "split": "test"})

    result = sanitize_metadata(metadata)

    assert result["env_args"] == {"dataset": "gsm8k", "split": "test"}


def test_sanitize_metadata_preserves_sampling_args():
    """Test that sampling_args dict is preserved."""
    metadata = _make_metadata(sampling_args={"temperature": 0.8, "max_tokens": 1024})

    result = sanitize_metadata(metadata)

    assert result["sampling_args"] == {"temperature": 0.8, "max_tokens": 1024}


def test_sanitize_metadata_preserves_avg_metrics():
    """Test that avg_metrics dict is preserved."""
    metadata = _make_metadata(avg_metrics={"acc": 0.9, "f1": 0.85, "recall": 0.88})

    result = sanitize_metadata(metadata)

    assert result["avg_metrics"] == {"acc": 0.9, "f1": 0.85, "recall": 0.88}


# =============================================================================
# save_to_disk tests
# =============================================================================


def test_save_to_disk_creates_directory(tmp_path):
    """Test that save_to_disk creates the output directory."""
    output_path = tmp_path / "nested" / "output" / "dir"
    results = [{"example_id": 0, "reward": 0.5}]
    metadata = {"env_id": "test"}

    save_to_disk(results, metadata, output_path)

    assert output_path.exists()
    assert output_path.is_dir()


def test_save_to_disk_creates_metadata_file(tmp_path):
    """Test that metadata.json is created with correct content."""
    results = [{"example_id": 0, "reward": 0.5}]
    metadata = {"env_id": "test-env", "model": "test-model", "avg_reward": 0.5}

    save_to_disk(results, metadata, tmp_path)

    metadata_path = tmp_path / "metadata.json"
    assert metadata_path.exists()

    with open(metadata_path) as f:
        saved_metadata = json.load(f)

    assert saved_metadata["env_id"] == "test-env"
    assert saved_metadata["model"] == "test-model"
    assert saved_metadata["avg_reward"] == 0.5


def test_save_to_disk_creates_results_file(tmp_path):
    """Test that results.jsonl is created with correct content."""
    results = [
        {"example_id": 0, "reward": 0.3},
        {"example_id": 1, "reward": 0.7},
        {"example_id": 2, "reward": 0.9},
    ]
    metadata = {"env_id": "test"}

    save_to_disk(results, metadata, tmp_path)

    results_path = tmp_path / "results.jsonl"
    assert results_path.exists()

    with open(results_path) as f:
        lines = f.readlines()

    assert len(lines) == 3

    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["example_id"] == 0
    assert parsed[1]["reward"] == 0.7
    assert parsed[2]["example_id"] == 2


def test_save_to_disk_uses_make_serializable_for_results(tmp_path):
    """Test that results are serialized using make_serializable."""
    results = [
        {
            "example_id": 0,
            "timestamp": datetime(2025, 1, 15, 10, 30, 0),
            "path": Path("/data/file.json"),
        }
    ]
    metadata = {"env_id": "test"}

    save_to_disk(results, metadata, tmp_path)

    with open(tmp_path / "results.jsonl") as f:
        parsed = json.loads(f.readline())

    assert parsed["timestamp"] == "2025-01-15T10:30:00"
    assert parsed["path"] == "/data/file.json"


def test_save_to_disk_uses_make_serializable_for_metadata(tmp_path):
    """Test that metadata is serialized using make_serializable."""
    results = [{"example_id": 0}]
    metadata = {
        "env_id": "test",
        "created_at": datetime(2025, 6, 20, 14, 0, 0),
        "config_path": Path("/configs/eval.toml"),
    }

    save_to_disk(results, metadata, tmp_path)

    with open(tmp_path / "metadata.json") as f:
        parsed = json.load(f)

    assert parsed["created_at"] == "2025-06-20T14:00:00"
    assert parsed["config_path"] == "/configs/eval.toml"


def test_save_to_disk_handles_pydantic_models(tmp_path):
    """Test serialization of Pydantic models in results."""
    model = SimpleModel(name="test", value=42)
    results = [{"example_id": 0, "config": model}]
    metadata = {"env_id": "test"}

    save_to_disk(results, metadata, tmp_path)

    with open(tmp_path / "results.jsonl") as f:
        parsed = json.loads(f.readline())

    assert parsed["config"] == {"name": "test", "value": 42}


def test_save_to_disk_handles_empty_results(tmp_path):
    """Test saving with empty results list."""
    results = []
    metadata = {"env_id": "test"}

    save_to_disk(results, metadata, tmp_path)

    with open(tmp_path / "results.jsonl") as f:
        content = f.read()

    assert content == ""


def test_save_to_disk_jsonl_format(tmp_path):
    """Test that results.jsonl has correct JSONL format (one JSON per line)."""
    results = [
        {"id": 0, "data": "first"},
        {"id": 1, "data": "second"},
    ]
    metadata = {"env_id": "test"}

    save_to_disk(results, metadata, tmp_path)

    with open(tmp_path / "results.jsonl") as f:
        content = f.read()

    # Should have exactly 2 newlines (one after each JSON object)
    assert content.count("\n") == 2

    # Each line should be valid JSON
    lines = content.strip().split("\n")
    for line in lines:
        json.loads(line)  # Should not raise


def test_save_to_disk_overwrites_existing(tmp_path):
    """Test that save_to_disk overwrites existing files."""
    results_v1 = [{"version": 1}]
    results_v2 = [{"version": 2}]
    metadata = {"env_id": "test"}

    save_to_disk(results_v1, metadata, tmp_path)
    save_to_disk(results_v2, metadata, tmp_path)

    with open(tmp_path / "results.jsonl") as f:
        parsed = json.loads(f.readline())

    assert parsed["version"] == 2


# =============================================================================
# Integration tests
# =============================================================================


def test_full_serialization_pipeline(tmp_path):
    """Test the full pipeline: create states -> sanitize -> save."""
    # Create realistic states
    states = [
        _make_state(
            prompt=[{"role": "user", "content": "What is 2+2?"}],
            completion=[{"role": "assistant", "content": "The answer is 4"}],
            reward=1.0,
            answer="4",
            metrics={"accuracy": 1.0},
            example_id=0,
        ),
        _make_state(
            prompt=[{"role": "user", "content": "What is 3+3?"}],
            completion=[{"role": "assistant", "content": "The answer is 5"}],
            reward=0.0,
            answer="6",
            metrics={"accuracy": 0.0},
            example_id=1,
        ),
    ]

    # Create metadata
    metadata = _make_metadata(
        env_id="math-test",
        num_examples=2,
        rollouts_per_example=1,
        avg_reward=0.5,
    )

    # Sanitize
    sanitized_states = sanitize_states(states)
    sanitized_metadata = sanitize_metadata(metadata)

    # Save
    save_to_disk(sanitized_states, sanitized_metadata, tmp_path)

    # Verify files exist
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "results.jsonl").exists()

    # Verify metadata content
    with open(tmp_path / "metadata.json") as f:
        saved_metadata = json.load(f)

    assert saved_metadata["env_id"] == "math-test"
    assert saved_metadata["avg_reward"] == 0.5
    assert "path_to_save" not in saved_metadata
    assert "date" not in saved_metadata

    # Verify results content
    with open(tmp_path / "results.jsonl") as f:
        lines = f.readlines()

    assert len(lines) == 2

    result_0 = json.loads(lines[0])
    result_1 = json.loads(lines[1])

    assert result_0["reward"] == 1.0
    assert result_0["accuracy"] == 1.0
    assert result_1["reward"] == 0.0


def test_serialization_with_complex_types(tmp_path):
    """Test serialization with various complex types mixed together."""
    state = _make_state()
    state["timestamp"] = datetime(2025, 1, 1, 0, 0, 0)
    state["config_path"] = Path("/etc/config.yaml")
    state["model_config"] = SimpleModel(name="gpt", value=4)

    sanitized = sanitize_states(
        [state], state_columns=["timestamp", "config_path", "model_config"]
    )

    save_to_disk(sanitized, {"env_id": "test"}, tmp_path)

    with open(tmp_path / "results.jsonl") as f:
        parsed = json.loads(f.readline())

    assert parsed["timestamp"] == "2025-01-01T00:00:00"
    assert parsed["config_path"] == "/etc/config.yaml"
    assert parsed["model_config"] == {"name": "gpt", "value": 4}
