"""Tests for verifiers.utils.save_utils serialization behavior.

Covers:
- make_serializable: JSON serialization for non-standard types
- states_to_outputs: state to output conversion before saving
- sanitize_metadata: metadata sanitization before saving
- save_to_disk: disk saving with proper serialization
"""

import json
from datetime import date, datetime
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from verifiers.utils.save_utils import (
    extract_usage_tokens,
    make_serializable,
    states_to_outputs,
)


# Test models for make_serializable tests
class SimpleModel(BaseModel):
    name: str
    value: int


class NestedModel(BaseModel):
    inner: SimpleModel
    tags: list[str]


class TestSerialization:
    def test_serialize_simple_pydantic_model(self):
        model = SimpleModel(name="test", value=42)
        result = json.loads(json.dumps(model, default=make_serializable))

        assert result == {"name": "test", "value": 42}
        assert isinstance(result, dict)

    def test_serialize_nested_pydantic_model(self):
        model = NestedModel(inner=SimpleModel(name="test", value=42), tags=["a", "b"])
        result = json.loads(json.dumps(model, default=make_serializable))

        assert result == {"inner": {"name": "test", "value": 42}, "tags": ["a", "b"]}
        assert isinstance(result, dict)

    def test_serialize_datetime(self):
        """Test that datetime is converted to ISO format string."""
        dt = datetime(2025, 1, 15, 10, 30, 45)
        result = json.loads(json.dumps(dt, default=make_serializable))

        assert result == "2025-01-15T10:30:45"
        assert isinstance(result, str)

    def test_serializable_date(self):
        """Test that date is converted to ISO format string."""
        d = date(2025, 12, 25)
        result = json.loads(json.dumps(d, default=make_serializable))

        assert result == "2025-12-25"
        assert isinstance(result, str)

    def test_serialize_path(self):
        """Test that Path is converted to POSIX string."""
        p = Path("/home/user/data/file.json")
        result = json.loads(json.dumps(p, default=make_serializable))

        assert result == "/home/user/data/file.json"
        assert isinstance(result, str)

    def test_serialize_exception(self):
        """Test that Exception is converted to string."""
        e = Exception("test exception")
        result = json.loads(json.dumps(e, default=make_serializable))

        assert result == "Exception('test exception')"
        assert isinstance(result, str)

    def test_serialize_unknown_type(self):
        class UnknownType:
            def __repr__(self):
                return "UnknownType()"

        obj = UnknownType()
        result = json.loads(json.dumps(obj, default=make_serializable))

        assert result == "UnknownType()"
        assert isinstance(result, str)


class TestSavingMetadata:
    def test_serialize_metadata(self, make_metadata):
        """Test serialization of complex nested structures."""

        metadata = make_metadata(
            env_args={"arg1": "value1"},
            model="test-model",
            base_url="http://localhost:8000",
            num_examples=100,
            rollouts_per_example=2,
            sampling_args={"temperature": 0.7},
            date="2025-01-01",
            time_ms=1000.0,
            avg_reward=0.5,
            avg_metrics={"num_turns": 1.0},
            usage={"input_tokens": 12.0, "output_tokens": 7.0},
            state_columns=[],
            path_to_save=Path("/results/test"),
            tools=None,
        )

        result = json.loads(json.dumps(metadata, default=make_serializable))

        assert result["env_id"] == "test-env"
        assert result["env_args"] == {"arg1": "value1"}
        assert result["model"] == "test-model"
        assert result["base_url"] == "http://localhost:8000"
        assert result["num_examples"] == 100
        assert result["rollouts_per_example"] == 2
        assert result["sampling_args"] == {"temperature": 0.7}
        assert result["date"] == "2025-01-01"
        assert result["time_ms"] == 1000.0
        assert result["avg_reward"] == 0.5
        assert result["avg_metrics"] == {"num_turns": 1.0}
        assert result["usage"] == {"input_tokens": 12.0, "output_tokens": 7.0}
        assert result["state_columns"] == []


class TestSavingResults:
    def test_extract_usage_tokens_prompt_completion(self):
        response = type(
            "Response",
            (),
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "input_tokens": 999,
                    "output_tokens": 999,
                }
            },
        )()
        input_tokens, output_tokens = extract_usage_tokens(response)
        assert input_tokens == 10
        assert output_tokens == 5

    def test_extract_usage_tokens_input_output(self):
        response = type(
            "Response",
            (),
            {"usage": {"input_tokens": 8, "output_tokens": 3}},
        )()
        input_tokens, output_tokens = extract_usage_tokens(response)
        assert input_tokens == 8
        assert output_tokens == 3

    def test_states_to_outputs(self, make_state):
        states = [
            make_state(
                prompt=[{"role": "user", "content": "What is 2+2?"}],
                completion=[{"role": "assistant", "content": "The answer is 4"}],
                answer="",
                info={},
                reward=1.0,
            ),
        ]
        outputs = states_to_outputs(states, state_columns=["foo"])
        result = json.loads(json.dumps(outputs, default=make_serializable))
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["example_id"] == 0
        assert result[0]["prompt"] == [{"role": "user", "content": "What is 2+2?"}]
        assert result[0]["completion"] == [
            {"role": "assistant", "content": "The answer is 4"}
        ]
        assert result[0].get("answer") is None  # empty answer not included
        assert result[0].get("info") is None  # empty info not included
        assert result[0].get("foo") == "bar"  # custom field from make_state fixture
        assert result[0]["reward"] == 1.0

    def test_non_serializable_state_column_raises(self, make_state):
        """Non-serializable state_columns should raise ValueError."""
        import pytest

        states = [
            make_state(
                prompt=[{"role": "user", "content": "test"}],
                completion=[{"role": "assistant", "content": "test"}],
                client=OpenAI(api_key="EMPTY"),
            ),
        ]
        with pytest.raises(ValueError, match="not JSON-serializable"):
            states_to_outputs(states, state_columns=["client"])
