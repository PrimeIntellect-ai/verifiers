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

import pytest
from openai import OpenAI
from pydantic import BaseModel

from verifiers.types import ClientConfig
from verifiers.utils.save_utils import (
    GenerateOutputsBuilder,
    build_trajectories_list,
    extract_usage_tokens,
    load_outputs,
    make_serializable,
    save_new_outputs,
    save_trajectories,
    state_to_output,
    states_to_outputs,
    trajectory_to_serializable,
    validate_resume_metadata,
)
from verifiers.utils.usage_utils import StateUsageTracker


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

    def test_generate_outputs_builder_serializes_endpoint_configs_base_url(self):
        builder = GenerateOutputsBuilder(
            env_id="test-env",
            env_args={},
            model="test-model",
            client=ClientConfig(
                api_base_url="http://localhost:8000/v1",
                endpoint_configs=[
                    ClientConfig(api_base_url="http://localhost:8000/v1"),
                    ClientConfig(api_base_url="http://localhost:8001/v1"),
                ],
            ),
            num_examples=1,
            rollouts_per_example=1,
            state_columns=[],
            sampling_args={},
            results_path=Path("/tmp/test-results"),
        )
        metadata = builder.build_metadata()
        assert isinstance(metadata["base_url"], str)
        assert (
            metadata["base_url"] == "http://localhost:8000/v1,http://localhost:8001/v1"
        )


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

    def test_extract_usage_tokens_invalid_values(self):
        response = type(
            "Response",
            (),
            {"usage": {"prompt_tokens": "bad", "completion_tokens": object()}},
        )()
        input_tokens, output_tokens = extract_usage_tokens(response)
        assert input_tokens == 0
        assert output_tokens == 0

    def test_state_with_tracker_and_no_usage_does_not_emit_token_usage(
        self, make_state
    ):
        state = make_state()
        tracker = StateUsageTracker()
        state["usage_tracker"] = tracker
        state["usage"] = tracker.usage
        state["trajectory"] = []
        output = states_to_outputs([state], state_columns=[])[0]
        assert "token_usage" not in output

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

    def test_trajectory_state_column_serializes_step_by_step_raw_content(
        self, make_state
    ):
        """state_columns=['trajectory'] saves step-by-step trajectory with raw prompt/completion (reasoning_content, content)."""
        # Step with raw content parts (e.g. reasoning_content + content)
        step_prompt = [{"role": "user", "content": "Solve 2+2"}]
        step_completion = [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning_content", "text": "Let me add."},
                    {"type": "content", "text": "4"},
                ],
            }
        ]
        mock_response = type(
            "Response",
            (),
            {"usage": {"input_tokens": 5, "output_tokens": 10}},
        )()
        trajectory = [
            {
                "prompt": step_prompt,
                "completion": step_completion,
                "response": mock_response,
                "tokens": None,
                "reward": None,
                "advantage": None,
                "is_truncated": False,
                "trajectory_id": "tid-1",
                "extras": {},
            }
        ]
        state = make_state(
            prompt=[{"role": "user", "content": "What is 2+2?"}],
            completion=[{"role": "assistant", "content": "4"}],
            trajectory=trajectory,
        )
        output = state_to_output(state, state_columns=["trajectory"])
        assert "trajectory" in output
        traj = output["trajectory"]
        assert len(traj) == 1
        assert traj[0]["prompt"] == step_prompt
        assert traj[0]["completion"] == step_completion
        assert traj[0]["usage"] == {"input_tokens": 5.0, "output_tokens": 10.0}
        assert traj[0]["trajectory_id"] == "tid-1"
        # Round-trip JSON (raw content with reasoning_content preserved)
        dumped = json.dumps(output, default=make_serializable)
        loaded = json.loads(dumped)
        assert loaded["trajectory"][0]["completion"][0]["content"][0]["type"] == "reasoning_content"
        assert loaded["trajectory"][0]["completion"][0]["content"][1]["type"] == "content"


class TestLoadOutputs:
    def test_ignores_malformed_trailing_line(self, tmp_path: Path):
        results_path = tmp_path / "results"
        results_path.mkdir()
        outputs_path = results_path / "results.jsonl"

        valid_outputs = [
            {"example_id": 0, "task": "task-0"},
            {"example_id": 1, "task": "task-1"},
        ]
        partial_trailing_line = '{"example_id": 2, "task": "task-2"'
        lines = [json.dumps(output) for output in valid_outputs]
        outputs_path.write_text(
            "\n".join(lines + [partial_trailing_line]) + "\n", encoding="utf-8"
        )

        outputs = load_outputs(results_path)

        assert len(outputs) == 2
        assert outputs[0]["example_id"] == 0
        assert outputs[1]["example_id"] == 1

    def test_raises_for_malformed_non_trailing_line(self, tmp_path: Path):
        results_path = tmp_path / "results"
        results_path.mkdir()
        outputs_path = results_path / "results.jsonl"

        malformed_non_trailing_line = '{"example_id": 0, "task": "broken"'
        valid_line = json.dumps({"example_id": 1, "task": "task-1"})
        outputs_path.write_text(
            "\n".join([malformed_non_trailing_line, valid_line]) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(json.JSONDecodeError):
            load_outputs(results_path)


class TestSaveNewOutputs:
    def test_truncates_malformed_trailing_line_before_append(self, tmp_path: Path):
        results_path = tmp_path / "results"
        results_path.mkdir()
        outputs_path = results_path / "results.jsonl"

        existing_outputs = [
            {"example_id": 0, "task": "task-0"},
            {"example_id": 1, "task": "task-1"},
        ]
        malformed_trailing_line = '{"example_id": 2, "task": "task-2"'
        lines = [json.dumps(output) for output in existing_outputs]
        outputs_path.write_text(
            "\n".join(lines + [malformed_trailing_line]), encoding="utf-8"
        )

        save_new_outputs(
            [{"example_id": 3, "task": "task-3"}],
            results_path,
        )

        persisted_lines = [
            line
            for line in outputs_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        parsed_outputs = [json.loads(line) for line in persisted_lines]

        assert [output["example_id"] for output in parsed_outputs] == [0, 1, 3]
        assert [output["example_id"] for output in load_outputs(results_path)] == [
            0,
            1,
            3,
        ]


class TestTrajectoriesFile:
    def test_build_trajectories_list_shape_and_metadata(self):
        """trajectories.json shape: list of {metadata, steps}, steps = [{input, output}], metadata maps to results.jsonl."""
        outputs = [
            {
                "example_id": 0,
                "task": "default",
                "reward": 0.5,
                "is_completed": True,
                "trajectory": [
                    {
                        "prompt": [{"role": "user", "content": "Hi"}],
                        "completion": [{"role": "assistant", "content": "Hello"}],
                    },
                    {
                        "prompt": [{"role": "user", "content": "Bye"}],
                        "completion": [{"role": "assistant", "content": "Bye"}],
                    },
                ],
            },
            {"example_id": 1, "task": "other", "reward": 0.0, "trajectory": []},
            {
                "example_id": 2,
                "task": "t2",
                "reward": 1.0,
                "trajectory": [
                    {
                        "prompt": [{"role": "user", "content": "One"}],
                        "completion": [{"role": "assistant", "content": "Two"}],
                    },
                ],
            },
        ]
        out = build_trajectories_list(outputs)
        assert len(out) == 2
        assert out[0]["metadata"]["results_index"] == 0
        assert out[0]["metadata"]["example_id"] == 0
        assert out[0]["metadata"]["task"] == "default"
        assert out[0]["metadata"]["reward"] == 0.5
        assert len(out[0]["steps"]) == 2
        assert out[0]["steps"][0]["input"] == [{"role": "user", "content": "Hi"}]
        assert out[0]["steps"][0]["output"] == [{"role": "assistant", "content": "Hello"}]
        assert out[1]["metadata"]["results_index"] == 2
        assert out[1]["steps"][0]["input"] == [{"role": "user", "content": "One"}]

    def test_save_trajectories_writes_json_and_skips_when_empty(self, tmp_path: Path):
        results_path = tmp_path / "results"
        outputs_with_traj = [
            {
                "example_id": 0,
                "task": "default",
                "reward": 0.0,
                "trajectory": [
                    {"prompt": [{"role": "user", "content": "x"}], "completion": [{"role": "assistant", "content": "y"}]},
                ],
            },
        ]
        save_trajectories(outputs_with_traj, results_path)
        assert (results_path / "trajectories.json").exists()
        data = json.loads((results_path / "trajectories.json").read_text())
        assert len(data) == 1
        assert data[0]["metadata"]["results_index"] == 0
        assert data[0]["steps"][0]["input"][0]["content"] == "x"
        save_trajectories([], results_path)
        assert (results_path / "trajectories.json").exists()
        save_trajectories([{"example_id": 0, "task": "t", "reward": 0.0}], results_path)
        content_after = (results_path / "trajectories.json").read_text()
        assert content_after  # file still exists from before; we don't delete it when empty
        data_after = json.loads(content_after)
        assert len(data_after) == 1


class TestResumeMetadataValidation:
    def test_validate_resume_metadata_accepts_matching_config(self, tmp_path: Path):
        results_path = tmp_path / "results"
        results_path.mkdir()
        metadata_path = results_path / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "env_id": "math-env",
                    "model": "test-model",
                    "num_examples": 3,
                    "rollouts_per_example": 2,
                }
            ),
            encoding="utf-8",
        )

        validate_resume_metadata(
            results_path=results_path,
            env_id="math-env",
            model="test-model",
            num_examples=3,
            rollouts_per_example=2,
        )

    def test_validate_resume_metadata_accepts_increased_num_examples(
        self, tmp_path: Path
    ):
        results_path = tmp_path / "results"
        results_path.mkdir()
        metadata_path = results_path / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "env_id": "math-env",
                    "model": "test-model",
                    "num_examples": 3,
                    "rollouts_per_example": 2,
                }
            ),
            encoding="utf-8",
        )

        validate_resume_metadata(
            results_path=results_path,
            env_id="math-env",
            model="test-model",
            num_examples=5,
            rollouts_per_example=2,
        )

    def test_validate_resume_metadata_raises_on_mismatch(self, tmp_path: Path):
        results_path = tmp_path / "results"
        results_path.mkdir()
        metadata_path = results_path / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "env_id": "math-env",
                    "model": "test-model",
                    "num_examples": 8,
                    "rollouts_per_example": 2,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="metadata mismatch"):
            validate_resume_metadata(
                results_path=results_path,
                env_id="math-env",
                model="test-model",
                num_examples=3,
                rollouts_per_example=2,
            )
