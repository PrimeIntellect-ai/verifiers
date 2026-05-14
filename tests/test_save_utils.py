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
from verifiers.utils.metric_utils import (
    EnvMetrics,
    ErrorRateMetric,
    InputTokensMetric,
    Metric,
    OutputTokensMetric,
    PassAtKMetric,
    RewardMetric,
)
from verifiers.utils.save_utils import (
    GenerateOutputsBuilder,
    _delta_intermediate_mm_data,
    extract_usage_tokens,
    load_outputs,
    make_serializable,
    save_new_outputs,
    states_to_outputs,
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
            time=1.0,
            avg_reward=0.5,
            avg_metrics={"num_turns": 1.0},
            usage={"input_tokens": 12.0, "output_tokens": 7.0},
            state_columns=[],
            path_to_save=Path("/results/test"),
            tools=None,
        )
        metadata["cost"] = {
            "input_usd": 0.001,
            "output_usd": 0.01,
            "total_usd": 0.011,
        }

        result = json.loads(json.dumps(metadata, default=make_serializable))

        assert result["env_id"] == "test-env"
        assert result["env_args"] == {"arg1": "value1"}
        assert result["model"] == "test-model"
        assert result["base_url"] == "http://localhost:8000"
        assert result["num_examples"] == 100
        assert result["rollouts_per_example"] == 2
        assert result["sampling_args"] == {"temperature": 0.7}
        assert result["date"] == "2025-01-01"
        assert result["time"] == 1.0
        assert result["avg_reward"] == 0.5
        assert result["avg_metrics"] == {"num_turns": 1.0}
        assert result["usage"] == {"input_tokens": 12.0, "output_tokens": 7.0}
        assert result["cost"] == {
            "input_usd": 0.001,
            "output_usd": 0.01,
            "total_usd": 0.011,
        }
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

    def test_states_to_outputs_completion_keeps_messages(self, make_state):
        states = [
            make_state(
                prompt=[
                    {"role": "text", "content": "Start:"},
                    {"role": "assistant", "content": "First response"},
                    {"role": "text", "content": " Continue."},
                ],
                completion=[
                    {"role": "assistant", "content": "Final DONE"},
                ],
                answer="",
                info={},
                reward=1.0,
                message_type="completion",
            )
        ]
        outputs = states_to_outputs(states, state_columns=[])
        result = json.loads(json.dumps(outputs, default=make_serializable))
        assert result[0]["prompt"] == [
            {"role": "text", "content": "Start:"},
            {"role": "assistant", "content": "First response"},
            {"role": "text", "content": " Continue."},
        ]
        assert result[0]["completion"] == [
            {"role": "assistant", "content": "Final DONE"},
        ]

    def test_states_to_outputs_preserves_multimodal_images_as_base64(self, make_state):
        states = [
            make_state(
                prompt=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this image"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,abc123"},
                            },
                        ],
                    }
                ],
                completion=[{"role": "assistant", "content": "A small chart."}],
                answer="",
                info={},
                reward=1.0,
            )
        ]

        outputs = states_to_outputs(states, state_columns=[])
        result = json.loads(json.dumps(outputs, default=make_serializable))

        assert result[0]["prompt"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                ],
            }
        ]

    def test_states_to_outputs_preserves_input_audio_payloads(self, make_state):
        states = [
            make_state(
                prompt=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "transcribe this"},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": " ZHVt\nbXk= ",
                                    "format": "MP3",
                                },
                            },
                        ],
                    }
                ],
                completion=[{"role": "assistant", "content": "dummy"}],
                answer="",
                info={},
                reward=1.0,
            )
        ]

        outputs = states_to_outputs(states, state_columns=[])
        result = json.loads(json.dumps(outputs, default=make_serializable))

        assert result[0]["prompt"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe this"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "ZHVtbXk=",
                            "format": "mp3",
                        },
                    },
                ],
            }
        ]

    def test_states_to_outputs_preserves_multimodal_completion_content(
        self, make_state
    ):
        states = [
            make_state(
                prompt=[{"role": "user", "content": "show me the observation"}],
                completion=[
                    {
                        "role": "tool",
                        "tool_call_id": "call_0",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,abc123"},
                            },
                            {
                                "type": "audio",
                                "data": " ZHVt\nbXk= ",
                                "format": "WAV",
                            },
                        ],
                    }
                ],
                answer="",
                info={},
                reward=1.0,
            )
        ]

        outputs = states_to_outputs(states, state_columns=[])
        result = json.loads(json.dumps(outputs, default=make_serializable))

        assert result[0]["completion"] == [
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "ZHVtbXk=",
                            "format": "wav",
                        },
                    },
                ],
            }
        ]

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


class TestLoadOutputs:
    def test_ignores_malformed_trailing_line(self, tmp_path: Path):
        results_path = tmp_path / "results"
        results_path.mkdir()
        outputs_path = results_path / "results.jsonl"

        valid_outputs = [
            {"example_id": 0, "label": "row-0"},
            {"example_id": 1, "label": "row-1"},
        ]
        partial_trailing_line = '{"example_id": 2, "label": "row-2"'
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

        malformed_non_trailing_line = '{"example_id": 0, "label": "broken"'
        valid_line = json.dumps({"example_id": 1, "label": "row-1"})
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
            {"example_id": 0, "label": "row-0"},
            {"example_id": 1, "label": "row-1"},
        ]
        malformed_trailing_line = '{"example_id": 2, "label": "row-2"'
        lines = [json.dumps(output) for output in existing_outputs]
        outputs_path.write_text(
            "\n".join(lines + [malformed_trailing_line]), encoding="utf-8"
        )

        save_new_outputs(
            [{"example_id": 3, "label": "row-3"}],
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


class TestBuilderPassAtK:
    """Tests for pass@k integration in GenerateOutputsBuilder."""

    def test_builder_includes_pass_at_k(self):
        """GenerateOutputsBuilder.build_metadata() includes pass_at_k and pass_all_k."""
        builder = GenerateOutputsBuilder(
            env_id="test-env",
            env_args={},
            model="test-model",
            client=ClientConfig(api_base_url="http://localhost:8000/v1"),
            num_examples=1,
            rollouts_per_example=4,
            state_columns=[],
            sampling_args={},
            results_path=Path("/tmp/test-results"),
        )
        builder.add_outputs(
            [
                {"example_id": 0, "reward": 1.0, "metrics": {}},
                {"example_id": 0, "reward": 0.0, "metrics": {}},
                {"example_id": 0, "reward": 1.0, "metrics": {}},
                {"example_id": 0, "reward": 0.0, "metrics": {}},
            ]
        )
        metadata = builder.build_metadata()
        assert set(metadata["pass_at_k"].keys()) == {"1", "2", "4"}
        assert set(metadata["pass_all_k"].keys()) == {"1", "2", "4"}
        assert metadata["pass_threshold"] == 0.5

    def test_builder_uses_custom_threshold(self):
        """GenerateOutputsBuilder respects pass_threshold."""
        builder = GenerateOutputsBuilder(
            env_id="test-env",
            env_args={},
            model="test-model",
            client=ClientConfig(api_base_url="http://localhost:8000/v1"),
            num_examples=1,
            rollouts_per_example=4,
            state_columns=[],
            sampling_args={},
            results_path=Path("/tmp/test-results"),
            pass_threshold=0.7,
        )
        builder.add_outputs(
            [
                {"example_id": 0, "reward": 0.4, "metrics": {}},
                {"example_id": 0, "reward": 0.6, "metrics": {}},
                {"example_id": 0, "reward": 0.8, "metrics": {}},
                {"example_id": 0, "reward": 0.3, "metrics": {}},
            ]
        )
        metadata = builder.build_metadata()
        assert metadata["pass_threshold"] == 0.7
        # 1 of 4 correct at threshold=0.7: pass@1 = 1 - C(3,1)/C(4,1) = 0.25
        assert metadata["pass_at_k"]["1"] == pytest.approx(0.25)
        # 1 of 4 correct at threshold=0.7: pass^1 = C(1,1)/C(4,1) = 0.25
        assert metadata["pass_all_k"]["1"] == pytest.approx(0.25)


class TestMetricProtocol:
    def test_all_metrics_satisfy_protocol(self):
        """All metric classes satisfy the Metric protocol."""
        assert isinstance(RewardMetric(), Metric)
        assert isinstance(ErrorRateMetric(), Metric)
        assert isinstance(InputTokensMetric(), Metric)
        assert isinstance(OutputTokensMetric(), Metric)
        assert isinstance(EnvMetrics(), Metric)
        assert isinstance(PassAtKMetric(rollouts_per_example=4), Metric)


class TestRewardMetric:
    def test_basic(self):
        m = RewardMetric()
        m.add_output({"reward": 0.3})
        m.add_output({"reward": 0.7})
        assert m.compute() == pytest.approx(0.5)
        assert m.count == 2

    def test_missing_reward_defaults_to_zero(self):
        m = RewardMetric()
        m.add_output({})
        assert m.compute() == pytest.approx(0.0)

    def test_add_outputs(self):
        m = RewardMetric()
        m.add_outputs([{"reward": 1.0}, {"reward": 0.0}])
        assert m.compute() == pytest.approx(0.5)

    def test_reset(self):
        m = RewardMetric()
        m.add_output({"reward": 1.0})
        m.reset()
        assert m.compute() == 0.0
        assert m.count == 0


class TestErrorRateMetric:
    def test_basic(self):
        m = ErrorRateMetric()
        m.add_output({"error": "some error"})
        m.add_output({})
        m.add_output({"error": None})
        m.add_output({"error": "another"})
        assert m.compute() == pytest.approx(0.5)


class TestInputTokensMetric:
    def test_skips_outputs_without_usage(self):
        m = InputTokensMetric()
        m.add_output({"token_usage": {"input_tokens": 100.0, "output_tokens": 50.0}})
        m.add_output({})  # skipped
        m.add_output({"token_usage": {"input_tokens": 200.0, "output_tokens": 100.0}})
        assert m.compute() == pytest.approx(150.0)
        assert m.count == 2


class TestOutputTokensMetric:
    def test_skips_outputs_without_usage(self):
        m = OutputTokensMetric()
        m.add_output({"token_usage": {"input_tokens": 100.0, "output_tokens": 50.0}})
        m.add_output({})  # skipped
        m.add_output({"token_usage": {"input_tokens": 200.0, "output_tokens": 100.0}})
        assert m.compute() == pytest.approx(75.0)
        assert m.count == 2


class TestEnvMetrics:
    def test_basic(self):
        m = EnvMetrics()
        m.add_output({"metrics": {"accuracy": 1.0, "f1": 0.8}})
        m.add_output({"metrics": {"accuracy": 0.0, "f1": 0.6}})
        result = m.compute()
        assert result["accuracy"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(0.7)

    def test_sparse_keys(self):
        m = EnvMetrics()
        m.add_output({"metrics": {"accuracy": 1.0, "f1": 0.8}})
        m.add_output({"metrics": {"accuracy": 0.0}})
        m.add_output({"metrics": {"accuracy": 0.0, "f1": 0.4}})
        result = m.compute()
        assert result["accuracy"] == pytest.approx(1.0 / 3)
        assert result["f1"] == pytest.approx(0.6)

    def test_empty(self):
        m = EnvMetrics()
        assert m.compute() == {}

    def test_reset(self):
        m = EnvMetrics()
        m.add_output({"metrics": {"acc": 1.0}})
        m.reset()
        assert m.compute() == {}


class TestPassAtKMetric:
    @staticmethod
    def _make_output(example_id: int, reward: float) -> dict:
        return {"example_id": example_id, "reward": reward}

    def test_incremental_matches_batch(self):
        """Incremental add_output matches batch compute_pass_at_k."""
        import random

        random.seed(42)
        rollouts_per_example = 8

        all_outputs = []
        for eid in range(10):
            for _ in range(rollouts_per_example):
                reward = random.choice([0.0, 0.5, 1.0])
                all_outputs.append(self._make_output(eid, reward))

        random.shuffle(all_outputs)

        # Incremental: one-by-one
        m1 = PassAtKMetric(rollouts_per_example)
        for output in all_outputs:
            m1.add_output(output)
            m1.compute()  # verify O(1) doesn't crash
        inc_pass_at_k, inc_pass_all_k = m1.compute()

        # Batch: add_outputs all at once
        m2 = PassAtKMetric(rollouts_per_example)
        m2.add_outputs(all_outputs)
        batch_pass_at_k, batch_pass_all_k = m2.compute()

        assert inc_pass_at_k == pytest.approx(batch_pass_at_k)
        assert inc_pass_all_k == pytest.approx(batch_pass_all_k)

    def test_add_outputs(self):
        m = PassAtKMetric(rollouts_per_example=2)
        m.add_outputs([self._make_output(0, 1.0), self._make_output(0, 0.0)])
        pass_at_k, _ = m.compute()
        assert pass_at_k["1"] == pytest.approx(0.5)

    def test_incomplete_examples_excluded(self):
        m = PassAtKMetric(rollouts_per_example=4)
        for _ in range(4):
            m.add_output(self._make_output(0, 1.0))
        for _ in range(2):
            m.add_output(self._make_output(1, 0.0))
        pass_at_k, pass_all_k = m.compute()
        assert pass_at_k["1"] == pytest.approx(1.0)
        assert pass_all_k["1"] == pytest.approx(1.0)

    def test_empty(self):
        m = PassAtKMetric(rollouts_per_example=4)
        assert m.compute() == ({}, {})

    def test_single_rollout(self):
        m = PassAtKMetric(rollouts_per_example=1)
        m.add_output(self._make_output(0, 1.0))
        assert m.compute() == ({}, {})

    def test_reset(self):
        m = PassAtKMetric(rollouts_per_example=2)
        m.add_output(self._make_output(0, 1.0))
        m.add_output(self._make_output(0, 1.0))
        pass_at_k, _ = m.compute()
        assert pass_at_k["1"] == pytest.approx(1.0)

        m.reset()
        m.add_output(self._make_output(0, 0.0))
        m.add_output(self._make_output(0, 0.0))
        pass_at_k, _ = m.compute()
        assert pass_at_k["1"] == pytest.approx(0.0)

    def test_custom_threshold(self):
        m = PassAtKMetric(rollouts_per_example=4, threshold=0.7)
        for o in [
            self._make_output(0, 0.4),
            self._make_output(0, 0.6),
            self._make_output(0, 0.8),
            self._make_output(0, 0.3),
        ]:
            m.add_output(o)
        pass_at_k, _ = m.compute()
        # 1 of 4 correct at threshold=0.7: pass@1 = 1 - C(3,1)/C(4,1) = 0.25
        assert pass_at_k["1"] == pytest.approx(0.25)
        # pass@2 = 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
        assert pass_at_k["2"] == pytest.approx(0.5)

    def test_all_correct(self):
        """All rollouts correct → pass@k = 1.0 and pass^k = 1.0 for all k."""
        m = PassAtKMetric(rollouts_per_example=8)
        m.add_outputs([self._make_output(0, 1.0) for _ in range(8)])
        pass_at_k, pass_all_k = m.compute()
        assert set(pass_at_k.keys()) == {"1", "2", "4", "8"}
        for k in pass_at_k:
            assert pass_at_k[k] == pytest.approx(1.0)
            assert pass_all_k[k] == pytest.approx(1.0)

    def test_none_correct(self):
        """No rollouts correct → pass@k = 0.0 and pass^k = 0.0 for all k."""
        m = PassAtKMetric(rollouts_per_example=8)
        m.add_outputs([self._make_output(0, 0.0) for _ in range(8)])
        pass_at_k, pass_all_k = m.compute()
        for k in pass_at_k:
            assert pass_at_k[k] == pytest.approx(0.0)
            assert pass_all_k[k] == pytest.approx(0.0)

    def test_partial_correctness(self):
        """Partial correctness: 2 correct out of 4 rollouts."""
        m = PassAtKMetric(rollouts_per_example=4)
        m.add_outputs(
            [
                self._make_output(0, 1.0),
                self._make_output(0, 1.0),
                self._make_output(0, 0.0),
                self._make_output(0, 0.0),
            ]
        )
        pass_at_k, pass_all_k = m.compute()
        assert pass_at_k["1"] == pytest.approx(0.5)
        assert pass_at_k["2"] == pytest.approx(1.0 - 1.0 / 6.0)
        assert pass_at_k["4"] == pytest.approx(1.0)
        assert pass_all_k["1"] == pytest.approx(0.5)
        assert pass_all_k["2"] == pytest.approx(1.0 / 6.0)
        assert pass_all_k["4"] == pytest.approx(0.0)

    def test_multiple_examples_averaged(self):
        """pass@k and pass^k are averaged across multiple examples."""
        m = PassAtKMetric(rollouts_per_example=4)
        m.add_outputs(
            [
                self._make_output(0, 1.0),
                self._make_output(0, 1.0),
                self._make_output(0, 1.0),
                self._make_output(0, 1.0),
                self._make_output(1, 0.0),
                self._make_output(1, 0.0),
                self._make_output(1, 0.0),
                self._make_output(1, 0.0),
            ]
        )
        pass_at_k, pass_all_k = m.compute()
        assert pass_at_k["1"] == pytest.approx(0.5)
        assert pass_all_k["1"] == pytest.approx(0.5)

    def test_powers_of_two_k_selection(self):
        """k values are powers of 2 in [1, n]."""
        m = PassAtKMetric(rollouts_per_example=16)
        m.add_outputs([self._make_output(0, 1.0) for _ in range(16)])
        pass_at_k, _ = m.compute()
        assert set(pass_at_k.keys()) == {"1", "2", "4", "8", "16"}

    def test_n3_k_values(self):
        """n=3 should give k=1,2."""
        m = PassAtKMetric(rollouts_per_example=3)
        m.add_outputs([self._make_output(0, 1.0) for _ in range(3)])
        pass_at_k, _ = m.compute()
        assert set(pass_at_k.keys()) == {"1", "2"}

    def test_correctness_threshold_boundary(self):
        """Only reward >= 0.5 counts as correct by default."""
        m = PassAtKMetric(rollouts_per_example=4)
        m.add_outputs(
            [
                self._make_output(0, 0.49),  # not correct
                self._make_output(0, 0.5),  # correct
                self._make_output(0, 1.0),  # correct
                self._make_output(0, 0.0),  # not correct
            ]
        )
        pass_at_k, _ = m.compute()
        assert pass_at_k["1"] == pytest.approx(0.5)


class TestDeltaIntermediateMmData:
    """Verify per-step delta encoding of trajectory mm_data sidecars.

    Renderer bridge_to_next_turn emits cumulative mm_data on every
    step. The transport-layer delta strips items whose mm_hash already
    appeared in the prior step, so the per-window TrainingSample
    assembler can recover its window's images by unioning step-deltas.
    """

    @staticmethod
    def _mm(*hashes: str):
        """Build a renderers.MultiModalData with one image item per hash."""
        from renderers.base import MultiModalData, PlaceholderRange

        return MultiModalData(
            mm_hashes={"image": list(hashes)},
            mm_placeholders={
                "image": [
                    PlaceholderRange(offset=i * 10, length=4)
                    for i in range(len(hashes))
                ]
            },
            mm_items={"image": [{"pixel_values": f"px-{h}"} for h in hashes]},
        )

    def _step(self, mm):
        return {"tokens": {"multi_modal_data": mm}}

    def test_none_and_single_step_passthrough(self):
        assert _delta_intermediate_mm_data(None) is None
        assert _delta_intermediate_mm_data([]) == []
        only = [self._step(self._mm("A"))]
        assert _delta_intermediate_mm_data(only) is only

    def test_linear_extension_keeps_only_new_items_per_step(self):
        traj = [
            self._step(self._mm("A")),
            self._step(self._mm("A", "B")),
            self._step(self._mm("A", "B", "C")),
        ]
        out = _delta_intermediate_mm_data(traj)

        assert out[0]["tokens"]["multi_modal_data"].mm_hashes == {"image": ["A"]}
        assert out[1]["tokens"]["multi_modal_data"].mm_hashes == {"image": ["B"]}
        assert out[2]["tokens"]["multi_modal_data"].mm_hashes == {"image": ["C"]}
        # Items and placeholders are reindexed in lockstep with hashes.
        assert out[1]["tokens"]["multi_modal_data"].mm_items["image"] == [
            {"pixel_values": "px-B"}
        ]
        assert (
            out[2]["tokens"]["multi_modal_data"].mm_placeholders["image"][0].offset
            == 20
        )

    def test_compaction_two_training_samples_assemble_correctly(self):
        """Rollout with one compaction event → two TrainingSamples.

        Models the prime-rl compaction flow: a single rollout produces
        multiple ``TrainingSample`` objects, one per compaction window.
        The pre-compaction sample's images are no longer in the
        post-compaction step's cumulative ``mm_data`` — the previous
        "keep last" strategy would have silently dropped them. With
        delta encoding, each per-window assembler recovers exactly the
        images its tokens reference: no leakage in either direction.
        """
        from renderers.base import MultiModalData, PlaceholderRange

        def step(*hashes: str, offsets: list[int]):
            return {
                "tokens": {
                    "multi_modal_data": MultiModalData(
                        mm_hashes={"image": list(hashes)},
                        mm_placeholders={
                            "image": [
                                PlaceholderRange(offset=o, length=4) for o in offsets
                            ]
                        },
                        mm_items={
                            "image": [{"pixel_values": f"px-{h}"} for h in hashes]
                        },
                    )
                }
            }

        # Turn 1: image A. Cumulative {A}.
        # Turn 2: image B. Cumulative {A, B}.
        # ── compaction event: turns 1+2 summarized in text, images dropped ──
        # Turn 3: image C. Cumulative {C} (offsets reset against the
        #         post-compaction prompt).
        # Turn 4: image D. Cumulative {C, D}.
        traj = [
            step("A", offsets=[10]),
            step("A", "B", offsets=[10, 50]),
            step("C", offsets=[8]),
            step("C", "D", offsets=[8, 40]),
        ]
        out = _delta_intermediate_mm_data(traj)

        # Per-step deltas keep only what's new since the immediately prior step.
        deltas = [s["tokens"]["multi_modal_data"].mm_hashes for s in out]
        assert deltas == [
            {"image": ["A"]},
            {"image": ["B"]},
            {"image": ["C"]},
            {"image": ["D"]},
        ]

        def assemble(steps):
            hashes: list[str] = []
            items: list[dict] = []
            placeholders: list[PlaceholderRange] = []
            for s in steps:
                mm = s["tokens"]["multi_modal_data"]
                hashes += mm.mm_hashes.get("image", [])
                items += mm.mm_items.get("image", [])
                placeholders += mm.mm_placeholders.get("image", [])
            return hashes, items, placeholders

        ts1_hashes, ts1_items, ts1_phs = assemble(out[0:2])  # pre-compaction
        ts2_hashes, ts2_items, ts2_phs = assemble(out[2:4])  # post-compaction

        assert ts1_hashes == ["A", "B"]
        assert ts2_hashes == ["C", "D"]
        # The invariant the previous "keep last" broke: pre-compaction TS
        # does not see post-compaction images, and vice versa.
        assert set(ts1_hashes).isdisjoint(set(ts2_hashes))

        # Items / placeholders are reindexed lock-step with hashes (no
        # off-by-one or cross-contamination during reindex).
        assert ts1_items == [{"pixel_values": "px-A"}, {"pixel_values": "px-B"}]
        assert ts2_items == [{"pixel_values": "px-C"}, {"pixel_values": "px-D"}]

        # Placeholder offsets travel verbatim per step; the assembler is
        # responsible for shifting them into each window's local frame.
        assert [p.offset for p in ts1_phs] == [10, 50]
        assert [p.offset for p in ts2_phs] == [8, 40]

    def test_same_image_rendered_in_two_turns_uses_multiset_diff(self):
        """Same image hash appearing N times must keep the right N-prior occurrences.

        The renderer doesn't dedupe by hash: ``emit_image`` appends to
        the parallel lists every time an image content part is rendered.
        So if image A is shown in turn 1 *and* turn 3, the cumulative
        ``mm_hashes`` is ``["A", "A"]`` with two distinct placeholder
        offsets, and ``mm_items`` is ``[pixA, pixA]`` (literally the
        same payload twice). Both placeholder runs need their own item
        — set-based diff would drop both as "already seen" and orphan
        the second placeholder. Multiset diff drops only the first.
        """
        from renderers.base import MultiModalData, PlaceholderRange

        def step(hashes, offsets):
            return {
                "tokens": {
                    "multi_modal_data": MultiModalData(
                        mm_hashes={"image": list(hashes)},
                        mm_placeholders={
                            "image": [
                                PlaceholderRange(offset=o, length=4) for o in offsets
                            ]
                        },
                        mm_items={
                            "image": [{"pixel_values": f"px-{h}"} for h in hashes]
                        },
                    )
                }
            }

        # Turn 1: image A at offset 10. Cumulative ["A"].
        # Turn 2: no image. Cumulative unchanged ["A"].
        # Turn 3: image A re-rendered at offset 200. Cumulative ["A", "A"].
        traj = [
            step(["A"], offsets=[10]),
            step(["A"], offsets=[10]),
            step(["A", "A"], offsets=[10, 200]),
        ]
        out = _delta_intermediate_mm_data(traj)

        # Step 0 keeps everything (no prior).
        assert out[0]["tokens"]["multi_modal_data"].mm_hashes == {"image": ["A"]}
        assert [
            p.offset
            for p in out[0]["tokens"]["multi_modal_data"].mm_placeholders["image"]
        ] == [10]

        # Step 1 introduced no new image (cumulative unchanged).
        assert out[1]["tokens"]["multi_modal_data"].mm_hashes == {"image": []}

        # Step 2: prior was ["A"], current is ["A", "A"]. Multiset budget
        # consumes the first A; the *second* A (the new one at offset
        # 200) survives the diff with its pixel_values intact. Set-based
        # diff would have produced [].
        step2_mm = out[2]["tokens"]["multi_modal_data"]
        assert step2_mm.mm_hashes == {"image": ["A"]}
        assert step2_mm.mm_items == {"image": [{"pixel_values": "px-A"}]}
        assert [p.offset for p in step2_mm.mm_placeholders["image"]] == [200]

        # End-to-end: assembling the single TrainingSample (no
        # compaction) recovers both placeholder runs with matching
        # pixel_values, so the trainer can satisfy both image-pad
        # token runs in the prompt.
        all_hashes: list[str] = []
        all_phs: list[PlaceholderRange] = []
        for s in out:
            mm = s["tokens"]["multi_modal_data"]
            all_hashes += mm.mm_hashes.get("image", [])
            all_phs += mm.mm_placeholders.get("image", [])
        assert all_hashes == ["A", "A"]
        assert [p.offset for p in all_phs] == [10, 200]

    def test_image_reintroduction_after_compaction(self):
        """A hash dropped at compaction and re-rendered later is re-transmitted.

        The delta is computed against the *immediately prior step's*
        cumulative, not a global seen-set. If image A appears in turn
        1, is compacted away (step 2's cumulative is empty), and is
        re-rendered in turn 3, A shows up in step 0's delta *and* step
        2's delta — necessary so the post-compaction TrainingSample
        also receives A's bytes.
        """
        traj = [
            self._step(self._mm("A")),
            self._step(self._mm()),
            self._step(self._mm("A")),
        ]
        out = _delta_intermediate_mm_data(traj)

        assert out[0]["tokens"]["multi_modal_data"].mm_hashes == {"image": ["A"]}
        assert out[1]["tokens"]["multi_modal_data"].mm_hashes == {"image": []}
        # A re-emerges in step 2's delta — its absence from step 1's
        # cumulative means it counts as "new" again.
        assert out[2]["tokens"]["multi_modal_data"].mm_hashes == {"image": ["A"]}

    def test_steps_with_no_new_items_collapse_to_empty_delta(self):
        # Step 2's cumulative equals step 1's — no new items.
        traj = [
            self._step(self._mm("A", "B")),
            self._step(self._mm("A", "B")),
            self._step(self._mm("A", "B", "C")),
        ]
        out = _delta_intermediate_mm_data(traj)

        assert out[1]["tokens"]["multi_modal_data"].mm_hashes == {"image": []}
        assert out[1]["tokens"]["multi_modal_data"].mm_items == {"image": []}
        assert out[2]["tokens"]["multi_modal_data"].mm_hashes == {"image": ["C"]}

    def test_non_mapping_steps_pass_through(self):
        traj = [self._step(self._mm("A")), "not-a-dict", self._step(self._mm("A", "B"))]
        out = _delta_intermediate_mm_data(traj)
        assert out[1] == "not-a-dict"
        # Delta of step 2 still computed against step 0 (last seen cumulative).
        assert out[2]["tokens"]["multi_modal_data"].mm_hashes == {"image": ["B"]}
