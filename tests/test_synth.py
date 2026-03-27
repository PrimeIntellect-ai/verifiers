"""Tests for the verifiers.synth module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset

from verifiers import Rubric, SingleTurnEnv, ToolEnv
from verifiers.synth.builder import (
    SynthDataBuilder,
    _normalize_dataset_seeds,
    _parse_json_array,
    _parse_json_object,
    _parse_model,
)
from verifiers.synth.prompts import render_env_spec
from verifiers.synth.types import BuildResult, SynthConfig, SynthSample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_dataset():
    return Dataset.from_dict(
        {
            "question": ["What is 2+2?", "Name the capital of France."],
            "answer": ["4", "Paris"],
        }
    )


@pytest.fixture
def simple_env(simple_dataset):
    async def correct_answer(completion, answer) -> float:
        """Check if the answer appears in the completion."""
        response = completion[-1]["content"]
        return 1.0 if answer in response else 0.0

    rubric = Rubric(funcs=[correct_answer])
    return SingleTurnEnv(
        dataset=simple_dataset,
        system_prompt="You are a helpful assistant.",
        rubric=rubric,
    )


@pytest.fixture
def tool_env_instance(simple_dataset):
    async def calculator(expression: str) -> str:
        """Evaluate a math expression.

        Args:
            expression: A math expression like '2+2'.

        Returns:
            The result as a string.
        """
        return str(eval(expression))

    async def exact_match(completion, answer) -> float:
        return 1.0 if answer in completion[-1]["content"] else 0.0

    rubric = Rubric(funcs=[exact_match])
    return ToolEnv(
        dataset=simple_dataset,
        tools=[calculator],
        rubric=rubric,
        system_prompt="Use tools to solve math problems.",
        max_turns=5,
    )


# ---------------------------------------------------------------------------
# EnvSpec extraction
# ---------------------------------------------------------------------------


class TestExtractEnvSpec:
    def test_singleturn_env_spec(self, simple_env):
        builder = SynthDataBuilder(env=simple_env)
        spec = builder.env_spec

        assert spec.env_type == "SingleTurnEnv"
        assert spec.system_prompt == "You are a helpful assistant."
        assert spec.tools is None
        assert spec.max_turns == 1
        weighted = [rf for rf in spec.reward_functions if rf["weight"] > 0]
        assert len(weighted) == 1
        assert weighted[0]["name"] == "correct_answer"
        assert weighted[0]["doc"] is not None

    def test_tool_env_spec(self, tool_env_instance):
        builder = SynthDataBuilder(env=tool_env_instance)
        spec = builder.env_spec

        assert spec.env_type == "ToolEnv"
        assert spec.system_prompt == "Use tools to solve math problems."
        assert spec.tools is not None
        assert len(spec.tools) == 1
        assert spec.tools[0]["name"] == "calculator"
        assert spec.max_turns == 5

    def test_dataset_schema_extraction(self, simple_env):
        builder = SynthDataBuilder(env=simple_env)
        spec = builder.env_spec

        assert len(spec.example_rows) > 0
        assert "answer" in spec.dataset_schema

    def test_reward_function_metadata(self, simple_env):
        builder = SynthDataBuilder(env=simple_env)
        spec = builder.env_spec

        rf = spec.reward_functions[0]
        assert "name" in rf
        assert "doc" in rf
        assert "weight" in rf
        assert rf["weight"] == 1.0


# ---------------------------------------------------------------------------
# Seed resolution & normalization
# ---------------------------------------------------------------------------


class TestResolveSeeds:
    def test_explicit_seeds_take_priority(self, simple_env):
        builder = SynthDataBuilder(env=simple_env)
        result = builder._resolve_seeds(["explicit text"])
        assert len(result) == 1
        assert result[0]["content"] == "explicit text"

    def test_falls_back_to_env_dataset(self, simple_env):
        builder = SynthDataBuilder(env=simple_env)
        result = builder._resolve_seeds(None)
        assert len(result) > 0
        assert result[0]["content"] == "What is 2+2?"

    def test_raises_when_no_seeds_and_empty_dataset(self):
        async def dummy_reward(completion, answer) -> float:
            return 1.0

        from verifiers import Rubric, SingleTurnEnv

        empty_ds = Dataset.from_dict({"question": [], "answer": []})
        env = SingleTurnEnv(
            dataset=empty_ds,
            system_prompt="test",
            rubric=Rubric(funcs=[dummy_reward]),
        )
        builder = SynthDataBuilder(env=env)
        with pytest.raises(ValueError, match="No seeds provided"):
            builder._resolve_seeds(None)


class TestNormalizeSeeds:
    def test_raw_strings(self):
        seeds = ["This is some raw text", "Another piece of text"]
        result = SynthDataBuilder._normalize_seeds(seeds)

        assert len(result) == 2
        assert result[0]["content"] == "This is some raw text"
        assert result[1]["content"] == "Another piece of text"
        assert result[0]["id"] == "0"
        assert result[1]["id"] == "1"

    def test_file_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "seed.txt"
            p.write_text("File content here")

            result = SynthDataBuilder._normalize_seeds([str(p)])
            assert len(result) == 1
            assert result[0]["content"] == "File content here"

    def test_glob_expansion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                (Path(tmpdir) / f"doc_{i}.md").write_text(f"Document {i}")

            pattern = str(Path(tmpdir) / "*.md")
            result = SynthDataBuilder._normalize_seeds([pattern])
            assert len(result) == 3
            contents = sorted(r["content"] for r in result)
            assert contents == ["Document 0", "Document 1", "Document 2"]

    def test_hf_dataset(self):
        ds = Dataset.from_dict({"content": ["seed one", "seed two"]})
        result = SynthDataBuilder._normalize_seeds(ds)

        assert len(result) == 2
        assert result[0]["content"] == "seed one"

    def test_hf_dataset_fallback_column(self):
        ds = Dataset.from_dict({"text": ["alpha", "beta"]})
        result = _normalize_dataset_seeds(ds)

        assert len(result) == 2
        assert result[0]["content"] == "alpha"

    def test_mixed_seeds(self):
        ds = Dataset.from_dict({"content": ["from dataset"]})
        result = SynthDataBuilder._normalize_seeds(["raw text", ds])

        assert len(result) == 2
        assert result[0]["content"] == "raw text"
        assert result[1]["content"] == "from dataset"


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


class TestJsonParsing:
    def test_parse_json_array_clean(self):
        result = _parse_json_array('["topic A", "topic B", "topic C"]')
        assert result == ["topic A", "topic B", "topic C"]

    def test_parse_json_array_with_markdown(self):
        text = '```json\n["a", "b"]\n```'
        result = _parse_json_array(text)
        assert result == ["a", "b"]

    def test_parse_json_array_with_preamble(self):
        text = 'Here are the subtopics:\n["x", "y"]'
        result = _parse_json_array(text)
        assert result == ["x", "y"]

    def test_parse_json_array_fallback(self):
        result = _parse_json_array("this is not json", fallback_count=2)
        assert len(result) == 2
        assert all(s.startswith("subtopic_") for s in result)

    def test_parse_json_object_clean(self):
        result = _parse_json_object('{"question": "What?", "answer": "Yes"}')
        assert result == {"question": "What?", "answer": "Yes"}

    def test_parse_json_object_with_markdown(self):
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_object(text)
        assert result == {"key": "value"}

    def test_parse_json_object_with_preamble(self):
        text = 'Here is the task:\n{"question": "Q", "answer": "A"}'
        result = _parse_json_object(text)
        assert result == {"question": "Q", "answer": "A"}

    def test_parse_json_object_invalid(self):
        result = _parse_json_object("not json at all")
        assert result is None


# ---------------------------------------------------------------------------
# Model string parsing
# ---------------------------------------------------------------------------


class TestParseModel:
    def test_bare_model(self):
        provider, model = _parse_model("gpt-4.1")
        assert provider == "openai"
        assert model == "gpt-4.1"

    def test_provider_model(self):
        provider, model = _parse_model("openai/gpt-4.1-mini")
        assert provider == "openai"
        assert model == "gpt-4.1-mini"

    def test_unknown_provider(self):
        provider, model = _parse_model("custom-host/my-model")
        assert provider == "openai"
        assert model == "custom-host/my-model"

    def test_deepseek_provider(self):
        provider, model = _parse_model("deepseek/deepseek-chat")
        assert provider == "deepseek"
        assert model == "deepseek-chat"


# ---------------------------------------------------------------------------
# SynthSample
# ---------------------------------------------------------------------------


class TestSynthSample:
    def test_to_row_basic(self):
        sample = SynthSample(
            question="What is 1+1?",
            answer="2",
            info={"difficulty": "easy"},
            seed_id="0",
            subtopic="arithmetic",
        )
        row = sample.to_row()

        assert row["question"] == "What is 1+1?"
        assert row["answer"] == "2"
        assert "info" in row
        parsed_info = json.loads(row["info"])
        assert parsed_info["difficulty"] == "easy"

    def test_to_row_empty_info(self):
        sample = SynthSample(
            question="Q", answer="A", info={}, seed_id="0", subtopic="t"
        )
        row = sample.to_row()
        assert row["question"] == "Q"
        assert row["answer"] == "A"


# ---------------------------------------------------------------------------
# BuildResult
# ---------------------------------------------------------------------------


class TestBuildResult:
    def test_save_creates_files(self):
        samples = [
            SynthSample(
                question="Q1",
                answer="A1",
                info={"topic": "math"},
                seed_id="0",
                subtopic="algebra",
                score_with_context=0.9,
                score_without_context=0.1,
            )
        ]
        result = BuildResult(
            raw_samples=samples * 3,
            filtered_samples=samples,
            stats={
                "total_generated": 3,
                "total_filtered": 1,
                "pass_rate": 1 / 3,
                "per_subtopic": {"algebra": {"generated": 3, "filtered": 1}},
                "config": {
                    "generator_model": "gpt-4.1",
                    "filter_model": "gpt-4.1",
                },
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result.save(tmpdir)

            data_path = Path(tmpdir) / "data.json"
            card_path = Path(tmpdir) / "dataset_card.md"

            assert data_path.exists()
            assert card_path.exists()

            data = json.loads(data_path.read_text())
            assert len(data) == 1
            assert data[0]["question"] == "Q1"
            assert data[0]["answer"] == "A1"

            card = card_path.read_text()
            assert "Total generated" in card
            assert "Post-filter" in card
            assert "algebra" in card


# ---------------------------------------------------------------------------
# SynthConfig filter modes
# ---------------------------------------------------------------------------


class TestFilterMode:
    def test_default_is_standard(self):
        config = SynthConfig()
        assert config.filter_mode == "standard"

    def test_icl_calibrated_mode(self):
        config = SynthConfig(filter_mode="icl_calibrated")
        assert config.filter_mode == "icl_calibrated"
        assert config.filter_ceiling == 0.2

    def test_standard_mode_ignores_ceiling(self):
        """In standard mode, filter_ceiling exists but is unused by _filter."""
        config = SynthConfig(filter_mode="standard", filter_threshold=0.7)
        assert config.filter_threshold == 0.7

    def test_dataset_card_shows_filter_mode(self):
        result = BuildResult(
            raw_samples=[],
            filtered_samples=[],
            stats={
                "total_generated": 0,
                "total_filtered": 0,
                "pass_rate": 0.0,
                "config": {"filter_mode": "icl_calibrated"},
            },
        )
        card = result._render_card()
        assert "ICL-calibrated" in card

    def test_dataset_card_standard_mode(self):
        result = BuildResult(
            raw_samples=[],
            filtered_samples=[],
            stats={
                "total_generated": 0,
                "total_filtered": 0,
                "pass_rate": 0.0,
                "config": {"filter_mode": "standard"},
            },
        )
        card = result._render_card()
        assert "Standard" in card


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


class TestRenderEnvSpec:
    def test_basic_render(self, simple_env):
        builder = SynthDataBuilder(env=simple_env)
        from dataclasses import asdict

        text = render_env_spec(asdict(builder.env_spec))

        assert "SingleTurnEnv" in text
        assert "You are a helpful assistant." in text
        assert "correct_answer" in text

    def test_tool_env_render(self, tool_env_instance):
        builder = SynthDataBuilder(env=tool_env_instance)
        from dataclasses import asdict

        text = render_env_spec(asdict(builder.env_spec))

        assert "ToolEnv" in text
        assert "calculator" in text
        assert "expression" in text
