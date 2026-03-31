from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvSpec:
    """Structured description of an Environment, extracted via introspection."""

    env_type: str
    system_prompt: str | None
    tools: list[dict[str, Any]] | None
    max_turns: int
    reward_functions: list[dict[str, Any]]
    dataset_schema: dict[str, Any]
    parser_info: str | None
    few_shot: list[dict[str, Any]] | None


@dataclass
class SynthConfig:
    """Configuration for the synthetic data generation pipeline.

    Filtering uses two thresholds on a single code path:
      - filter_threshold: minimum learnability score (with context). Samples
        scoring below this are dropped.
      - filter_ceiling: maximum novelty floor (without context). When set,
        samples the model can already solve WITHOUT context are dropped. Set to
        None to skip the novelty check (useful for general-knowledge tasks like
        math/code where reasoning, not context, is the bottleneck).
    """

    generator_model: str = "gpt-5.4-mini"
    filter_model: str = "gpt-5.4-mini"
    max_seed_examples: int = 3
    samples_per_subtopic: int = 5
    max_subtopics: int | None = None
    filter_threshold: float = 0.8
    filter_ceiling: float | None = None
    coverage_quality: float = 0.8


@dataclass
class SynthSample:
    """A single synthetic row matching the planned output schema."""

    row: dict[str, Any]
    subtopic: str
    score_with_context: float | None = None
    score_without_context: float | None = None

    def to_row(self) -> dict[str, Any]:
        """Dataset row with exactly the keys from generation (no extra metadata)."""
        return dict(self.row)


@dataclass
class SynthPlan:
    """Output of planning: subtopics plus filtering/generation guidance."""

    subtopics: list[str] = field(default_factory=list)
    total_target: int = 0
    generation_guidance: str | None = None
    task_field: str = "question"
    answer_field: str = "answer"
    reference_material: str | None = None
    """JSON or text passed to the generator and filter as bounded seed context."""
