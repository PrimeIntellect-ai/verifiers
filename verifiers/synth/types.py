from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class EnvSpec:
    """Structured description of an Environment, extracted via introspection."""

    env_type: str
    system_prompt: str | None
    tools: list[dict[str, Any]] | None
    max_turns: int
    reward_functions: list[dict[str, Any]]
    dataset_schema: dict[str, str]
    example_rows: list[dict[str, Any]]
    parser_info: str | None
    few_shot: list[dict[str, Any]] | None


FilterMode = Literal["standard", "icl_calibrated"]


@dataclass
class SynthConfig:
    """Configuration for the synthetic data generation pipeline.

    filter_mode controls verification strategy:
      - "standard": keep samples a frontier model solves (score >= threshold).
        Use for general-knowledge tasks where reasoning, not context, is the
        bottleneck (e.g. math, code, logic).
      - "icl_calibrated": keep samples the model solves WITH seed context but
        fails WITHOUT it.  Use for out-of-distribution knowledge tasks where
        the answer depends on specific source material (e.g. retrieval, niche
        domain QA).
    """

    generator_model: str = "gpt-4.1"
    filter_model: str = "gpt-4.1"
    samples_per_subtopic: int = 5
    subtopic_branches: int = 3
    filter_mode: FilterMode = "standard"
    filter_threshold: float = 0.8
    filter_ceiling: float = 0.2


@dataclass
class SynthSample:
    """A single synthetic data sample produced by back-translation."""

    question: str
    answer: str
    info: dict[str, Any]
    seed_id: str
    subtopic: str
    score_with_context: float | None = None
    score_without_context: float | None = None

    def to_row(self) -> dict[str, Any]:
        """Convert to a dataset row compatible with Environment._format_dataset."""
        row: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
        }
        if self.info:
            row["info"] = json.dumps(self.info) if self.info else "{}"
        return row


@dataclass
class SynthPlan:
    """Output of the planning stage: seeds annotated with subtopics."""

    seeds: list[dict[str, Any]] = field(default_factory=list)
    total_target: int = 0


@dataclass
class BuildResult:
    """Final output of the synthetic data pipeline."""

    raw_samples: list[SynthSample]
    filtered_samples: list[SynthSample]
    stats: dict[str, Any]

    def save(self, output_dir: str = "./synth_output") -> None:
        """Write data.json and dataset_card.md to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        rows = [s.to_row() for s in self.filtered_samples]
        (out / "data.json").write_text(json.dumps(rows, indent=2))
        (out / "dataset_card.md").write_text(self._render_card())

    def _render_card(self) -> str:
        config = self.stats.get("config", {})
        mode = config.get("filter_mode", "standard")
        mode_label = (
            "ICL-calibrated (with vs. without context)"
            if mode == "icl_calibrated"
            else "Standard (frontier-solvable)"
        )
        lines = [
            "# Synthetic Dataset Card",
            "",
            "## Statistics",
            "",
            f"- **Total generated**: {self.stats.get('total_generated', 0)}",
            f"- **Post-filter**: {self.stats.get('total_filtered', 0)}",
            f"- **Pass rate**: {self.stats.get('pass_rate', 0.0):.1%}",
            f"- **Filter mode**: {mode_label}",
            "",
        ]

        subtopic_stats = self.stats.get("per_subtopic", {})
        if subtopic_stats:
            lines.append("## Per-Subtopic Breakdown")
            lines.append("")
            lines.append("| Subtopic | Generated | Filtered |")
            lines.append("|----------|-----------|----------|")
            for name, counts in subtopic_stats.items():
                lines.append(
                    f"| {name} | {counts.get('generated', 0)} | {counts.get('filtered', 0)} |"
                )
            lines.append("")

        config = self.stats.get("config", {})
        if config:
            lines.append("## Generation Config")
            lines.append("")
            for k, v in config.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        return "\n".join(lines)
