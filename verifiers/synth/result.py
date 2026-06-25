from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verifiers.synth.models import SynthSample


@dataclass
class BuildResult:
    """Final output of the synthetic data pipeline."""

    raw_samples: list[SynthSample]
    filtered_samples: list[SynthSample]
    stats: dict[str, Any]

    @property
    def coverage_failures(self) -> list[str]:
        """Subtopics where learnability rate is below coverage_quality."""
        coverage = self.stats.get("coverage", {})
        threshold = self.stats.get("config", {}).get("coverage_quality", 0.8)
        return [name for name, info in coverage.items() if info["rate"] < threshold]

    def save(self, output_dir: str = "./synth_output") -> None:
        """Write data.json and dataset_card.md to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        rows = [s.to_row() for s in self.filtered_samples]
        (out / "data.json").write_text(json.dumps(rows, indent=2))
        (out / "dataset_card.md").write_text(self._render_card())

    def _render_card(self) -> str:
        config = self.stats.get("config", {})
        ceiling = config.get("filter_ceiling")
        mode_label = (
            f"ICL-calibrated (threshold={config.get('filter_threshold')}, "
            f"ceiling={ceiling})"
            if ceiling is not None
            else f"Learnability only (threshold={config.get('filter_threshold')})"
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
            lines.append("| Subtopic | Generated | Filtered | Coverage |")
            lines.append("|----------|-----------|----------|----------|")
            coverage = self.stats.get("coverage", {})
            for name, counts in subtopic_stats.items():
                rate = coverage.get(name, {}).get("rate", 0.0)
                lines.append(
                    f"| {name} | {counts.get('generated', 0)} "
                    f"| {counts.get('filtered', 0)} | {rate:.0%} |"
                )
            lines.append("")

        failures = self.coverage_failures
        if failures:
            lines.append("## Coverage Failures")
            lines.append("")
            for name in failures:
                lines.append(f"- **{name}**")
            lines.append("")

        if config:
            lines.append("## Generation Config")
            lines.append("")
            for k, v in config.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        return "\n".join(lines)
