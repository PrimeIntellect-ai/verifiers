from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha1
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from jinja2 import BaseLoader, Environment, StrictUndefined, select_autoescape

from verifiers.types import GenerateOutputs, Messages

# Hard cap on the number of fully rendered examples in the HTML report.
DETAILED_EXAMPLES_CAP: int = 50


@dataclass
class ReportMeta:
    env_id: str
    env_version: str
    model: str
    num_examples: int
    rollouts_per_example: int
    api_base_url: str
    sampling_args: Dict[str, Any]
    env_args: Dict[str, Any]


def get_env_version(module_name: str) -> str:
    """Return installed package version for the environment module.

    Falls back to "0.0.0" if not installed as a package.
    """
    try:
        return importlib_metadata.version(module_name)
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0"


def _safe_last_assistant_text(messages: Messages) -> str:
    """Extract the last assistant message content if present; otherwise a short placeholder.

    Messages may be a list of dicts in chat format. Return a trimmed snippet.
    """
    try:
        if isinstance(messages, list) and messages:
            # iterate backwards to find last assistant
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "") or ""
                    return _trim_snippet(str(content))
        return ""
    except Exception:
        return ""


def _trim_snippet(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _compute_basic_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "n": int(arr.size),
    }


def _compute_percentiles(
    values: List[float], percentiles: Tuple[int, ...] = (5, 25, 50, 75, 95)
) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in percentiles}
    qs = np.percentile(arr, percentiles)
    return {f"p{p}": float(q) for p, q in zip(percentiles, qs)}


def compute_summary(results: GenerateOutputs) -> Dict[str, Any]:
    """Compute aggregated statistics from GenerateOutputs in a format usable by templates.

    This function intentionally does not change layout based on dataset size.
    """
    summary: Dict[str, Any] = {}

    reward_stats = _compute_basic_stats(results.reward)
    reward_percentiles = _compute_percentiles(results.reward)
    summary["reward"] = {**reward_stats, **reward_percentiles}

    metric_summaries: Dict[str, Dict[str, float]] = {}
    for metric_name, metric_values in results.metrics.items():
        metric_summaries[metric_name] = {
            **_compute_basic_stats(metric_values),
            **_compute_percentiles(metric_values),
        }
    summary["metrics"] = metric_summaries

    return summary


def build_examples(
    results: GenerateOutputs, cap: int = DETAILED_EXAMPLES_CAP
) -> List[Dict[str, Any]]:
    """Prepare a capped list of example rows for rendering.

    Each row contains: index, reward, optional first metric column, and a completion snippet.
    """
    num = min(len(results.reward), cap)
    metric_first_name = next(iter(results.metrics.keys()), None)
    rows: List[Dict[str, Any]] = []
    for i in range(num):
        completion_snippet = _safe_last_assistant_text(results.completion[i])
        row = {
            "index": i,
            "reward": float(results.reward[i]),
            "metric_name": metric_first_name,
            "metric_value": float(results.metrics[metric_first_name][i])
            if metric_first_name
            else None,
            "completion": completion_snippet,
        }
        rows.append(row)
    return rows


def _hash_env_args(env_args: Dict[str, Any]) -> str:
    if not env_args:
        return "noargs"
    # Stable JSON-like representation for hashing
    try:
        import json

        normalized = json.dumps(env_args, sort_keys=True, separators=(",", ":"))
    except Exception:
        normalized = str(sorted(env_args.items()))
    return sha1(normalized.encode("utf-8")).hexdigest()[:8]


def build_report_filename(meta: ReportMeta) -> str:
    """Construct a deterministic report file name without timestamps.

    Pattern: {env_id}--v{env_version}--model={model}--n={n}--r={r}--args={hash}.html
    """
    args_hash = _hash_env_args(meta.env_args)
    safe_model = meta.model.replace("/", "--")
    return (
        f"{meta.env_id}--v{meta.env_version}--model={safe_model}"
        f"--n={meta.num_examples}--r={meta.rollouts_per_example}--args={args_hash}.html"
    )


_TEMPLATE = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, sans-serif; margin: 24px; }
      h1, h2, h3 { margin: 0.4em 0; }
      table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; }
      th, td { border: 1px solid #ddd; padding: 8px; font-size: 14px; text-align: left; }
      th { background: #f6f8fa; }
      .meta { margin-bottom: 16px; color: #444; }
      .code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; background: #f6f8fa; padding: 2px 6px; border-radius: 4px; }
      .examples pre { white-space: pre-wrap; }
      .muted { color: #777; }
    </style>
  </head>
  <body>
    <h3>{{ env_id }}: {{ model }} (n={{ num_examples }}, r={{ rollouts_per_example }})</h3>
    <div class="meta">
      <div><b>Environment</b>: {{ env_id }} (v{{ env_version }})</div>
      <div><b>Model</b>: <span class="code">{{ model }}</span></div>
      <div><b>Provider</b>: {{ api_base_url }}</div>
      <div><b>Samples</b>: n={{ num_examples }}, r={{ rollouts_per_example }}</div>
      <div><b>Date</b>: {{ date }}</div>
      <div><b>Time</b>: {{ time }}</div>
      <div><b>Sampling</b>: max_tokens={{ sampling_args.max_tokens }}, temperature={{ sampling_args.temperature }}</div>
    </div>

    <h2>Reward</h2>
    <table>
      <tr><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th></tr>
      <tr>
        <td>{{ reward.mean | round(4) }}</td>
        <td>{{ reward.std | round(4) }}</td>
        <td>{{ reward.n }}</td>
        <td>{{ reward.p5 | round(4) }}</td>
        <td>{{ reward.p25 | round(4) }}</td>
        <td>{{ reward.p50 | round(4) }}</td>
        <td>{{ reward.p75 | round(4) }}</td>
        <td>{{ reward.p95 | round(4) }}</td>
      </tr>
    </table>

    {% if metrics %}
    <h2>Metrics</h2>
    <table>
      <tr>
        <th>metric</th><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th>
      </tr>
      {% for name, m in metrics.items() %}
      <tr>
        <td>{{ name }}</td>
        <td>{{ m.mean | round(4) }}</td>
        <td>{{ m.std | round(4) }}</td>
        <td>{{ m.n }}</td>
        <td>{{ m.p5 | round(4) }}</td>
        <td>{{ m.p25 | round(4) }}</td>
        <td>{{ m.p50 | round(4) }}</td>
        <td>{{ m.p75 | round(4) }}</td>
        <td>{{ m.p95 | round(4) }}</td>
      </tr>
      {% endfor %}
    </table>
    {% endif %}

    <h2>Examples <span class="muted">(showing up to {{ examples|length }} of {{ total_examples }})</span></h2>
    <div class="examples">
      <table>
        <tr><th>#</th><th>reward</th><th>{% if examples and examples[0].metric_name %}{{ examples[0].metric_name }}{% else %}metric{% endif %}</th><th>completion</th></tr>
        {% for ex in examples %}
        <tr>
          <td>{{ ex.index }}</td>
          <td>{{ ex.reward | round(4) }}</td>
          <td>{% if ex.metric_value is not none %}{{ ex.metric_value | round(4) }}{% else %}-{% endif %}</td>
          <td><pre>{{ ex.completion }}</pre></td>
        </tr>
        {% endfor %}
      </table>
    </div>
  </body>
 </html>
"""


_env = Environment(
    loader=BaseLoader(),
    undefined=StrictUndefined,
    autoescape=select_autoescape(["html", "xml"]),
)


def render_html(
    meta: ReportMeta,
    summary: Dict[str, Any],
    examples: List[Dict[str, Any]],
    total_examples: int,
) -> str:
    template = _env.from_string(_TEMPLATE)
    return template.render(
        env_id=meta.env_id,
        env_version=meta.env_version,
        model=meta.model,
        api_base_url=meta.api_base_url,
        num_examples=meta.num_examples,
        rollouts_per_example=meta.rollouts_per_example,
        date=datetime.now().strftime("%Y-%m-%d"),
        time=datetime.now().strftime("%H:%M:%S"),
        sampling_args={
            "max_tokens": meta.sampling_args.get("max_tokens"),
            "temperature": meta.sampling_args.get("temperature"),
        },
        reward=summary.get("reward", {}),
        metrics=summary.get("metrics", {}),
        examples=examples,
        total_examples=total_examples,
    )


def write_html_report(
    report_dir: Path,
    meta: ReportMeta,
    results: GenerateOutputs,
) -> Path:
    """Render and write the HTML report next to the environment under `reports/`.

    Returns the path to the written HTML file.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_summary(results)
    examples = build_examples(results, cap=DETAILED_EXAMPLES_CAP)
    html = render_html(
        meta=meta,
        summary=summary,
        examples=examples,
        total_examples=len(results.reward),
    )
    filename = build_report_filename(meta)
    out_path = report_dir / filename
    out_path.write_text(html, encoding="utf-8")
    return out_path
