"""Push a finished eval run to the Prime Intellect platform (`uv run eval`, `--no-push` to skip).

On by default. Converts each in-memory v1 `Trace` to the platform's (v0) eval-sample schema
(`trace_to_sample`) and uploads the run over the `/evaluations/` API (create -> push samples ->
finalize) — the same contract
`prime eval push` uploads a saved run through, done inline at the end of a run rather than
later from disk. Auth + base URL come from `$PRIME_API_KEY` / `~/.prime/config.json`
(written by `prime login`), like the rest of the CLI.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from verifiers.utils.client_utils import load_prime_config
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.primeintellect.ai"
DEFAULT_FRONTEND_URL = "https://app.primeintellect.ai"


@dataclass
class PushState:
    """Mutable upload status shared with the dashboard."""

    started: bool = False
    done: bool = False
    url: str | None = None
    error: str | None = None


def trace_to_sample(trace: Trace, rollout_number: int = 1) -> dict[str, Any]:
    """One rollout -> the platform's sample dict (the "old" v0 eval-sample format).

    The conversation is the unit — no prompt/completion split (meaningless mid-branch):
    `completion` is the final branch's messages and `trajectory` carries one message list per
    branch. `rollout_number` is this rollout's 1-based index within its task's group (callers that
    don't group rollouts can leave it at the default)."""

    def dump(messages):
        return [m.model_dump(mode="json", exclude_none=True) for m in messages]

    task = trace.task.data.model_dump(mode="json", exclude_none=True)
    branches = trace.branches
    sample = {
        "sample_id": trace.id,
        "example_id": trace.task.data.idx,
        "rollout_number": rollout_number,
        "task": task,
        "prompt": [],
        "completion": dump(branches[-1].messages) if branches else [],
        "answer": task.get("answer"),
        "reward": trace.reward,
        "timing": trace.timing.model_dump(mode="json", exclude_none=True),
        "is_completed": trace.is_completed,
        "is_truncated": trace.is_truncated,
        "metrics": trace.metrics,
        "error": trace.error.model_dump(mode="json", exclude_none=True) if trace.error else None,
        "stop_condition": trace.stop_condition,
        "trajectory": [
            {
                "messages": dump(branch.messages),
                "num_input_tokens": branch.num_input_tokens,
                "num_output_tokens": branch.num_output_tokens,
            }
            for branch in branches
        ],
        "token_usage": trace.usage.model_dump(mode="json", exclude_none=True) if trace.usage else None,
        "info": dict(trace.info) or None,
    }
    # Flatten each sub-reward onto the sample as a top-level key, the way v0's `state_to_output`
    # does. Env metrics stay in the nested `metrics` field.
    for name, value in trace.rewards.items():
        sample.setdefault(name, value)
    return sample


def _creds() -> tuple[str | None, str, str, str | None]:
    """(api_key, api_base, frontend_url, team_id) from env vars / `~/.prime/config.json`."""
    cfg = load_prime_config()
    api_key = os.getenv("PRIME_API_KEY") or cfg.get("api_key")
    base = os.getenv("PRIME_API_BASE_URL") or os.getenv("PRIME_BASE_URL") or cfg.get("base_url") or DEFAULT_API_URL
    base = base.rstrip("/").removesuffix("/api/v1")
    frontend = os.getenv("PRIME_FRONTEND_URL") or cfg.get("frontend_url") or DEFAULT_FRONTEND_URL
    team_id = os.getenv("PRIME_TEAM_ID") or cfg.get("team_id")
    return api_key, base, frontend, team_id


def push_traces(traces: list[Trace], config: EvalConfig, state: "PushState | None" = None) -> str | None:
    """Upload a finished run to the platform; return the viewer URL (None if skipped/failed).

    Resolves the env (get-or-create) by name so a local run uploads without a prior
    `prime env push`, then create evaluation -> push samples -> finalize. When `state` is given
    (the v1 `--rich` path), record the outcome on it (`url` on success, `error` on skip/failure,
    `done` when finished) so the dashboard's status line resolves."""

    def finish(url: str | None = None, error: str | None = None) -> str | None:
        if state is not None:
            state.url = url
            state.error = error
            state.done = True
        return url

    api_key, base, frontend, team_id = _creds()
    if not api_key:
        logger.warning("--push: no PRIME_API_KEY (set it or run `prime login`); skipping upload")
        return finish(error="no PRIME_API_KEY (run `prime login`)")

    def compute_metrics() -> dict[str, Any]:
        """Run-level aggregates as v0's `GenerateMetadata`: `avg_reward` (mean over all traces),
        `avg_metrics` (each sub-reward and env-metric averaged over the traces that recorded it),
        and `avg_error` (errored fraction) — what the overview renders."""
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for trace in traces:
            for name, value in {**trace.rewards, **trace.metrics}.items():
                sums[name] = sums.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + 1
        n = len(traces)
        return {
            "avg_reward": sum(t.reward for t in traces) / n if n else 0.0,
            "avg_metrics": {name: sums[name] / counts[name] for name in sums},
            "avg_error": sum(t.has_error for t in traces) / n if n else 0.0,
        }

    topology = getattr(config, "topology", None)
    env_name = topology.id if topology is not None else config.taskset.id or config.id
    metrics = compute_metrics()
    counts: dict[int, int] = {}
    samples = []
    for trace in traces:
        counts[trace.task.data.idx] = counts.get(trace.task.data.idx, 0) + 1
        samples.append(trace_to_sample(trace, counts[trace.task.data.idx]))

    metadata = {
        "framework": "verifiers",
        "run_id": config.uuid,
        "model": config.model,
        "num_examples": len(counts),
        "rollouts_per_example": config.num_rollouts,
        **metrics,
    }

    team = {"team_id": team_id} if team_id else {}
    api = f"{base}/api/v1"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # The run is already done and its results saved; a network blip or platform error here must
    # not crash the run (or, on the non-rich path, swallow the final per-trace output that prints
    # after this) — log and skip the upload instead.
    try:
        with httpx.Client(headers=headers, timeout=300.0) as client:

            def post(path: str, body: dict) -> dict:
                resp = client.post(f"{api}{path}", json=body)
                resp.raise_for_status()
                return resp.json()

            env_id = post("/environmentshub/resolve", {"name": env_name, **team})["data"]["id"]
            eval_id = post(
                "/evaluations/",
                {
                    "name": f"{env_name}--{config.model}--{config.uuid[:8]}",
                    "environments": [{"id": env_id}],
                    "model_name": config.model,
                    "dataset": env_name,
                    "framework": "verifiers",
                    "metadata": metadata,
                    "metrics": metrics,
                    "tags": [],
                    **team,
                },
            )["evaluation_id"]
            post(f"/evaluations/{eval_id}/samples", {"samples": samples})
            post(f"/evaluations/{eval_id}/finalize", {"metrics": metrics})
    except Exception as e:
        logger.warning("--push: upload failed (%s: %s); skipping", type(e).__name__, e)
        return finish(error=f"{type(e).__name__}: {e}")

    url = f"{frontend}/dashboard/evaluations/{eval_id}"
    logger.info("--push: uploaded %d samples -> %s", len(samples), url)
    return finish(url=url)
