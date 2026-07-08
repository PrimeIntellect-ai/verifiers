"""Push a finished eval run to the Prime Intellect platform (`uv run eval --push`).

Off by default. Converts each in-memory v1 `Trace` to the platform's sample schema
(`verifiers.v1.samples.trace_to_sample`, shared with the prime-rl monitor) and uploads the
run over the `/evaluations/` API (create -> push samples -> finalize) — the same contract
`prime eval push` uploads a saved run through, done inline at the end of a run rather than
later from disk. Auth + base URL come from `$PRIME_API_KEY` / `~/.prime/config.json`
(written by `prime login`), like the rest of the CLI.
"""

import logging
import os
from typing import Any

import httpx

from verifiers.utils.client_utils import load_prime_config
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.samples import trace_to_sample
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.primeintellect.ai"
DEFAULT_FRONTEND_URL = "https://app.primeintellect.ai"


def _creds() -> tuple[str | None, str, str, str | None]:
    """(api_key, api_base, frontend_url, team_id) from env vars / `~/.prime/config.json`."""
    cfg = load_prime_config()
    api_key = os.getenv("PRIME_API_KEY") or cfg.get("api_key")
    base = (
        os.getenv("PRIME_API_BASE_URL")
        or os.getenv("PRIME_BASE_URL")
        or cfg.get("base_url")
        or DEFAULT_API_URL
    )
    base = base.rstrip("/").removesuffix("/api/v1")
    frontend = (
        os.getenv("PRIME_FRONTEND_URL")
        or cfg.get("frontend_url")
        or DEFAULT_FRONTEND_URL
    )
    team_id = os.getenv("PRIME_TEAM_ID") or cfg.get("team_id")
    return api_key, base, frontend, team_id


def push_traces(traces: list[Trace], config: EvalConfig) -> str | None:
    """Upload a finished run to the platform; return the viewer URL (None if skipped).

    Resolves the env (get-or-create) by name so a local run uploads without a prior
    `prime env push`, then create evaluation -> push samples -> finalize."""
    api_key, base, frontend, team_id = _creds()
    if not api_key:
        logger.warning(
            "--push: no PRIME_API_KEY (set it or run `prime login`); skipping upload"
        )
        return None

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

    env_name = config.taskset.id or config.id
    metrics = compute_metrics()
    counts: dict[int, int] = {}
    samples = []
    for trace in traces:
        counts[trace.task.idx] = counts.get(trace.task.idx, 0) + 1
        samples.append(trace_to_sample(trace, counts[trace.task.idx]))

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

            env_id = post("/environmentshub/resolve", {"name": env_name, **team})[
                "data"
            ]["id"]
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
        return None

    url = f"{frontend}/dashboard/evaluations/{eval_id}"
    logger.info("--push: uploaded %d samples (evaluation_id=%s)", len(samples), eval_id)
    return url
