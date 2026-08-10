"""Push a finished eval run to the Prime Intellect platform (`--no-push` to skip).

Uploads one sample per v1 `Episode` over the `/evaluations/` API (create -> push
samples -> finalize). Each sample keeps the complete native Episode as its source
of truth and includes a flat summary for older Platform consumers. Auth + base URL
come from `$PRIME_API_KEY` / `~/.prime/config.json`.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.episode import Episode
from verifiers.v1.trace import Trace
from verifiers.v1.utils.prime_config import load_prime_config

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.primeintellect.ai"
DEFAULT_FRONTEND_URL = "https://app.primeintellect.ai"
# Repeated /samples posts append; match the Prime Evals client's request ceiling.
_MAX_SAMPLES_PAYLOAD_BYTES = 25 * 1024 * 1024


def json_bytes(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


@dataclass
class PushState:
    """Mutable upload status shared with the dashboard."""

    started: bool = False
    done: bool = False
    url: str | None = None
    error: str | None = None


def trace_to_sample(
    trace: Trace, rollout_number: int = 1, episode_id: str | None = None
) -> dict[str, Any]:
    """One trace -> the platform's sample dict (the v0 eval-sample format).

    The hub table stays flat — one row per trace; its episode is denormalized onto
    the row (`episode_id` from the envelope, plus the trace's own `agent`/`trainable`),
    so a multi-trace rollout's grouping travels with each row without a nested
    schema. No prompt/completion split (meaningless mid-branch): `completion` is the
    final branch's messages, `trajectory` one message list per branch."""

    def dump(messages):
        return [m.model_dump(mode="json", exclude_none=True) for m in messages]

    task = trace.task.data.model_dump(mode="json", exclude_none=True)
    branches = trace.branches
    sample = {
        "sample_id": trace.id,
        "example_id": trace.task.data.idx,
        "rollout_number": rollout_number,
        "episode_id": episode_id,
        "agent": trace.agent.name,
        "trainable": trace.agent.trainable,
        "task": task,
        "prompt": [],
        "completion": dump(branches[-1].messages) if branches else [],
        "answer": task.get("answer"),
        # Keyed `tool_defs` because the v0 sample format already carries it there.
        "tool_defs": [t.model_dump(mode="json", exclude_none=True) for t in trace.tools]
        if trace.tools
        else None,
        "reward": trace.reward,
        "timing": trace.timing.model_dump(mode="json", exclude_none=True),
        "is_completed": trace.is_completed,
        "is_truncated": trace.is_truncated,
        "metrics": trace.metrics,
        "error": trace.last_error.model_dump(mode="json", exclude_none=True)
        if trace.last_error
        else None,
        "stop_condition": trace.stop_condition,
        "trajectory": [
            {
                "messages": dump(branch.messages),
                "num_input_tokens": branch.num_input_tokens,
                "num_output_tokens": branch.num_output_tokens,
            }
            for branch in branches
        ],
        "token_usage": trace.usage.model_dump(mode="json", exclude_none=True)
        if trace.usage
        else None,
        "info": dict(trace.info) or None,
    }
    # Flatten sub-rewards to top-level keys the way v0 does (raw scores, as v0's
    # per-function outputs were); env metrics stay nested.
    for name, reward in trace.rewards.items():
        if reward is not None:
            sample.setdefault(name, reward.score)
    return sample


def credentials() -> tuple[str | None, str, str, str | None]:
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


def run_metrics(episodes: list[Episode], traces: list[Trace]) -> dict[str, Any]:
    """Run-level aggregates as v0's `GenerateMetadata`. Rewards/metrics aggregate
    over the trainable traces only — fixed agents (a judge, a modeled user) often
    carry no rewards and would dilute every mean with structural zeros — falling
    back to all traces when none are trainable (same rule as the dashboard).
    `avg_error` is the share of EPISODES that aren't ok: a hook failure counts
    even when its traces are clean or it left none."""
    scored = [t for t in traces if t.agent.trainable] or traces
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for trace in scored:
        scores = {
            name: reward.score
            for name, reward in trace.rewards.items()
            if reward is not None
        }
        metrics = {
            name: value for name, value in trace.metrics.items() if value is not None
        }
        for name, value in {**scores, **metrics}.items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    n = len(scored)
    avg_error = sum(not e.ok for e in episodes) / len(episodes) if episodes else 0.0
    return {
        "avg_reward": sum(t.reward for t in scored) / n if n else 0.0,
        "avg_metrics": {name: sums[name] / counts[name] for name in sums},
        "avg_error": avg_error,
    }


def build_samples(episodes: list[Episode]) -> list[dict[str, Any]]:
    """One Platform sample per Episode, with a legacy-compatible trace summary.

    The native Episode in `info.native_wrapper` is authoritative and contains every
    trace. One trainable trace (or the first trace) supplies only the flat summary
    used by older consumers. `native_trace_index` identifies that summary trace.
    """
    counts: dict[int, int] = {}
    samples = []
    for episode in episodes:
        if not episode.traces:
            continue
        summary_trace_index = next(
            (
                index
                for index, candidate in enumerate(episode.traces)
                if candidate.agent.trainable
            ),
            0,
        )
        summary_trace = episode.traces[summary_trace_index]
        idx = summary_trace.task.data.idx
        counts[idx] = number = counts.get(idx, 0) + 1
        sample = trace_to_sample(summary_trace, number, episode.id)
        sample["sample_id"] = episode.id
        sample["info"] = {
            **(sample["info"] or {}),
            "native_wrapper": episode.to_record(),
            "native_trace_index": summary_trace_index,
        }
        if len(b'{"samples":[]}') + json_bytes(sample) <= _MAX_SAMPLES_PAYLOAD_BYTES:
            samples.append(sample)
            continue
        logger.warning(
            "Episode %s exceeds the Platform sample limit; uploading projected traces",
            episode.id,
        )
        samples.extend(
            trace_to_sample(candidate, number, episode.id)
            for candidate in episode.traces
        )
    return samples


def push_traces(
    episodes: list[Episode],
    config: EvalConfig,
    state: "PushState | None" = None,
) -> str | None:
    """Upload a finished run to the platform; return the viewer URL (None if
    skipped/failed). Resolves the env by name (get-or-create, so a local run
    uploads without a prior `prime env push`); when `state` is given, records the
    outcome on it so the dashboard's status line resolves."""

    def finish(url: str | None = None, error: str | None = None) -> str | None:
        if state is not None:
            state.url = url
            state.error = error
            state.done = True
        return url

    api_key, base, frontend, team_id = credentials()
    if not api_key:
        logger.warning(
            "--push: no PRIME_API_KEY (set it or run `prime login`); skipping upload"
        )
        return finish(error="no PRIME_API_KEY (run `prime login`)")

    traces = [trace for episode in episodes for trace in episode.traces]
    env_name = config.env.taskset.id
    metrics = run_metrics(episodes, traces)
    num_examples = len({t.task.data.idx for t in traces})
    metadata = {
        "framework": "verifiers",
        "run_id": config.uuid,
        "model": config.model,
        "num_examples": num_examples,
        "rollouts_per_example": config.num_rollouts,
        **metrics,
    }

    team = {"team_id": team_id} if team_id else {}
    api = f"{base}/api/v1"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # The run is done and its results saved; a network blip here must not crash it
    # — log and skip the upload instead.
    try:
        samples = build_samples(episodes)
        batches: list[list[dict[str, Any]]] = []
        batch: list[dict[str, Any]] = []
        payload_bytes = len(b'{"samples":[]}')
        for i, sample in enumerate(samples):
            sample_bytes = json_bytes(sample)
            sample_payload_bytes = len(b'{"samples":[]}') + sample_bytes
            if sample_payload_bytes > _MAX_SAMPLES_PAYLOAD_BYTES:
                raise ValueError(
                    f"sample {i} is too large to upload "
                    f"({sample_payload_bytes} > "
                    f"{_MAX_SAMPLES_PAYLOAD_BYTES} bytes)"
                )
            next_payload_bytes = payload_bytes + (1 if batch else 0) + sample_bytes
            if batch and next_payload_bytes > _MAX_SAMPLES_PAYLOAD_BYTES:
                batches.append(batch)
                batch = []
                payload_bytes = len(b'{"samples":[]}')
                next_payload_bytes = payload_bytes + sample_bytes
            batch.append(sample)
            payload_bytes = next_payload_bytes
        if batch or not samples:
            batches.append(batch)

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
            for batch in batches:
                body = json.dumps(
                    {"samples": batch},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                resp = client.post(
                    f"{api}/evaluations/{eval_id}/samples",
                    content=body,
                )
                resp.raise_for_status()
            post(f"/evaluations/{eval_id}/finalize", {"metrics": metrics})
    except Exception as e:  # noqa: BLE001 - push is best-effort across the full upload
        logger.warning("--push: upload failed (%s: %s); skipping", type(e).__name__, e)
        return finish(error=f"{type(e).__name__}: {e}")

    url = f"{frontend}/dashboard/evaluations/{eval_id}"
    logger.info("--push: uploaded %d samples -> %s", len(samples), url)
    return finish(url=url)
