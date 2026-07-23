"""Push a finished eval run to the Prime Intellect platform (`--no-push` to skip).

Converts each v1 `Trace` to the platform's (v0) eval-sample schema and uploads the
run over the `/evaluations/` API (create -> push samples -> finalize) — the same
contract as `prime eval push`, done inline at the end of a run. Auth + base URL
come from `$PRIME_API_KEY` / `~/.prime/config.json`.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from verifiers.utils.client_utils import load_prime_config
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.episode import Episode
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.primeintellect.ai"
DEFAULT_FRONTEND_URL = "https://app.primeintellect.ai"
# Repeated /samples posts append; match the Prime Evals client's request ceiling.
_MAX_SAMPLES_PAYLOAD_BYTES = 25 * 1024 * 1024


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
        "agent": trace.agent_name,
        "trainable": trace.trainable,
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
        "error": trace.error.model_dump(mode="json", exclude_none=True)
        if trace.error
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
    # Flatten sub-rewards to top-level keys the way v0 does; env metrics stay nested.
    for name, value in trace.rewards.items():
        sample.setdefault(name, value)
    return sample


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


def _run_metrics(episodes: list[Episode], traces: list[Trace]) -> dict[str, Any]:
    """Run-level aggregates as v0's `GenerateMetadata`. Rewards/metrics aggregate
    over the trainable traces only — fixed agents (a judge, a modeled user) often
    carry no rewards and would dilute every mean with structural zeros — falling
    back to all traces when none are trainable (same rule as the dashboard).
    `avg_error` is the share of EPISODES that aren't ok: a hook failure counts
    even when its traces are clean or it left none."""
    scored = [t for t in traces if t.trainable] or traces
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for trace in scored:
        for name, value in {**trace.rewards, **trace.metrics}.items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    n = len(scored)
    avg_error = sum(not e.ok for e in episodes) / len(episodes) if episodes else 0.0
    return {
        "avg_reward": sum(t.reward for t in scored) / n if n else 0.0,
        "avg_metrics": {name: sums[name] / counts[name] for name in sums},
        "avg_error": avg_error,
    }


def _build_samples(episodes: list[Episode]) -> list[dict[str, Any]]:
    """One platform sample per trace, with one `rollout_number` per EPISODE: a
    multi-agent rollout's seats are the same attempt at the task, not attempts
    1..n. The grouping is the episode envelope — every trace in one episode shares
    its `rollout_number` and `episode_id`."""
    counts: dict[int, int] = {}
    episode_numbers: dict[str, int] = {}
    samples = []
    for episode in episodes:
        # One rollout_number per episode: all its seats are the same attempt.
        number = episode_numbers.get(episode.id)
        for trace in episode.traces:
            if number is None:
                idx = trace.task.data.idx
                counts[idx] = number = counts.get(idx, 0) + 1
                episode_numbers[episode.id] = number
            samples.append(trace_to_sample(trace, number, episode.id))
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

    api_key, base, frontend, team_id = _creds()
    if not api_key:
        logger.warning(
            "--push: no PRIME_API_KEY (set it or run `prime login`); skipping upload"
        )
        return finish(error="no PRIME_API_KEY (run `prime login`)")

    traces = [trace for episode in episodes for trace in episode.traces]
    env_name = (config.env.taskset.id) or config.id
    metrics = _run_metrics(episodes, traces)
    samples = _build_samples(episodes)
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
        batches: list[list[dict[str, Any]]] = []
        batch: list[dict[str, Any]] = []
        payload_bytes = len(b'{"samples":[]}')
        for i, sample in enumerate(samples):
            sample_bytes = len(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
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
    except Exception as e:
        logger.warning("--push: upload failed (%s: %s); skipping", type(e).__name__, e)
        return finish(error=f"{type(e).__name__}: {e}")

    url = f"{frontend}/dashboard/evaluations/{eval_id}"
    logger.info("--push: uploaded %d samples -> %s", len(samples), url)
    return finish(url=url)
