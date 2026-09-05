"""Push a finished eval run to the Prime Intellect platform (`--no-push` to skip).

Uploads one sample per v1 `Episode` over the `/evaluations/` API (create -> push
samples -> finalize). Each sample keeps a reviewable native Episode projection and
a flat summary for older Platform consumers. Run-local configuration remains in the
saved output directory. Auth + base URL come from `$PRIME_API_KEY` /
`~/.prime/config.json`.
"""

import logging
from pathlib import Path
from typing import Any

from prime_evals import (
    MAX_SAMPLES_PAYLOAD_BYTES,
    APIClient,
    Config,
    CreateEvaluationRequest,
    EvalsClient,
    prepare_upload,
    secret_values,
    serialize_json,
)
from pydantic import BaseModel

from verifiers.v1.cli.output import read_upload_secret_fingerprints
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.client import resolve_api_key, resolve_headers
from verifiers.v1.episode import Episode
from verifiers.v1.trace import EXCLUDE_FIELDS, Trace
from verifiers.v1.types import Messages

logger = logging.getLogger(__name__)

PROVIDER_STATE_FIELDS = {
    "encrypted_content",
    "signature",
    "data",
}
PLATFORM_TRACE_EXCLUDE = {
    "upload_secrets": True,
    "agent": {
        "config": {
            "client": {"headers"},
            "harness": {"env", "skills"},
        },
        "runtime": {"id"},
    },
    "nodes": {
        "__all__": {
            **{field: True for field in EXCLUDE_FIELDS["nodes"]["__all__"]},
            **{
                field: True
                for field in (
                    "token_ids",
                    "mask",
                    "is_content",
                    "logprobs",
                    "advantages",
                    "reference_logprobs",
                    "loss_weights",
                )
            },
            "message": {"provider_state": {"__all__": PROVIDER_STATE_FIELDS}},
        }
    },
    "mm_token_type_id_map": True,
}


class PushState(BaseModel):
    """Mutable upload status shared with the dashboard."""

    started: bool = False
    done: bool = False
    url: str | None = None
    error: str | None = None


def dump_messages(messages: Messages) -> list[dict[str, Any]]:
    return [
        message.model_dump(
            mode="json",
            exclude={"provider_state": {"__all__": PROVIDER_STATE_FIELDS}},
            exclude_none=True,
        )
        for message in messages
    ]


def finish_push(
    state: PushState | None,
    url: str | None = None,
    error: str | None = None,
) -> str | None:
    if state is not None:
        state.url = url
        state.error = error
        state.done = True
    return url


def trace_to_sample(
    trace: Trace,
    rollout_number: int = 1,
    episode_id: str | None = None,
) -> dict[str, Any]:
    """One trace -> the platform's sample dict (the v0 eval-sample format).

    The hub table stays flat — one row per trace; its episode is denormalized onto
    the row (`episode_id` from the envelope, plus the trace's own `agent`/`trainable`),
    so a multi-trace rollout's grouping travels with each row without a nested
    schema. No prompt/completion split (meaningless mid-branch): `completion` is the
    final branch's messages, `trajectory` one message list per branch."""

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
        "completion": dump_messages(branches[-1].messages) if branches else [],
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
                "messages": dump_messages(branch.messages),
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

    The Episode projection in `info.native_wrapper` contains every trace's review
    data, without transport, runtime identity, opaque continuation, or per-token
    training state. One trainable trace (or the first trace) supplies the flat
    summary used by older consumers. `native_trace_index` identifies that trace.
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
            "native_wrapper": episode.model_dump(
                mode="json",
                exclude={
                    "upload_secrets": True,
                    "traces": {"__all__": PLATFORM_TRACE_EXCLUDE},
                },
                exclude_none=True,
            ),
            "native_trace_index": summary_trace_index,
        }
        if len(serialize_json({"samples": [sample]})) <= MAX_SAMPLES_PAYLOAD_BYTES:
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
    results_dir: Path | None = None,
) -> str | None:
    """Upload a finished run to the platform; return the viewer URL (None if
    skipped/failed). Resolves the env by name (get-or-create, so a local run
    uploads without a prior `prime env push`); when `state` is given, records the
    outcome on it so the dashboard's status line resolves."""

    prime_config = Config()
    api_key = prime_config.api_key
    if not api_key:
        logger.warning(
            "--push: no PRIME_API_KEY (set it or run `prime login`); skipping upload"
        )
        return finish_push(state, error="no PRIME_API_KEY (run `prime login`)")

    traces = [trace for episode in episodes for trace in episode.traces]
    env_name = config.env.taskset.id
    metrics = run_metrics(episodes, traces)
    num_examples = len({t.task.data.idx for t in traces})
    metadata = {
        "framework": "verifiers",
        "run_id": config.run.id,
        "model": config.model,
        "num_examples": num_examples,
        "rollouts_per_example": config.num_rollouts,
        **metrics,
    }

    # The run is done and its results saved; a network blip here must not crash it
    # — log and skip the upload instead.
    try:
        clients = [config.client]
        known_secrets = list(
            secret_values(
                api_key,
                *(secret for episode in episodes for secret in episode.upload_secrets),
            )
        )
        secret_sources = []
        for trace in traces:
            agent = trace.agent.config
            known_secrets.extend(trace.upload_secrets)
            if agent.client is not None:
                clients.append(agent.client)
            if agent.harness is not None:
                secret_sources.append(agent.harness.resolved_env)
        for client in clients:
            known_secrets.append(resolve_api_key(client))
            secret_sources.append(resolve_headers(client))

        saved_episode_ids = {episode.id for episode in episodes if episode.traces}
        secret_fingerprints = (
            read_upload_secret_fingerprints(results_dir, saved_episode_ids)
            if results_dir is not None
            else ()
        )

        samples = build_samples(episodes)
        request = CreateEvaluationRequest(
            name=config.run.name,
            environments=[{"name": env_name}],
            model_name=config.model,
            dataset=env_name,
            framework="verifiers",
            metadata=metadata,
            metrics=metrics,
        )
        prepared = prepare_upload(
            {
                "request": request.model_dump(mode="json"),
                "samples": samples,
            },
            known_secrets,
            secret_sources,
            secret_fingerprints,
        )
        payload = prepared.data
        redactions = prepared.report.locations
        request = CreateEvaluationRequest.model_validate(payload["request"])
        samples = payload["samples"]
        if redactions:
            logger.warning(
                "--push: preflight redacted %d credential-bearing value(s); "
                "saved traces were not changed",
                redactions,
            )
        with APIClient(config=prime_config) as api_client:
            eval_id = EvalsClient(api_client).push_evaluation(
                request,
                samples,
                max_payload_bytes=MAX_SAMPLES_PAYLOAD_BYTES,
            )
    except Exception as e:  # noqa: BLE001 - push is best-effort across the full upload
        logger.warning("--push: upload failed (%s: %s); skipping", type(e).__name__, e)
        return finish_push(state, error=f"{type(e).__name__}: {e}")

    url = f"{prime_config.frontend_url.rstrip('/')}/dashboard/evaluations/{eval_id}"
    logger.info("--push: uploaded %d samples -> %s", len(samples), url)
    return finish_push(state, url=url)
