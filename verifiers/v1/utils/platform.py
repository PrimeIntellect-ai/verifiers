"""Push a finished eval run to the Prime Intellect platform (`--no-push` to skip).

Uploads one sample per v1 `Episode` over the `/evaluations/` API (create -> push
samples -> finalize). Each sample keeps the complete native Episode as its source
of truth and includes a flat summary for older Platform consumers. Auth + base URL
come from `$PRIME_API_KEY` / `~/.prime/config.json`.

Credentials stay out of the upload two ways. The config fields that hold them (client
headers, harness env) are dropped from the projection, and every credential value the
run knows — the clients' API keys, credential-named values from client headers, harness
and task environments and the host environment, whatever each rollout recorded on
`Trace.upload_secrets` — is replaced with `[REDACTED]` in every outbound request body.
Redaction is exact-match only: nothing is guessed from the shape of the text, so
ordinary content is never rewritten. Saved traces are unchanged, so a resumed `--push`
redacts everything except the values only the earlier attempt's rollouts knew: their
interception and runtime tokens (dead with that run) and credentials rotated since.
"""

import json
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.client import resolve_api_key
from verifiers.v1.episode import EPISODE_EXCLUDE_FIELDS, Episode
from verifiers.v1.trace import EXCLUDE_FIELDS, Trace
from verifiers.v1.utils.prime import load_prime_config
from verifiers.v1.utils.redact import (
    MIN_SECRET_LENGTH,
    REDACTED,
    Redactor,
    env_credentials,
    url_credentials,
)

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.primeintellect.ai"
DEFAULT_FRONTEND_URL = "https://app.primeintellect.ai"
# Repeated /samples posts append; match the Prime Evals client's request ceiling.
_MAX_SAMPLES_PAYLOAD_BYTES = 25 * 1024 * 1024
_FRAME_BYTES = len('{"samples":[]}')

UPLOAD_EXCLUDE = {
    **EPISODE_EXCLUDE_FIELDS,
    "traces": {
        "__all__": {
            **EXCLUDE_FIELDS,
            "agent": {"config": {"client": {"headers"}, "harness": {"env"}}},
        }
    },
}
"""The episode projection uploaded: the disk record minus the config fields that carry
credentials (`harness.forward_env` names variables without their values and stays)."""

CREDENTIAL_MAPPING = re.compile(r"(?:^|_)(?:env|headers)$")
"""Config and task-data fields holding an environment or header mapping (a harness
`env`, Harbor's `verifier_env`, a client's or a task config's `headers`). Name-based
discovery stays inside these: applied to every field it would take `api_key_var`'s
value, the *name* of a variable, for a credential, and keeping it out would need the
reference-suffix lists this design avoids. A credential stored under a bare task-data
field (`api_key: ...`) is recognised only through its value's URL shape."""


def strings(value: Any) -> Iterator[str]:
    """Every string in a JSON tree."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def credential_mappings(value: Any, name: str = "") -> Iterator[dict]:
    """Environment and header mappings anywhere in a JSON tree, nested ones included
    (a harness config's `env`, Harbor's `verifier.env`, a task config's `headers`)."""
    if isinstance(value, dict):
        if CREDENTIAL_MAPPING.search(name):
            yield value
        for key, child in value.items():
            yield from credential_mappings(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from credential_mappings(child, name)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def known_secrets(
    episodes: list[Episode], config: EvalConfig, *values: str
) -> set[str]:
    """Every credential this run could have put into a trace: the clients' API keys;
    the credentials in the host environment and in every environment or header mapping
    of the run's env config, the traced agent configs, and the trace and episode task
    data (`env_credentials`); URL credentials anywhere in those configs and task data (a
    client `base_url`, a harness endpoint, a task's connection string); what each
    rollout recorded on
    `Trace.upload_secrets` as it was then (discarded attempts' on
    `Episode.upload_secrets`); and `values`. Values shorter than `MIN_SECRET_LENGTH` are
    dropped — redacting them would rewrite ordinary text — and so are placeholders that
    sit inside the `[REDACTED]` marker."""
    traces = [trace for episode in episodes for trace in episode.traces]
    clients = [
        config.client,
        *(t.agent.config.client for t in traces if t.agent.config.client is not None),
    ]
    dumps = [
        config.client.model_dump(mode="json"),
        config.env.model_dump(mode="json"),
        *(trace.agent.config.model_dump(mode="json") for trace in traces),
        *(trace.task.data.model_dump(mode="json") for trace in traces),
        *(episode.task.data.model_dump(mode="json") for episode in episodes),
    ]
    named = [
        os.environ,
        *(mapping for dump in dumps for mapping in credential_mappings(dump)),
    ]
    secrets = {
        *values,
        *(resolve_api_key(client) for client in clients),
        *(credential for mapping in named for credential in env_credentials(mapping)),
        *(
            credential
            for dump in dumps
            for text in strings(dump)
            for credential in url_credentials(text)
        ),
        *(secret for trace in traces for secret in trace.upload_secrets),
        *(secret for episode in episodes for secret in episode.upload_secrets),
    }
    # A value inside the marker (`API_TOKEN=REDACTED`) is a sanitized placeholder; one
    # that merely contains or borders the marker is a credential and stays.
    return {
        secret
        for secret in secrets
        if len(secret) >= MIN_SECRET_LENGTH and secret not in REDACTED
    }


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


def build_samples(episodes: list[Episode], redactor: Redactor) -> list[str]:
    """One Platform sample per Episode, serialized and redacted, with a
    legacy-compatible trace summary.

    The Episode projection in `info.native_wrapper` is authoritative and contains
    every trace. One trainable trace (or the first trace) supplies only the flat
    summary used by older consumers. `native_trace_index` identifies that summary trace.
    """
    counts: dict[int, int] = {}
    rows = []
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
                mode="json", exclude=UPLOAD_EXCLUDE, exclude_none=True
            ),
            "native_trace_index": summary_trace_index,
        }
        row = redactor.json(json_text(sample))
        if _FRAME_BYTES + len(row.encode()) <= _MAX_SAMPLES_PAYLOAD_BYTES:
            rows.append(row)
            continue
        logger.warning(
            "Episode %s exceeds the Platform sample limit; uploading projected traces",
            episode.id,
        )
        rows.extend(
            redactor.json(json_text(trace_to_sample(candidate, number, episode.id)))
            for candidate in episode.traces
        )
    return rows


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
        "run_id": config.run.id,
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
        redactor = Redactor(known_secrets(episodes, config, api_key))
        rows = build_samples(episodes, redactor)
        if redactor.count:
            logger.warning(
                "--push: redacted %d occurrence(s) of known secrets from the upload; "
                "saved traces are unchanged",
                redactor.count,
            )
        # Batch by exact request size; each body is `{"samples":[<row>,<row>,...]}`.
        batches: list[list[str]] = [[]]
        size = _FRAME_BYTES
        for i, row in enumerate(rows):
            row_bytes = len(row.encode())
            if _FRAME_BYTES + row_bytes > _MAX_SAMPLES_PAYLOAD_BYTES:
                raise ValueError(
                    f"sample {i} is too large to upload "
                    f"({_FRAME_BYTES + row_bytes} > {_MAX_SAMPLES_PAYLOAD_BYTES} bytes)"
                )
            if batches[-1] and size + 1 + row_bytes > _MAX_SAMPLES_PAYLOAD_BYTES:
                batches.append([])
                size = _FRAME_BYTES
            size += row_bytes + (1 if batches[-1] else 0)
            batches[-1].append(row)

        with httpx.Client(headers=headers, timeout=300.0) as client:

            def post(path: str, body: dict) -> dict:
                resp = client.post(
                    f"{api}{path}", content=redactor.json(json_text(body)).encode()
                )
                resp.raise_for_status()
                return resp.json()

            env_id = post("/environmentshub/resolve", {"name": env_name, **team})[
                "data"
            ]["id"]
            eval_id = post(
                "/evaluations/",
                {
                    "name": config.run.name,
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
                resp = client.post(
                    f"{api}/evaluations/{eval_id}/samples",
                    content=f'{{"samples":[{",".join(batch)}]}}'.encode(),
                )
                resp.raise_for_status()
            post(f"/evaluations/{eval_id}/finalize", {"metrics": metrics})
    except Exception as e:  # noqa: BLE001 - push is best-effort across the full upload
        logger.warning("--push: upload failed (%s: %s); skipping", type(e).__name__, e)
        return finish(error=f"{type(e).__name__}: {e}")

    url = f"{frontend}/dashboard/evaluations/{eval_id}"
    logger.info("--push: uploaded %d samples -> %s", len(rows), url)
    return finish(url=url)
