"""Push a finished eval run to the Prime Intellect platform (`--no-push` to skip).

Uploads one sample per v1 `Episode` over the `/evaluations/` API (create -> push
samples -> finalize). Each sample keeps the complete native Episode as its source
of truth and includes a flat summary for older Platform consumers. Auth + base URL
come from `$PRIME_API_KEY` / `~/.prime/config.json`.

Credentials stay out of the upload two ways. The config fields that hold them (client
headers, harness env) are dropped from the projection, and every credential value the
run knows — the clients' API keys, credential-named header / harness / host environment
values, the rollouts' interception tokens — is replaced with `[REDACTED]` wherever it
appears in the serialized samples. Redaction is exact-match only: nothing is guessed
from the shape of the text, so ordinary content is never rewritten. Saved traces are
unchanged, so a resumed `--push` redacts everything except the earlier attempt's
interception tokens, which died with its interception server.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.client import resolve_api_key
from verifiers.v1.episode import Episode
from verifiers.v1.trace import EXCLUDE_FIELDS, Trace
from verifiers.v1.utils.prime import load_prime_config

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.primeintellect.ai"
DEFAULT_FRONTEND_URL = "https://app.primeintellect.ai"
# Repeated /samples posts append; match the Prime Evals client's request ceiling.
_MAX_SAMPLES_PAYLOAD_BYTES = 25 * 1024 * 1024
_FRAME_BYTES = len('{"samples":[]}')

UPLOAD_EXCLUDE = {
    "traces": {
        "__all__": {
            **EXCLUDE_FIELDS,
            "agent": {"config": {"client": {"headers"}, "harness": {"env"}}},
        }
    }
}
"""The episode projection uploaded: the disk record minus the config fields that carry
credentials (`harness.forward_env` names variables without their values and stays)."""

REDACTED = "[REDACTED]"
MIN_SECRET_LENGTH = 8
SECRET_NAME = re.compile(
    r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|COOKIE|AUTHORIZATION|(?:^|[_-])AUTH(?:[_-]|$)",
    re.IGNORECASE,
)
"""Variable and header names whose values are credentials."""
JSON_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class Redactor:
    """Replaces every occurrence of the secrets inside JSON strings, counting hits."""

    def __init__(self, secrets: set[str]) -> None:
        # Inside a JSON string a secret is escaped; inside a JSON document quoted within
        # a string (a tool result) it is escaped twice. Match every spelling.
        forms = set(secrets)
        for _ in range(2):
            forms |= {
                json.dumps(form, ensure_ascii=escape)[1:-1]
                for form in list(forms)
                for escape in (True, False)
            }
        alternatives = "|".join(
            re.escape(form) for form in sorted(forms, key=len, reverse=True)
        )
        self.pattern = re.compile(alternatives) if forms else None
        self.count = 0

    def json(self, text: str) -> str:
        """Redact one JSON document given as text; structure and non-string values stay."""
        pattern = self.pattern
        if pattern is None:
            return text

        def string(match: re.Match[str]) -> str:
            inner, hits = pattern.subn(REDACTED, match.group(1))
            self.count += hits
            return f'"{inner}"'

        return JSON_STRING.sub(string, text)


def known_secrets(
    episodes: list[Episode], config: EvalConfig, *values: str
) -> set[str]:
    """Every credential this run could have echoed into a trace: the clients' API keys,
    credential-named client header / harness / host environment values, the rollouts'
    interception tokens, and `values`. Values shorter than `MIN_SECRET_LENGTH` are
    dropped — redacting them would rewrite ordinary text."""
    traces = [trace for episode in episodes for trace in episode.traces]
    agents = [trace.agent.config for trace in traces]
    clients = [config.client, *(a.client for a in agents if a.client is not None)]
    named = [
        os.environ,
        *(client.headers for client in clients),
        *(a.harness.resolved_env for a in agents if a.harness is not None),
    ]
    secrets = {
        *values,
        *(resolve_api_key(client) for client in clients),
        *(secret for trace in traces for secret in trace.upload_secrets),
        *(
            value
            for mapping in named
            for name, value in mapping.items()
            if SECRET_NAME.search(name)
        ),
    }
    return {secret for secret in secrets if len(secret) >= MIN_SECRET_LENGTH}


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
                resp = client.post(f"{api}{path}", json=body)
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
