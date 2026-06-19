# mini-browse-apps-platform-v1

Sandboxed local-app **browser-agent** environment. Each task boots a local single-page web app
(SPA server + headless-Chromium CDP service) inside a per-task Docker image; a browser agent drives
it by **screenshots → vision model → click/type actions**, then submits a structured JSON result. An
LLM judge scores the submission against a deterministic answer key.

The model must be **multimodal** (the agent's only input is screenshots).

## Proprietary agent (fetched at run time)

The browser agent is **proprietary and not vendored in this repo**. The harness fetches it at run
time from a **private GitHub repo** (pinned to a commit), caches it under
`~/.cache/verifiers/browse-agent/<sha>/`, then stages it into the sandbox. Configure via
`--harness.*`:

| Field | Default | Meaning |
| --- | --- | --- |
| `agent.repo` | `PrimeIntellect-ai/plex-mini-browse` | Private `owner/name` to fetch the agent from. |
| `agent.ref` | _(unset)_ | **Pinned commit sha to fetch (required unless `agent.path` is set).** |
| `agent.token_env` | `MINI_BROWSE_GITHUB_TOKEN` | Env var holding a GitHub token with read access to `agent.repo`. |
| `agent.path` | _(unset)_ | Local dir containing the agent package — skips the fetch (development). |

## Tasks (pulled dynamically)

Tasks are **pulled from the Prime hub and cached locally** — nothing is bundled in this package.
`load_tasks` pulls the dataset from `prime/mini-browse-apps-platform-v1` (private; via `prime env
pull`) into `~/.cache/verifiers/mini-browse-apps/<version>/`. Override with `--taskset.dataset_path
<file>`, or repoint `--taskset.hub_env_id` / `--taskset.hub_version`.

## Run

The taskset and harness are co-packaged (resolved via `__all__`), so `--harness.id` matches the
taskset id. The task image is a Prime-registry image, so use the `prime` runtime:

```bash
export MINI_BROWSE_GITHUB_TOKEN=<token>
uv run eval mini-browse-apps-platform-v1 \
  --harness.id mini-browse-apps-platform-v1 \
  --harness.runtime.type prime \
  --harness.agent.ref <agent-commit-sha> \
  -m <multimodal-model> \
  -n 1 -r 1 -c 1
```

## Reward & metrics

`answer_key` (weight 1.0) judges the submitted result against the gold answer key. The judge uses a
structured-output (`json_schema`) model — default `openai/gpt-4.1-mini` on Prime inference
(auto-resolved); override with `--taskset.judge.model` / `--taskset.judge.client.*`. Reward 1.0 ==
all expected fields correct (`verdict: "yes"`); partial credit is `correct_fields / total_fields`.
Metrics: `result_present`, `submitted_result_present`, `agent_error`, `transcript_image_count`,
`message_count`.

## Config (`--taskset.*`)

| Field | Default | Meaning |
| --- | --- | --- |
| `dataset_path` | `null` | Local dataset override (skips the hub pull). |
| `hub_env_id` | `prime/mini-browse-apps-platform-v1` | Hub env the dataset is pulled from. |
| `hub_version` | `latest` | Hub env version to pull. |
| `judge.model` | `openai/gpt-4.1-mini` | Structured-output judge model. |
| `judge.client` | Prime inference | OpenAI-compatible endpoint for the judge (auto-resolved). |
