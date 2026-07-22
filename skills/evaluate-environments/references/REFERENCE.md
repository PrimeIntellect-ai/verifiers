# REFERENCE.md

A complete reference of every settable config field for **evaluating tasksets in `verifiers.v1`**. The config tree is parsed from CLI flags (dotted, e.g. `--env.agent.runtime.type docker`) and/or `@ file.toml` by `prime-pydantic-config`; every field below is settable either way unless noted.

The root config the eval CLI parses is [`EvalConfig`](#evalconfig--the-run). It composes the environment (`env` — the whole `[env]` block: taskset, agents, limits) with the run knobs (model, sampling, counts) and the worker pool. The tree:

```text
EvalConfig                          (the run)
├─ model, sampling, client, num_tasks, num_rollouts, shuffle, max_concurrent, …
├─ env: EnvConfig                   (subclass resolved by --env.id, else the taskset's env)
│  ├─ id                            (which Environment — empty = the taskset's story)
│  ├─ taskset: TasksetConfig | None (subclass resolved by --env.taskset.id / positional)
│  │  └─ task: TaskConfig           (judges, scoring knobs, task-scoped server config)
│  ├─ <agent>: AgentConfig          (per-agent harness/model/client/sampling +
│  │  │                              per-run caps; `agent` on the single-agent env;
│  │  │                              subclass resolved by --env.<agent>.harness —
│  │  │                              harness knobs sit flat on the seat)
│  │  ├─ harness                    (the harness id; None = the taskset's default)
│  │  ├─ runtime: RuntimeConfig     (subprocess | docker | prime | modal)
│  │  ├─ env / forward_env / disabled_tools
│  │  ├─ timeout: TimeoutConfig     (per-stage: setup/rollout/finalize/scoring)
│  │  ├─ retries: RolloutRetryConfig (per-agent whole-run retries)
│  │  └─ max_turns / max_input_tokens / max_output_tokens / max_total_tokens
│  ├─ timeout: EnvTimeoutConfig     (episode / finalize — the env's own hooks)
│  ├─ retries: RolloutRetryConfig   (whole-episode fallback for faults no agent owns)
│  └─ interception
└─ pool: PoolConfig                 (static | elastic) — env-server only
```

There is no run-level harness: each agent pins its own (`--env.agent.harness <id>` on the single-agent env), an unpinned agent runs the taskset's default harness (its bundled one, else `bash`), and a declared pin is the env author's default. The harness's config is flat on the agent: its knobs address as `--env.<agent>.<knob>`. The retired flat axes error with a pointer: `--taskset.*` → `--env.taskset.*`, `--harness.*` → `--env.<agent>.harness`.

Sibling entrypoints reuse the same tree: [`ServeConfig`](#serveconfig--the-env-server-cli) (env server) and [`ValidateConfig`](#validateconfig--the-validate-cli) (per-task validation). All three live in `verifiers/v1/configs/`.

---

## EvalConfig — the run

`verifiers/v1/configs/eval.py` — `EvalConfig(EnvServerConfig)`. The single config object the eval CLI parses. Inherits [`EnvServerConfig`](#envserverconfig--the-pool) (the `env` block + `--pool.*` + the legacy v0 fields) and adds the run knobs. Everything environment-shaped lives under `--env.*` / `[env]`.

| Field | Type | Default | Aliases | Notes |
| --- | --- | --- | --- | --- |
| `uuid` | `str` | `uuid4()` | — | Auto-generated run id; the leaf of the output dir. Excluded from the saved config. |
| `model` | `str` | `"deepseek/deepseek-v4-flash"` | `model`, `m` | Model id. |
| `client` | `ClientConfig` | `EvalClientConfig()` | — | The model client (discriminated union — see [Client config](#client-config)). |
| `sampling` | `SamplingConfig` | `SamplingConfig()` | — | Per-request sampling knobs (see [Sampling config](#sampling-config)). |
| `num_tasks` | `int \| None` | `None` | `batch_size`, `num_examples`, `num_tasks`, `n` | How many tasks to evaluate (None = all). |
| `num_rollouts` | `int` | `1` | `group_size`, `rollouts_per_example`, `num_rollouts`, `r` | Independent episodes per task — the trainer's group size. Env-internal fan-out (e.g. best-of-n attempts) is the env's own knob, not `-r`. |
| `shuffle` | `bool` | `False` | `shuffle`, `s` | Shuffle tasks before taking the first `num_tasks`. |
| `max_concurrent` | `int \| None` | `128` | `max_concurrent`, `c` | Max rollouts in flight at once. |
| `verbose` | `bool` | `False` | `verbose`, `v` | Log at debug level instead of info. |
| `dry_run` | `bool` | `False` | — | Resolve + validate the config and dump it, then exit. |
| `rich` | `bool` | `True` | — | Live dashboard instead of per-rollout logs (in-process only). |
| `server` | `bool` | `False` | — | Drive rollouts through the env-server worker pool (sized by `pool`) instead of in-process — the path prime-rl trains through. Incompatible with `--rich`. |
| `push` | `bool` | `True` | — | Upload the finished run to the private Evaluations tab. Disable with `--no-push`. |
| `output_dir` | `Path \| None` | `None` | `output_dir`, `o` | Where to write the run (`config.toml` + `traces.jsonl`). None = a fresh per-run dir under `outputs/<env>--<model>--<harness>/<uuid>`. |
| `resume` | `Path \| None` | `None` | — | Set by `--resume <dir>`: re-run missing/errored rollouts, episode-atomically (a multi-trace rollout is kept or redone as a unit; `Environment.complete` is the keep verdict). Excluded from the saved config; takes no other args. |

Validator: `--rich` + `--server` together is rejected (the dashboard is in-process only).

Inherited from `EnvServerConfig`: [`env`](#envconfig--the-environment), [`pool`](#pool-config), and the legacy `id` / `args` / `extra_env_kwargs`.

---

## Sampling config

`verifiers/v1/types.py` — `SamplingConfig(BaseModel)` (alias `Sampling`). Used as `EvalConfig.sampling` and embedded in [`JudgeSamplingConfig`](#judge-config). `extra='allow'`, so provider-specific keys pass through.

| Field | Type | Default | Aliases | Notes |
| --- | --- | --- | --- | --- |
| `temperature` | `float \| None` | `None` | — | Sampling temperature. |
| `top_p` | `float \| None` | `None` | — | Nucleus sampling. |
| `reasoning_effort` | `str \| None` | `None` | — | Provider reasoning-effort level. |
| `max_tokens` | `int \| None` | `None` | `max_tokens`, `max_completion_tokens` | Max completion tokens per request. |
| *(extra)* | any | — | — | Any provider-specific key passes through (`extra='allow'`). |

---

## Client config

`verifiers/v1/clients/config.py`. Discriminated on `type` (`eval` | `train`); selected with `--client.type`.

### `BaseClientConfig` (common)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `base_url` | `str` | `https://api.pinference.ai/api/v1` | OpenAI-compatible endpoint. Falls back to the active Prime CLI config (or `PRIME_INFERENCE_URL`). |
| `api_key_var` | `str` | `"PRIME_API_KEY"` | Env var the key is read from. |
| `headers` | `dict[str, str]` | `{}` | Extra HTTP headers on every request. `X-Prime-Team-ID` auto-set for pinference hosts. |

### `EvalClientConfig(BaseClientConfig)` — `type: "eval"` (default)

The default: forward each request to a matching endpoint. No extra fields.

### `TrainClientConfig(BaseClientConfig)` — `type: "train"`

A vLLM `/inference/v1/generate` endpoint with client-side tokenization (responses carry token ids + logprobs). Needs a running vLLM engine.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `renderer` | `RendererConfig \| None` | `None` | The `renderers.RendererConfig`. `None` auto-resolves from the model (falls back to the default renderer — no tool support — for unknown models). |
| `pool_size` | `int` | `1` | Renderer slots shared across concurrent rollouts (client-side tokenization). |
| `renderer_model_name` | `str \| None` | `None` | Model the tokenizer/renderer pool is built for. Pin to the base model so a LoRA adapter name never drives tokenizer loading. Falls back to the per-request model. |

---

## EnvConfig — the environment

`verifiers/v1/env.py` — `EnvConfig(BaseConfig)`, the run's whole `[env]` block. One subclass per `Environment` class (bound via `Environment[YourConfig]`): the base carries which env (`id`), the seed taskset, and the env-agnostic run limits; the subclass declares each agent as an `AgentConfig` field plus the env's own knobs (`--env.n`, `--env.weight`, ...). The run's `env` field is narrowed to the selected env's config class by `--env.id` (else the taskset id, else `SingleAgentEnvConfig`), so everything renders typed in `-h`. Each loaded `Task` supplies the row's behavior, tools, user simulator, and scoring; only its `TaskData` is stored on the trace.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `id` | `ID` | `""` | Which `Environment` (control flow between agents) runs: a bundled env (`best-of-n`, `agentic-judge`), a local package, or a Hub `org/name[@version]`. Empty = the taskset's own story (its exported `Environment` subclass, else `SingleAgentEnv`). |
| `taskset` | `TasksetConfig \| None` | `None` | The seed taskset every rollout starts from; resolved to its concrete subclass by `--env.taskset.id` (positional shorthand: `eval <taskset-id>`). See [Taskset config](#taskset-config). `None` only for a taskset-less env; every bundled env requires one. |
| *(agents)* | `AgentConfig` | env-declared | Each declared agent: `agent` on `SingleAgentEnvConfig`, `solver`/`judge` on the agentic-judge env, the env's own names elsewhere. See [Agent config](#agent-config). |
| `timeout` | `EnvTimeoutConfig` | `EnvTimeoutConfig()` | The env's own hook bounds: `--env.timeout.episode` (the whole `run()` hook) and `--env.timeout.finalize` (the `finalize()` hook). Per-run stage timeouts are each agent's (`--env.<agent>.timeout.*`). |
| `retries` | `RolloutRetryConfig` | `RolloutRetryConfig()` | Whole-EPISODE retries, the coarse fallback for faults no agent owns. See [Retry config](#retry-config). Set via `--env.retries.*`. |
| `interception` | `InterceptionConfig` | `ElasticInterceptionPoolConfig()` | The interception shape: `elastic` (default — servers grown on demand, `multiplex` rollouts each), `server` (one server, tunnel choice incl. bring-your-own endpoint), or `static` (a fixed list). |

Per-run caps (turns, tokens, stage timeouts, retries) are agent fields, not env fields — see [Agent config](#agent-config).

### Agent config

`verifiers/v1/harness.py` — `AgentConfig(BaseConfig)`. One per declared agent, addressed `--env.<agent>.*`. The seat's config **is** the harness's config: a harness subclasses `AgentConfig` to add its run knobs, the seat narrows to that subclass by `--env.<agent>.harness`, and the knobs sit flat on the seat (`--env.<agent>.<knob>`). The **model context** defaults to the run's own (the serve protocol carries model/client/sampling per rollout request — what makes self-play trainable). The **harness** does not: an unpinned agent runs the taskset's default harness; a declared pin is the env author's per-agent default; partial overrides deep-merge onto it.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `harness` | `ID \| None` | `None` | The harness id — the agent's program, selecting the seat's concrete config subclass. `None` = the taskset's default harness (its bundled one, else `bash`). See [Harness config](#harness-config). |
| `runtime` | `RuntimeConfig` | `SubprocessConfig()` | Where the harness runs. Discriminated union — see [Runtime configs](#runtime-configs). Set with `--env.<agent>.runtime.type docker\|prime\|modal`. |
| `env` | `dict[str, str]` | `{}` | Additional env vars for the harness program. Harness-owned endpoint/auth/model vars take precedence. |
| `forward_env` | `list[str]` | `[]` | Names of env vars to forward from `os.environ` into the harness program's runtime (for secrets not in checked-in config). Absent names are skipped; explicit `env` wins. |
| `disabled_tools` | `list[str] \| None` | `None` | Harness-specific tool names to disable. |
| `model` | `str \| None` | `None` | Pin a model for this agent (None = the run's `--model`). |
| `client` | `ClientConfig \| None` | `None` | Pin an endpoint (None = the run's `--client.*`) — route a frozen judge or user sim off the training endpoint. |
| `sampling` | `SamplingConfig \| None` | `None` | Pin sampling (None = the run's). |
| `timeout` | `TimeoutConfig` | `TimeoutConfig()` | Per-stage wall-clock timeouts for this agent's runs (`setup`/`rollout`/`finalize`/`scoring`; each stage falls back to the task's own). |
| `max_turns` / `max_input_tokens` / `max_output_tokens` / `max_total_tokens` | `int \| None` | `None` | Per-agent caps (None = no limit); map onto [`RolloutLimits`](#rollout-limits), each checked between turns (soft by one turn). |

Trainability is not a config field: it is env truth, set in place by the env's `setup(agents)` hook (default: every agent trains) and stamped on each trace. An env that wants the flip run-configurable exposes its own switch (e.g. proposer-solver's `--env.train_solver false`).

### Legacy (v0) backwards-compat fields

On `EnvServerConfig` (below): set `id` (leave `env.taskset` unset) to run a classic `verifiers.load_environment` env through the legacy bridge.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `id` | `ID \| None` | `None` | Classic v0 env id (`name`, `org/name`, or `org/name@version`). |
| `args` | `dict` | `{}` | Construction kwargs forwarded to `load_environment(id, **args)`. |
| `extra_env_kwargs` | `dict` | `{}` | Post-load kwargs applied via `env.set_kwargs(**...)` (e.g. `max_total_completion_tokens`, `max_seq_len`, `timeout_seconds`). |

`EnvServerConfig.is_legacy` → `id is not None and no v1 taskset`.

### EnvServerConfig — the pool

`EnvServerConfig(BaseConfig)`. The `env` block plus the worker pool sizing and the legacy v0 fields. Shared by the `serve` CLI, server-backed eval, and prime-rl's orchestrator.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `env` | `EnvConfig` | `SingleAgentEnvConfig()` | The environment (above). `SerializeAsAny`, so the resolved subclass's agents and knobs survive `model_dump` onto the wire. |
| `pool` | `PoolConfig` | `ElasticPoolConfig()` | See [Pool config](#pool-config). |

---

## Timeout config

`verifiers/v1/env.py` — `TimeoutConfig(BaseConfig)`. Framework-enforced wall-clock timeouts per rollout stage, in seconds (None = no limit). An **agent** field (`--env.<agent>.timeout.*`); precedence: cli/toml > per-task [`TaskTimeout`](#task-resources--timeouts) > default.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `setup` | `float \| None` | `None` | Shared wall-clock budget for `Task.setup` and harness provisioning. |
| `rollout` | `float \| None` | `None` | Max wall-clock for the rollout (the harness run). |
| `finalize` | `float \| None` | `None` | Max wall-clock for the task's `finalize` hook. |
| `scoring` | `float \| None` | `None` | Max wall-clock for task rewards/metrics/judges and harness metrics. |

`EnvTimeoutConfig` (the env's `--env.timeout.*`) keeps only `episode` — the bound on the whole `run()` interaction — and `finalize` — the bound on the env's `finalize()` hook.

> Remote sandboxes cap any harness timeout at 24 hours (provider max lifetime).

---

## Retry config

`verifiers/v1/retries.py`. Per-call model/runtime retries are owned by the harness/runtime SDKs; the framework keeps only **whole-run** retries, in two atoms sharing one shape: each agent's own (`--env.<agent>.retries.*` — rerun that agent's rollout; never into a borrowed box) and the env's whole-episode fallback (`--env.retries.*` — for faults no agent owns; a retried episode reruns whole).

### `RolloutRetryConfig`

Rerun when the run ends with a captured error. Matching is by the error's **exception type name**.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `max_retries` | `int` | `0` (≥0) | Retries beyond the first attempt (0 = no retry). |
| `include` | `list[str]` | `[]` | Only retry errors whose type is listed. Empty = retry anything not excluded. |
| `exclude` | `list[str]` | `[]` | Never retry these types (wins over `include`). |

---

## Pool config

`verifiers/v1/env.py`. Discriminated on `type`; selected with `--pool.type static|elastic`. Drives the env-server worker pool (the `--server` path).

### `StaticPoolConfig` — `type: "static"`

Fixed pool: pre-spawn `num_workers` up front.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `num_workers` | `int` | `4` (≥1) | Worker processes to pre-spawn (1 = a single in-process server, no pool). |

### `ElasticPoolConfig` — `type: "elastic"` (default)

Elastic pool: start at one worker and scale up on demand.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `max_workers` | `int \| None` | `None` | Upper bound on workers (None = unbounded). |
| `multiplex` | `int` | `128` (≥1) | Rollouts per worker for the scale-up trigger: add a worker once in-flight rollouts reach 90% of `workers * multiplex`. |

---

## Taskset config

`verifiers/v1/taskset.py` — `TasksetConfig(BaseConfig)`. Subclass it for values used while `Taskset.load()` builds the task list: dataset id, split, seed, sample count, difficulty filters, and similar load-time choices. The concrete subclass is selected through `env.taskset.id`, so its fields become typed dotted flags such as `--env.taskset.split test`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `id` | `ID` | `""` | Local package or Hub `org/name[@version]`; selects the taskset and its config type. Set via `--env.taskset.id` or the positional `eval <taskset-id>`. |
| `task` | `TaskConfig` | `TaskConfig()` | Task-facing config passed to every constructed task. `SerializeAsAny` preserves a narrowed subclass. Set through `--env.taskset.task.*`. |

`.name` → the package name (id with org / version stripped).

A taskset implements `load()` and declares exactly one task type through its generic base. It may also declare task-agnostic tool classes on `Taskset.tools`; those servers are shared by the rollouts handled by one environment worker.

### Task config

`TaskConfig` contains knobs read by task behavior. Subclass it for scoring parameters and task-scoped `ToolsetConfig` or `UserConfig` fields, then narrow the taskset config's `task` field to that subclass. These are run-wide knobs, not per-row data; the row itself belongs on `TaskData`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `judges` | `Judges` | `[]` | Judge plugins run by `Task.score`; set through `--env.taskset.task.judges`. |

---

## Harness config

A harness's config is **flat on the agent**: each harness subclasses [`AgentConfig`](#agent-config) to add its run knobs, the seat narrows to that subclass by `--env.<agent>.harness` (`--env.agent.harness` on the single-agent env), and the knobs address as `--env.<agent>.<knob>`. An unpinned agent runs the taskset's bundled harness, else `bash`.

`.name` → the package name; `.resolved_env` → `env` merged with forwarded `forward_env` vars.

A harness class also declares capability flags (ClassVars, not user-settable): `APPENDS_SYSTEM_PROMPT`, `SUPPORTS_MCP`, `SUPPORTS_USER_SIM`, `SUPPORTS_MESSAGE_PROMPT`.

### Built-in harness configs

All inherit the base `AgentConfig` fields (`harness`, `runtime`, `env`, `forward_env`, `disabled_tools`, the model-context pins, and the per-run caps).

#### `BashHarnessConfig` — `harness = "bash"` (the fallback)

A growing-message-list chat loop with a local `bash` tool, plus optional `edit`/`search`. A uv script (deps: `openai`, `mcp`).

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `edit` | `bool` | `True` | Offer the local `edit` tool (single-occurrence string replacement) alongside `bash`. |
| `search` | `bool` | `False` | Offer a `search` tool (Google web results via serper.dev). Requires `SERPER_API_KEY` in the eval environment. |

#### `NullHarnessConfig` — `harness = "null"`

A growing-message-list chat loop with the task- and taskset-scoped MCP tools, and **no built-in tools of its own**. It runs as a uv script whose dependencies are `openai` and `mcp`, so setup bootstraps everything it needs in the selected runtime. `NullHarnessConfig` adds no fields beyond the base `AgentConfig` fields. Use it for pure chat or tasksets whose entire tool surface is provided through MCP; use an agentic harness for shell/edit capabilities.

#### `CodexHarnessConfig` — `harness = "codex"`

Installs the Codex CLI into the runtime and runs `codex exec`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `version` | `str` | `"0.144.5"` | Codex release to install (the `rust-v<version>` GitHub release); pinned. |

#### `RLMHarnessConfig` — `harness = "rlm"`

Installs the rlm CLI and runs it. Knobs map onto `RLM_*` env vars; the base `env` field passes any other `RLM_*` var through verbatim.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `version` | `str` | `"main"` | Git ref (branch/tag/commit) of rlm to install. |
| `max_depth` | `int` | `0` | Recursion depth rlm may spawn sub-harnesses to (`RLM_MAX_DEPTH`). |
| `skills` | `list["edit" \| "search"]` | `[]` | Built-in rlm skills to enable (`RLM_SKILLS`). Empty enables none. |
| `summarize_at_tokens` | `int \| (int, int) \| None` | `None` | Auto-compaction threshold (`RLM_SUMMARIZE_AT_TOKENS`): compact once context grows past this many tokens. An int is fixed; a `(lo, hi)` pair draws a per-group threshold (seeded by task index). `None` disables. Ints must be positive. |

#### `MiniSWEAgentHarnessConfig` — `harness = "mini-swe-agent"`

Runs the native bash-tool agent through LiteLLM.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `version` | `str` | `"2.4.5"` | mini-swe-agent release to install, pinned. |

#### `Terminus2HarnessConfig` — `harness = "terminus-2"`

Runs Harbor's tmux agent through LiteLLM.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `version` | `str` | `"0.14.0"` | Harbor release to install, pinned. |

#### `KimiCodeHarnessConfig` — `harness = "kimi-code"`

Installs the Kimi Code CLI and runs it headlessly.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `version` | `str` | `"0.27.0"` | Kimi Code release to install, pinned. |

---

## Runtime configs

`verifiers/v1/runtimes/`. Discriminated on `type`; selected with the agent's `--env.<agent>.runtime.type` (or `--runtime.type` for the validate CLI). The same union is reused as `ToolsetConfig.runtime` and `UserConfig.runtime`.

### `SubprocessConfig` — `type: "subprocess"` (default)

Run on the host in a fresh `/tmp/<name>` workspace per rollout. **No extra fields.** Implicit host inheritance removes names containing `API_KEY`; explicit values passed in the runtime `env` mapping are merged afterward and inherited by child processes.

### `DockerConfig` — `type: "docker"`

Local Docker container. A bare wildcard with no block entries shares the host network;
every narrower policy uses an isolated bridge during agent execution.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `image` | `str` | `"python:3.11-slim"` | Container image. |
| `workdir` | `str` | `"/app"` | Working directory. |
| `cpu` | `float \| None` | `None` | Pin to this many CPU cores (`docker --cpus`). None = unlimited. |
| `memory` | `float \| None` | `None` | Hard memory limit in GB (`docker --memory`). None = unlimited. |
| `gpu` | `str \| None` | `None` | GPU spec, e.g. `"A100"` or `"2"` (`docker --gpus` uses the count; needs the nvidia toolkit). |
| `disk` | `float \| None` | `None` | Advisory disk request in GB. Docker has no portable per-container size limit, so accepted but **not enforced**. |
| `allow` | `list[str]` | `[]` | URL origins or host patterns allowed during execution, e.g. `"https://*.wikipedia.org"`. Empty permits only automatically added interception/MCP routes; bare `"*"` with no block entries opts out of filtering. Wildcards are supported; `*.example.com` also matches the apex. URL paths are ignored. An explicit HTTPS origin authorizes a nonstandard CONNECT port; CONNECT authority and TLS SNI must both match policy. Under filtered policies, non-global addresses (including host loopback, private, and link-local) are reserved for framework routes. |
| `block` | `list[str]` | `[]` | URL origins or host patterns denied during execution. Block wins over user `allow`; interception and MCP routes always remain reachable. |

### `PrimeConfig` — `type: "prime"`

Remote Prime sandbox; reached via native port exposure.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `image` | `str` | `"python:3.11-slim"` | Container image. |
| `workdir` | `str` | `"/app"` | Working directory. |
| `network_access` | `bool` | `True` | Allow outbound network from the sandbox. |
| `vm` | `bool` | `False` | Run as a micro-VM (kernel features / stronger isolation). |
| `guaranteed` | `bool` | `False` | Request guaranteed (vs best-effort) capacity. |
| `region` | `str \| None` | `None` | Region to provision in (None = provider-chosen). Note: port exposure is region-gated; `us` supports it. |
| `labels` | `list[str]` | `[]` | Labels attached to the sandbox. |
| `cpu` | `float` | `1.0` | CPU cores. |
| `memory` | `float` | `2.0` | Memory in GB. |
| `gpu` | `str \| None` | `None` | GPU spec, e.g. `"A100"` or `"A100:2"` (bare count = provider-chosen type). |
| `disk` | `float` | `5.0` | Disk in GB. |
| `idle_timeout` | `float \| None` | `3600` | Seconds of inactivity before the sandbox is deleted; `None` disables it. |
| `creates_per_min` | `int \| None` | `None` | Pace sandbox creation to this many per minute, host-wide across every env-server worker (None/≤0 disables). Tunnel creation is limited separately and globally. |

### `ModalConfig` — `type: "modal"`

Remote Modal sandbox; reached via Modal's own port forwarding (`encrypted_ports`).

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `image` | `str` | `"python:3.11-slim"` | Container image (registry). |
| `workdir` | `str` | `"/app"` | Working directory. |
| `network_access` | `bool` | `True` | Allow outbound network (`block_network` is the negation). |
| `region` | `str \| None` | `None` | Region to provision in (None = provider-chosen). |
| `cpu` | `float` | `1.0` | CPU cores. |
| `memory` | `float` | `2.0` | Memory in GB (sent to Modal as MB). |
| `gpu` | `str \| None` | `None` | GPU spec, e.g. `"A100"` or `"A100:2"`. |
| `disk` | `float` | `5.0` | Disk in GB. Modal sandboxes have no disk knob, so **accepted but not enforced**. |
| `creates_per_sec` | `float \| None` | `40.0` | Pace sandbox creation to this many per second, host-wide across every env-server worker (None/≤0 disables). |

Before each rollout or validation check, `resolve_runtime_config` combines the selected runtime config with the row's `TaskData`:

- `TaskData.image` is the row's required execution image and replaces the runtime's base image. A
  row with an image cannot use the subprocess runtime.
- `TaskData.workdir` fills the runtime workdir only while that field is still at the runtime
  class's default. Any non-default runtime-config workdir wins.
- Non-`None` `TaskData.resources` values similarly fill supported runtime fields only while those
  fields remain at their defaults. Any non-default runtime-config resource value wins.
- Non-wildcard task URL policy fields require framework-aware Docker policy support.
  `TaskData.network_allow=["*"]` is neutral; a concrete task list replaces an evaluator
  wildcard, otherwise concrete task/runtime lists combine. `TaskData.network_block`
  combines with runtime `block`, and every block rule wins over allow rules.
- A resource field unsupported by the chosen runtime is ignored; evaluation warns once per
  runtime/field combination. Docker and Modal accept `disk` so portable task data validates, but neither enforces a disk limit.

This resolution produces a copied runtime config; it does not mutate the frozen row or the shared base config.

---

## Task resources & timeouts

`verifiers/v1/task.py`. `TaskData` is the frozen, serializable row stored on `trace.task` and sent across worker/runtime boundaries. Taskset packages subclass it for typed row-specific fields such as reference answers, repository metadata, or judge inputs. Those fields are not CLI flags; they are produced by `Taskset.load()`.

`Task` is the behavior wrapper around that row. It owns hooks and scoring and reads uniform, run-configurable knobs from `TaskConfig`, but the `Task` object itself is not serialized onto the trace. Replay validates each recorded row as the taskset's declared `TaskData`, reattaches the declared task class, and propagates validation failures instead of changing task types.

### `TaskResources` (frozen)

Portable runtime resources requested by one row. CPU is measured in cores; memory and disk are in gigabytes. `None` means “do not override the chosen runtime/provider default.” Values are applied only to runtime fields that remain at their defaults; any non-default runtime-config value wins. Unsupported fields are ignored; evaluation warns once per runtime/field combination.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `cpu` | `float \| None` | `None` | CPU cores. Docker treats this as a hard `--cpus` limit; sandbox providers receive the same core count. |
| `memory` | `float \| None` | `None` | Memory in GB. Docker enforces it as a hard limit; Modal receives the value converted to MB. |
| `gpu` | `str \| None` | `None` | GPU spec such as `"A100"` or `"A100:2"` (`type[:count]`; a bare count lets supported providers choose the type). |
| `disk` | `float \| None` | `None` | Disk in GB. Enforced by Prime; accepted but advisory/not enforced by Docker and Modal. |

### `TaskTimeout` (frozen)

Per-row wall-clock timeout requests, in seconds, one for each rollout stage. For eval, a non-`None` value in the agent's `TimeoutConfig` (`--env.<agent>.timeout.*`) wins; otherwise the corresponding row value is used. If both are `None`, that stage has no framework timeout. Remote harness execution is capped at the provider's 24-hour sandbox lifetime.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `setup` | `float \| None` | `None` | Task and harness setup stage. Overridden by the agent's `timeout.setup`; validate uses it for `Task.setup` when `CheckTimeoutConfig.setup` is unset. |
| `harness` | `float \| None` | `None` | Harness execution. Overridden by the agent's `timeout.rollout`. |
| `finalize` | `float \| None` | `None` | Task `finalize` hook. Overridden by the agent's `timeout.finalize`. |
| `scoring` | `float \| None` | `None` | Task rewards/metrics/judges and harness metrics. Overridden by the agent's `timeout.scoring`. |

### `TaskData` (frozen)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `idx` | `int` | — | Stable integer index within the taskset. Used for selection, grouping, display, and reproducibility. |
| `name` | `str \| None` | `None` | Optional human-readable label used in logs and dashboards. |
| `description` | `str \| None` | `None` | Optional human-readable description. |
| `prompt` | `str \| Messages \| None` | — | Initial user input. A string is one user prompt; `Messages` seeds a full initial conversation and requires a harness with `SUPPORTS_MESSAGE_PROMPT`; `None` lets the user simulator open via `respond("")`. |
| `system_prompt` | `str \| None` | `None` | Optional system prompt. Harnesses with `APPENDS_SYSTEM_PROMPT` emit a real system message; otherwise a string prompt is prefixed with a warning. A separate system prompt cannot be folded into `Messages` or `None`. |
| `image` | `str \| None` | `None` | Required container/sandbox image for this row. It replaces the base runtime image; subprocess is refused when set. |
| `workdir` | `str \| None` | `None` | Working directory for harness execution and task hooks. Applied when the runtime supports it and its config remains at the default. |
| `network_allow` | `list[str]` | `["*"]` | Docker destinations needed by the task. The wildcard is neutral and leaves evaluator policy intact; empty requests framework-only access. |
| `network_block` | `list[str]` | `[]` | Destinations merged into Docker's `block` list. A non-empty list requires Docker filtering. |
| `timeout` | `TaskTimeout` | `TaskTimeout()` | Per-stage timeout requests described above. |
| `resources` | `TaskResources` | `TaskResources()` | Portable runtime resource requests described above. |

`TaskData.prompt_text` renders a string prompt directly or joins the textual content of a `Messages` prompt. Judges use it as the default question text when no dedicated question field is configured.

---

## Toolset config

`verifiers/v1/mcp/toolset.py`. Tool scope is structural: a class declared on `Task.tools` is task-scoped, while a class declared on `Taskset.tools` is shared by one environment worker. The framework finds the matching config field by the toolset's generic config type. Subclass either config to add knobs consumed by the tool's `@vf.tool` methods.

### `ToolsetConfig` — `Task.tools`

A task-scoped server is launched per rollout. Its matching config field normally lives on `TaskConfig`, under `--env.taskset.task.*`.

The default placement is the toolset's own subprocess runtime on the host, where verifiers and the taskset package are already installed. The harness reaches that server over the host network when local or through a tunnel when remote. `colocated` instead runs the server inside the harness runtime; this is useful when both must see the same filesystem or processes, but a remote sandbox must then upload and install verifiers plus the taskset package for every rollout.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `colocated` | `bool` | `False` | Run inside the harness's own runtime and reach the tool on an in-runtime local port. The separate `runtime` field is ignored. |
| `runtime` | `RuntimeConfig` | `SubprocessConfig()` | The server's own runtime when not colocated. Select Docker/Prime/Modal to isolate it from the host. See [Runtime configs](#runtime-configs). |
| `url` | `str \| None` | `None` | Existing streamable-HTTP MCP endpoint. When set, verifiers connects to it instead of launching the class, so placement fields do not take effect. |

### `SharedToolsetConfig` — `Taskset.tools`

A taskset-scoped tool uses one framework-launched server per environment worker, or reuses a configured external `url` without launching a server. Its matching config field lives directly on the taskset config, not under `task`.

The framework-launched form is intended for expensive task-agnostic setup such as loading a corpus, index, or graph: `setup()` runs once per worker and `setup_task()` is not called because no single row owns the server. A vf-native shared tool may still use mutable `self.state`; the framework attaches each calling rollout's state channel to the shared URL so those values remain per rollout. There is no `colocated` option because a shared server has no single harness runtime.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `runtime` | `RuntimeConfig` | `SubprocessConfig()` | The framework-launched server's own runtime. Host subprocess is cheapest; a remote runtime pays setup once per worker. See [Runtime configs](#runtime-configs). |
| `url` | `str \| None` | `None` | Existing streamable-HTTP MCP endpoint reused across workers and rollouts instead of launching a server. |

There is no `shared` boolean on `ToolsetConfig`: declare the class on `Task.tools` or `Taskset.tools` and use the matching config type to choose its scope.

---

## User config

`UserConfig` controls a simulator declared on `Task.user`; its matching config field belongs on `TaskConfig` under `--env.taskset.task.*`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `colocated` | `bool` | `False` | Run the user simulator inside the harness's runtime (its port is published back to the host so the framework can still drive it). |
| `runtime` | `RuntimeConfig` | `SubprocessConfig()` | The user simulator's own runtime, used unless `colocated`. See [Runtime configs](#runtime-configs). |

---

## Judge config

`verifiers/v1/judge.py`. `TaskConfig.judges` holds plugged judge configs that `Task.score` constructs and runs after the task's own rewards. Each list entry needs an `id`; the loader resolves that plugin's concrete `JudgeConfig` subclass before validation, just like tasksets and harnesses. Duplicate reward keys are rejected unless entries receive distinct `name` values.

A task may instead declare a custom `JudgeConfig` field on its own `TaskConfig`, construct the judge inside a reward, and call `evaluate()` directly. That direct-use config may leave `id` empty.

### `JudgeConfig(BaseClientConfig)`

Inherits `base_url`, `api_key_var`, and `headers` from [`BaseClientConfig`](#client-config). The default Prime endpoint, key, and team header use the same Prime CLI/environment fallback as the rollout client. Subclass `JudgeConfig` for additional knobs needed by a custom or plugin judge.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `id` | `ID` | `""` | Judge plugin id: built-in (`reference`, `rubric`), local package, or Hub package. Required in `TaskConfig.judges`; empty is allowed for direct calls from task code. |
| `name` | `str` | `""` | Reward-key override for a plugged judge. When empty, the plugin id supplies the key. |
| `weight` | `float` | `1.0` | Weight applied when the plugged judge records its verdict into aggregate `trace.reward`. |
| `model` | `str` | `"openai/gpt-5.4-nano"` | Judge model id. |
| `sampling` | `JudgeSamplingConfig` | `JudgeSamplingConfig()` | Per-call sampling defaults; individual calls may override them. |
| `prompt` | `str \| None` | `None` | Inline prompt-template override for this configured judge instance. |
| `prompt_file` | `Path \| None` | `None` | Load the prompt template from a UTF-8 text file. Mutually exclusive with `prompt`. |

### `JudgeSamplingConfig(SamplingConfig)`

The same extensible shape as the rollout's [`SamplingConfig`](#sampling-config): `temperature`, `top_p`, `reasoning_effort`, and `max_tokens`, plus provider-specific keys because `extra='allow'`. Values passed directly to `complete()` override these configured defaults.

### Judge class behavior

A judge class may define:

- `prompt: str | None` — the default template formatted by `build_messages(**fields)`. A configured
  `prompt` or `prompt_file` overrides it for that instance.
- `schema: type[BaseModel] | None` — a Pydantic schema for structured output. `evaluate()` sends it
  through the OpenAI-compatible parsed-completion path and places the validated object on `JudgeResponse.parsed`; without a schema, `parse()` receives the text response.

`evaluate()` renders the prompt, performs one judge completion, and calls `parse()`. When a trace is supplied, billed judge usage is recorded even if parsing later fails. Plugin judges implement `score(task, trace)` so `Task.score` can record the verdict under the configured key and weight.

---

## Rollout limits

`verifiers/v1/interception/server.py` — `RolloutLimits` (frozen dataclass). Not directly user-settable; built from the `max_*` fields of the run's agent ([`AgentConfig`](#agent-config)). Checked before each turn is served; the first limit reached refuses the turn (the same mechanism as a `@stop`) and becomes the trace's stop condition. Token caps are **soft by one turn** (the turn that crosses a cap still completes).

| Field | Type | Default | Maps from | Caps |
| --- | --- | --- | --- | --- |
| `max_turns` | `int \| None` | `None` | `AgentConfig.max_turns` | `trace.num_turns` |
| `max_input_tokens` | `int \| None` | `None` | `AgentConfig.max_input_tokens` | `trace.num_input_tokens` |
| `max_output_tokens` | `int \| None` | `None` | `AgentConfig.max_output_tokens` | `trace.num_output_tokens` |
| `max_total_tokens` | `int \| None` | `None` | `AgentConfig.max_total_tokens` | `trace.num_total_tokens` |

---

## ServeConfig — the env-server CLI

`verifiers/v1/configs/serve.py` — `ServeConfig(EnvServerConfig)`. The env-server CLI. Inherits the `env` block + pool, so `--env.*` / `--pool.*` are the same flags as eval. Adds only CLI-specific serving knobs.

| Field | Type | Default | Aliases | Notes |
| --- | --- | --- | --- | --- |
| `address` | `str` | `"tcp://127.0.0.1:5000"` | `address`, `a` | ZMQ address the ROUTER binds. |
| `verbose` | `bool` | `False` | `verbose`, `v` | Log at debug level. |
| `dry_run` | `bool` | `False` | — | Resolve + validate and dump, then exit. |

Plus all inherited `EnvServerConfig` fields (`env`, `pool`, legacy).

---

## ValidateConfig — the validate CLI

`verifiers/v1/configs/validate.py` — `ValidateConfig(BaseConfig)`. Model-free validation has no harness, model client, or sampling. For every selected task, the default mode runs two independent checks in separate fresh runtimes:

1. **gold:** start the runtime, run `Task.setup`, call `Task.validate`, then tear down;
2. **setup:** start another runtime, run `Task.setup`, mark it valid if setup returns, then tear down.

The independent result separates basic provisioning/setup viability from the task's gold-check result. A task whose `validate()` returns `False` is `invalid`; a timeout is `timeout`; an exception is `error`. The base `Task.validate()` returns `True`, so tasks without a custom gold check still receive the setup checks.

Use `only_gold` or `only_setup` to select one mode; setting both is rejected. The CLI writes no config or traces to disk. Its output is the live dashboard when `rich` is enabled, otherwise one log line per task.

| Field | Type | Default | Aliases | Notes |
| --- | --- | --- | --- | --- |
| `taskset` | `TasksetConfig` | `TasksetConfig()` | — | Selected by `--taskset.id` or the bare positional id. Narrowed to the concrete taskset config before validation and serialized as that subclass. |
| `runtime` | `RuntimeConfig` | `DockerConfig()` | — | Runtime used for setup and validation. Docker is the default because gold checks often need the row's image; use subprocess only when no selected task requires a container. Row image/workdir/resources are resolved as described above. |
| `timeout` | `CheckTimeoutConfig` | `CheckTimeoutConfig()` | — | Nested setup and check budgets; see below. |
| `only_setup` | `bool` | `False` | — | Run only the independent setup check. Mutually exclusive with `only_gold`. |
| `only_gold` | `bool` | `False` | — | Run only setup followed by `Task.validate` in one runtime. Mutually exclusive with `only_setup`. |
| `num_tasks` | `int \| None` | `None` | `num_tasks`, `n`, `num_examples`, `batch_size` | Number of tasks after optional shuffling; `None` means all. |
| `shuffle` | `bool` | `False` | `shuffle`, `s` | Deterministically shuffle before taking the first `num_tasks`. |
| `max_concurrent` | `int \| None` | `128` | `max_concurrent`, `c` | Maximum checks in flight, and therefore the maximum live containers/sandboxes. `None` disables the semaphore. |
| `verbose` | `bool` | `False` | `verbose`, `v` | Log at debug rather than info level. |
| `rich` | `bool` | `True` | — | Show one live dashboard row per task. Disable for per-task log lines. |

### `CheckTimeoutConfig`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `setup` | `float \| None` | `None` | Maximum seconds for `Task.setup`. When unset, validation falls back to the row's `TaskData.timeout.setup`; if that is also `None`, setup is unlimited. |
| `total` | `float \| None` | `None` | Maximum seconds for the gold check's `Task.validate` call. `None` means no limit. The setup-only mode has no action after setup, so this field does not apply there. |

---

## Notes & conventions

- **Plugin resolution.** The run's `env` field narrows to the selected env's config class
  (`--env.id`, else the taskset's exported env, else `SingleAgentEnvConfig`) *before* validation; inside it, `taskset` narrows by its id and each seat by its `harness` id. Entries in `TaskConfig.judges` are similarly narrowed by each judge `id`. This is why local and Hub plugin fields remain typed and appear in CLI validation instead of living in an untyped arguments dictionary.
- **Dotted flags.** Every nested field is part of the same CLI tree
  (`--env.agent.runtime.type docker`, `--env.taskset.split test`, `--pool.max_workers 8`, `--env.agent.retries.max_retries 2`). An `@ file.toml` describes the identical tree; explicit CLI values layer over values loaded from the file.
- **Runtime precedence.** An explicit, non-default CLI/TOML `workdir` or resource field wins over
  `TaskData`; otherwise a non-`None` row value fills it, and otherwise the runtime/provider default remains. `TaskData.image` is the required image for that row and replaces the runtime's base image. Unsupported resource fields are ignored; evaluation warns once per runtime/field.
- **Timeout precedence.** For eval stages, a non-`None` agent-level `TimeoutConfig` value
  (`--env.<agent>.timeout.*`) wins over the corresponding `TaskData.timeout` value; if both are `None`, there is no framework timeout. The setup value is one deadline shared by task setup and harness provisioning. Validate uses `CheckTimeoutConfig.setup`, then falls back to `TaskData.timeout.setup`, while `CheckTimeoutConfig.total` independently bounds `Task.validate`.
- **Discriminated unions** are selected by their `type` field: `client.type` (eval|train), `pool.type` (static|elastic), `env.<agent>.runtime.type` / `runtime.type` (subprocess|docker|prime|modal).
- **Frozen models.** `TaskData`, `TaskResources`, and `TaskTimeout` are immutable wire input, not
  mutable runtime state. Put per-rollout coordination on typed `trace.state`. `RolloutLimits` is an immutable framework limit derived from the env's `max_*` fields.
- **Legacy v0.** Set the run-level `id` (leave `env.taskset` unset) to run a classic `load_environment` env through the bridge. `--resume` is not supported for legacy evals.
