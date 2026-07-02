# REFERENCE.md

A complete reference of every settable config field for **evaluating environments in `verifiers.v1`**. The config tree is parsed from CLI flags (dotted, e.g. `--harness.runtime.type docker`) and/or `@ file.toml` by `prime-pydantic-config`; every field below is settable either way unless noted.

The root config the eval CLI parses is [`EvalConfig`](#evalconfig--the-run). It inherits the full environment config (taskset + harness + timeouts + token limits + worker pool), then adds the run knobs (model, sampling, counts). The tree:

```
EvalConfig                       (the run + the env)
├─ model, sampling, client, num_tasks, num_rollouts, shuffle, max_concurrent, …
├─ taskset: TasksetConfig        (subclass resolved by --taskset.id)
├─ harness: HarnessConfig        (subclass resolved by --harness.id)
│  └─ runtime: RuntimeConfig     (subprocess | docker | prime | modal)
├─ timeout: TimeoutConfig
├─ retries: RetryConfig
│  └─ rollout: RolloutRetryConfig
├─ max_turns / max_input_tokens / max_output_tokens / max_total_tokens
├─ multiplex
└─ pool: PoolConfig              (static | elastic) — env-server only
```

Sibling entrypoints reuse the same tree: [`ServeConfig`](#serveconfig--the-env-server-cli) (env server) and [`ValidateConfig`](#validateconfig--the-validate-cli) (per-task validation). All three live in `verifiers/v1/configs/`.

---

## EvalConfig — the run

`verifiers/v1/configs/eval.py` — `EvalConfig(EnvServerConfig)`. The single config object the eval CLI parses. Inherits [`EnvConfig`](#envconfig--the-environment) + [`EnvServerConfig`](#envserverconfig--the-pool) (so `--taskset.*`, `--harness.*`, `--pool.*`, `--timeout.*`, etc. are all top-level flags with no `--env.` prefix) and adds the run knobs.

| Field | Type | Default | Aliases | Notes |
|---|---|---|---|---|
| `uuid` | `str` | `uuid4()` | — | Auto-generated run id; the leaf of the output dir. Excluded from the saved config. |
| `model` | `str` | `"deepseek/deepseek-v4-flash"` | `model`, `m` | Model id. |
| `client` | `ClientConfig` | `EvalClientConfig()` | — | The model client (discriminated union — see [Client config](#client-config)). |
| `sampling` | `SamplingConfig` | `SamplingConfig()` | — | Per-request sampling knobs (see [Sampling config](#sampling-config)). |
| `num_tasks` | `int \| None` | `None` | `batch_size`, `num_examples`, `num_tasks`, `n` | How many tasks to evaluate (None = all). |
| `num_rollouts` | `int` | `1` | `group_size`, `rollouts_per_example`, `num_rollouts`, `r` | Rollouts per task. A taskset with `@group_reward`s requires ≥ 2. |
| `shuffle` | `bool` | `False` | `shuffle`, `s` | Shuffle tasks before taking the first `num_tasks`. |
| `max_concurrent` | `int \| None` | `128` | `max_concurrent`, `c` | Max rollouts in flight at once. |
| `verbose` | `bool` | `False` | `verbose`, `v` | Log at debug level instead of info. |
| `dry_run` | `bool` | `False` | — | Resolve + validate the config and dump it, then exit. |
| `rich` | `bool` | `True` | — | Live dashboard instead of per-rollout logs (in-process only). |
| `server` | `bool` | `False` | — | Drive rollouts through the env-server worker pool (sized by `pool`) instead of in-process — the path prime-rl trains through. Incompatible with `--rich`. |
| `output_dir` | `Path \| None` | `None` | `output_dir`, `o` | Where to write the run (`config.toml` + `results.jsonl`). None = a fresh per-run dir under `outputs/<env>--<model>--<harness>/<uuid>`. |
| `resume` | `Path \| None` | `None` | — | Set by `--resume <dir>`: re-run only the rollouts a previous run left missing/errored. Excluded from the saved config; takes no other args. |

Validator: `--rich` + `--server` together is rejected (the dashboard is in-process only).

Inherited from `EnvConfig`: [`taskset`](#taskset-config), [`harness`](#harness-config), [`timeout`](#timeout-config), [`retries`](#retry-config), `max_turns`, `max_input_tokens`, `max_output_tokens`, `max_total_tokens`, [`multiplex`](#envconfig--the-environment), the legacy `id` / `args` / `extra_env_kwargs`.
Inherited from `EnvServerConfig`: [`pool`](#pool-config).

---

## Sampling config

`verifiers/v1/types.py` — `SamplingConfig(BaseModel)` (alias `Sampling`). Used as `EvalConfig.sampling` and embedded in [`JudgeSamplingConfig`](#judge-config). `extra='allow'`, so provider-specific keys pass through.

| Field | Type | Default | Aliases | Notes |
|---|---|---|---|---|
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
|---|---|---|---|
| `base_url` | `str` | `https://api.pinference.ai/api/v1` | OpenAI-compatible endpoint. Falls back to the active Prime CLI config (or `PRIME_INFERENCE_URL`). |
| `api_key_var` | `str` | `"PRIME_API_KEY"` | Env var the key is read from. |
| `headers` | `dict[str, str]` | `{}` | Extra HTTP headers on every request. `X-Prime-Team-ID` auto-set for pinference hosts. |

### `EvalClientConfig(BaseClientConfig)` — `type: "eval"` (default)
The default: forward each request to a matching endpoint. No extra fields.

### `TrainClientConfig(BaseClientConfig)` — `type: "train"`
A vLLM `/inference/v1/generate` endpoint with client-side tokenization (responses carry token ids + logprobs). Needs a running vLLM engine.

| Field | Type | Default | Notes |
|---|---|---|---|
| `renderer` | `RendererConfig \| None` | `None` | The `renderers.RendererConfig`. `None` auto-resolves from the model (falls back to the default renderer — no tool support — for unknown models). |
| `pool_size` | `int` | `1` | Renderer slots shared across concurrent rollouts (client-side tokenization). |
| `renderer_model_name` | `str \| None` | `None` | Model the tokenizer/renderer pool is built for. Pin to the base model so a LoRA adapter name never drives tokenizer loading. Falls back to the per-request model. |

---

## EnvConfig — the environment

`verifiers/v1/env.py` — `EnvConfig(BaseConfig)`. The two peers of a rollout: the taskset (data + scoring) and the harness (which program drives it, and where it runs — `harness.runtime`).

| Field | Type | Default | Notes |
|---|---|---|---|
| `taskset` | `TasksetConfig` | `TasksetConfig()` | Resolved to its concrete subclass by `--taskset.id` (see [Taskset config](#taskset-config)). `SerializeAsAny` so subclass fields survive `model_dump`. |
| `harness` | `HarnessConfig` | `HarnessConfig(id="default")` | Resolved to its concrete subclass by `--harness.id` (or the taskset's bundled harness). See [Harness config](#harness-config). |
| `timeout` | `TimeoutConfig` | `TimeoutConfig()` | See [Timeout config](#timeout-config). |
| `retries` | `RetryConfig` | `RetryConfig()` | See [Retry config](#retry-config). |
| `max_turns` | `int \| None` | `None` | Max model turns per rollout (None = no limit). Framework-enforced between turns. |
| `max_input_tokens` | `int \| None` | `None` | Max input (prompt) tokens per rollout. Caps `trace.num_input_tokens`. |
| `max_output_tokens` | `int \| None` | `None` | Max output (completion) tokens per rollout. Caps `trace.num_output_tokens`. |
| `max_total_tokens` | `int \| None` | `None` | Max total (prompt + completion) tokens per rollout. Caps `trace.num_total_tokens`. |
| `multiplex` | `int` | `32` (≥1) | Rollouts that share one interception server (and, behind a remote runtime, one tunnel). N concurrent rollouts use ~N/multiplex servers + tunnels. 1 = a server per rollout. |

The four `max_*` limits map onto [`RolloutLimits`](#rollout-limits) (interception server); each caps a trace computed property, checked between turns (soft by one turn).

### Legacy (v0) backwards-compat fields
Set `id` (leave `taskset` unset) to run a classic `verifiers.load_environment` env through the legacy bridge.

| Field | Type | Default | Notes |
|---|---|---|---|
| `id` | `EnvId \| None` | `None` | Classic v0 env id (`name`, `org/name`, or `org/name@version`). |
| `args` | `dict` | `{}` | Construction kwargs forwarded to `load_environment(id, **args)`. |
| `extra_env_kwargs` | `dict` | `{}` | Post-load kwargs applied via `env.set_kwargs(**...)` (e.g. `max_total_completion_tokens`, `max_seq_len`, `timeout_seconds`). |

`EnvConfig.is_legacy` → `id is not None and not taskset.id`.

### EnvServerConfig — the pool

`EnvServerConfig(EnvConfig)`. Adds the env-server worker pool sizing. Shared by the `serve` CLI, server-backed eval, and prime-rl's orchestrator.

| Field | Type | Default | Notes |
|---|---|---|---|
| `pool` | `PoolConfig` | `ElasticPoolConfig()` | See [Pool config](#pool-config). |

---

## Timeout config

`verifiers/v1/env.py` — `TimeoutConfig(BaseConfig)`. Framework-enforced wall-clock timeouts per rollout stage, in seconds (None = no limit). Precedence: cli/toml > per-task [`TaskTimeout`](#task-resources--timeouts) > default.

| Field | Type | Default | Notes |
|---|---|---|---|
| `setup` | `float \| None` | `None` | Max wall-clock for the taskset's `setup` hook. |
| `rollout` | `float \| None` | `None` | Max wall-clock for the rollout (the harness run). |
| `finalize` | `float \| None` | `None` | Max wall-clock for the taskset's `finalize` hook. |
| `scoring` | `float \| None` | `None` | Max wall-clock for scoring — verify + rewards/metrics. |

> Remote sandboxes cap any harness timeout at 24 hours (provider max lifetime).

---

## Retry config

`verifiers/v1/retries.py`. Per-call model/runtime retries are owned by the harness/runtime SDKs; the framework keeps only **whole-rollout** retries.

### `RetryConfig`
| Field | Type | Default | Notes |
|---|---|---|---|
| `rollout` | `RolloutRetryConfig` | `RolloutRetryConfig()` | See below. |

### `RolloutRetryConfig`
Rerun the whole trajectory when it ends with a captured error. Matching is by the error's **exception type name**.

| Field | Type | Default | Notes |
|---|---|---|---|
| `max_retries` | `int` | `0` (≥0) | Whole-rollout retries beyond the first attempt (0 = no retry). |
| `include` | `list[str]` | `[]` | Only retry errors whose type is listed. Empty = retry anything not excluded. |
| `exclude` | `list[str]` | `[]` | Never retry these types (wins over `include`). |

---

## Pool config

`verifiers/v1/env.py`. Discriminated on `type`; selected with `--pool.type static|elastic`. Drives the env-server worker pool (the `--server` path).

### `StaticPoolConfig` — `type: "static"`
Fixed pool: pre-spawn `num_workers` up front.

| Field | Type | Default | Notes |
|---|---|---|---|
| `num_workers` | `int` | `4` (≥1) | Worker processes to pre-spawn (1 = a single in-process server, no pool). |

### `ElasticPoolConfig` — `type: "elastic"` (default)
Elastic pool: start at one worker and scale up on demand.

| Field | Type | Default | Notes |
|---|---|---|---|
| `max_workers` | `int \| None` | `None` | Upper bound on workers (None = unbounded). |
| `multiplex` | `int` | `128` (≥1) | Rollouts per worker for the scale-up trigger: add a worker once in-flight rollouts reach 90% of `workers * multiplex`. |

---

## Taskset config

`verifiers/v1/taskset.py` — `TasksetConfig(BaseConfig)`. The base; **subclass per taskset to add task-generation knobs** (e.g. `split`, `difficulty`). The concrete subclass is resolved by `--taskset.id`, so a taskset's own fields become dotted flags (`--taskset.split test`).

| Field | Type | Default | Notes |
|---|---|---|---|
| `id` | `EnvId` | `""` | The taskset id, which selects it: a local package, or `org/name[@version]` from the Environments Hub. Set via `--taskset.id`. |

`.name` → the package name (id with org / version stripped).

A taskset class also declares capabilities that affect config validation (not user-settable fields):
- `NEEDS_CONTAINER: ClassVar[bool]` — refuse the subprocess runtime when True.

See [`TaskResources`](#task-resources--timeouts) and [`TaskTimeout`](#task-resources--timeouts) for what a *task* (not the taskset) can carry.

---

## Harness config

`verifiers/v1/harness.py` — `HarnessConfig(BaseConfig)`. The base; **subclass per harness to add run knobs**. The concrete subclass is resolved by `--harness.id` (or a taskset's bundled harness). Mirrors `TasksetConfig`.

### Base `HarnessConfig`
| Field | Type | Default | Notes |
|---|---|---|---|
| `id` | `EnvId` | `"default"` | The harness id, which selects it. Set via `--harness.id`. |
| `runtime` | `RuntimeConfig` | `SubprocessConfig()` | Where the harness runs. Discriminated union — see [Runtime configs](#runtime-configs). Set with `--harness.runtime.type docker\|prime\|modal`. |
| `env` | `dict[str, str]` | `{}` | Additional env vars for the harness program. Harness-owned endpoint/auth/model vars take precedence. |
| `forward_env` | `list[str]` | `[]` | Names of env vars to forward from `os.environ` into the harness program's runtime (for secrets not in checked-in config). Absent names are skipped; explicit `env` wins. |
| `disabled_tools` | `list[str] \| None` | `None` | Harness-specific tool names to disable. |

`.name` → the package name; `.resolved_env` → `env` merged with forwarded `forward_env` vars.

A harness class also declares capability flags (ClassVars, not user-settable):
`APPENDS_SYSTEM_PROMPT`, `SUPPORTS_MCP`, `SUPPORTS_USER_SIM`, `SUPPORTS_MESSAGE_PROMPT`.

### Built-in harness configs

All inherit the base `HarnessConfig` fields (`id`, `runtime`, `env`, `forward_env`, `disabled_tools`).

#### `DefaultHarnessConfig` — `id: "default"` (the fallback)
A growing-message-list chat loop with a local `bash` tool, plus optional `edit`/`search`. A uv script (deps: `openai`, `mcp`).

| Field | Type | Default | Notes |
|---|---|---|---|
| `edit` | `bool` | `True` | Offer the local `edit` tool (single-occurrence string replacement) alongside `bash`. |
| `search` | `bool` | `False` | Offer a `search` tool (Google web results via serper.dev). Requires `SERPER_API_KEY` in the eval environment. |

#### `NullHarnessConfig` — `id: "null"`
A pure chat-loop program with the taskset's MCP tools and **no tools of its own**. A uv script (deps: `openai`, `mcp`). No extra fields.

#### `CodexHarnessConfig` — `id: "codex"`
Installs the Codex CLI into the runtime and runs `codex exec`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `version` | `str` | `"0.137.0"` | Codex release to install (the `rust-v<version>` GitHub release); pinned. |

#### `RLMHarnessConfig` — `id: "rlm"`
Installs the rlm CLI and runs it. Knobs map onto `RLM_*` env vars; base `HarnessConfig.env` passes any other `RLM_*` var through verbatim.

| Field | Type | Default | Notes |
|---|---|---|---|
| `version` | `str` | `"main"` | Git ref (branch/tag/commit) of rlm to install. |
| `max_depth` | `int` | `0` | Recursion depth rlm may spawn sub-harnesses to (`RLM_MAX_DEPTH`). |
| `skills` | `list["edit" \| "search"]` | `[]` | Built-in rlm skills to enable (`RLM_SKILLS`). Empty enables none. |
| `summarize_at_tokens` | `int \| (int, int) \| None` | `None` | Auto-compaction threshold (`RLM_SUMMARIZE_AT_TOKENS`): compact once context grows past this many tokens. An int is fixed; a `(lo, hi)` pair draws a per-group threshold (seeded by task index). `None` disables. Ints must be positive. |

#### `MiniSWEAgentHarnessConfig` — `id: "mini-swe-agent"`
Runs the native bash-tool agent through LiteLLM.

| Field | Type | Default | Notes |
|---|---|---|---|
| `version` | `str` | `"2.2.8"` | mini-swe-agent release to install, pinned. |

#### `Terminus2HarnessConfig` — `id: "terminus-2"`
Runs Harbor's tmux agent through LiteLLM.

| Field | Type | Default | Notes |
|---|---|---|---|
| `version` | `str` | `"0.14.0"` | Harbor release to install, pinned. |

#### `KimiCodeHarnessConfig` — `id: "kimi-code"`
Installs the Kimi Code CLI and runs it headlessly.

| Field | Type | Default | Notes |
|---|---|---|---|
| `version` | `str` | `"0.14.3"` | Kimi Code release to install, pinned. |

---

## Runtime configs

`verifiers/v1/runtimes/`. Discriminated on `type`; selected with `--harness.runtime.type` (or `--runtime.type` for the validate CLI). The same union is reused as `ToolsetConfig.runtime` and `UserConfig.runtime`.

### `SubprocessConfig` — `type: "subprocess"` (default)
Run on the host in a fresh `/tmp/<name>` workspace per rollout. **No extra fields.** Inherits the host env *except* any var whose name contains `API_KEY`.

> Tool servers placed on subprocess can't receive host API keys (the strip applies to every program run here); give a key-needing tool its own runtime or have it fetch the key itself.

### `DockerConfig` — `type: "docker"`
Local Docker container sharing the host network (`--network host`).

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | `str` | `"python:3.11-slim"` | Container image. |
| `workdir` | `str` | `"/app"` | Working directory. |
| `cpu` | `float \| None` | `None` | Pin to this many CPU cores (`docker --cpus`). None = unlimited. |
| `memory` | `float \| None` | `None` | Hard memory limit in GB (`docker --memory`). None = unlimited. |
| `gpu` | `str \| None` | `None` | GPU spec, e.g. `"A100"` or `"2"` (`docker --gpus` uses the count; needs the nvidia toolkit). |
| `disk` | `float \| None` | `None` | Advisory disk request in GB. Docker has no portable per-container size limit, so accepted but **not enforced**. |

### `PrimeConfig` — `type: "prime"`
Remote Prime sandbox; reached via native port exposure.

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | `str` | `"python:3.11-slim"` | Container image. |
| `workdir` | `str` | `"/app"` | Working directory. |
| `network_access` | `bool` | `True` | Allow outbound network from the sandbox. |
| `vm` | `bool` | `False` | Run as a micro-VM (kernel features / stronger isolation). |
| `guaranteed` | `bool` | `False` | Request guaranteed (vs best-effort) capacity. |
| `region` | `str \| None` | `None` | Region to provision in (None = provider-chosen). Note: port exposure is region-gated; `us` supports it. |
| `labels` | `list[str]` | `[]` | Labels for the sandbox and its tunnels. The eval defaults them to the run's uuid when unset. |
| `cpu` | `float` | `1.0` | CPU cores. |
| `memory` | `float` | `2.0` | Memory in GB. |
| `gpu` | `str \| None` | `None` | GPU spec, e.g. `"A100"` or `"A100:2"` (bare count = provider-chosen type). |
| `disk` | `float` | `5.0` | Disk in GB. |
| `creates_per_min` | `int \| None` | `None` | Pace sandbox creation to this many per minute, host-wide across every env-server worker (None/≤0 disables). Tunnel creation is limited separately and globally. |

### `ModalConfig` — `type: "modal"`
Remote Modal sandbox; reached via Modal's own port forwarding (`encrypted_ports`).

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | `str` | `"python:3.11-slim"` | Container image (registry). |
| `workdir` | `str` | `"/app"` | Working directory. |
| `network_access` | `bool` | `True` | Allow outbound network (`block_network` is the negation). |
| `region` | `str \| None` | `None` | Region to provision in (None = provider-chosen). |
| `cpu` | `float` | `1.0` | CPU cores. |
| `memory` | `float` | `2.0` | Memory in GB (sent to Modal as MB). |
| `gpu` | `str \| None` | `None` | GPU spec, e.g. `"A100"` or `"A100:2"`. |
| `disk` | `float` | `5.0` | Disk in GB. Modal sandboxes have no disk knob, so **accepted but not enforced**. |
| `creates_per_sec` | `float \| None` | `40.0` | Pace sandbox creation to this many per second, host-wide across every env-server worker (None/≤0 disables). |

All four `image`/`workdir`/`resource` fields are overridable per task via [`Task.image`](#task-resources--timeouts) / `Task.workdir` / [`TaskResources`](#task-resources--timeouts), with precedence cli/toml > task > runtime default.

---

## Task resources & timeouts

`verifiers/v1/task.py`. A `Task` is an immutable, frozen pydantic model; environments subclass it to add typed task-specific fields (the reference answer, ground truths, …). These fields flow — fully typed — through the rollout into scoring. Not CLI flags (they come from `Taskset.load_tasks`), but they affect runtime resolution, so they're part of the reference.

### `TaskResources` (frozen)
Runtime resources a task requests, in Modal's units. Applied where the runtime supports the field; an unsupported field is warned about and ignored. Precedence: cli/toml > task > runtime default (`None` here = use the runtime/provider default).

| Field | Type | Default | Notes |
|---|---|---|---|
| `cpu` | `float \| None` | `None` | CPU cores. |
| `memory` | `float \| None` | `None` | Memory in GB. |
| `gpu` | `str \| None` | `None` | GPU spec, e.g. `"A100"` or `"A100:2"` (type[:count]). |
| `disk` | `float \| None` | `None` | Disk in GB (enforced by prime; advisory on docker/modal). |

### `TaskTimeout` (frozen)
Per-task wall-clock timeout overrides (seconds), one per rollout stage. Each merges with the eval's `timeout` ([`TimeoutConfig`](#timeout-config)): cli/toml > this > default (no limit).

| Field | Type | Default | Notes |
|---|---|---|---|
| `setup` | `float \| None` | `None` | The taskset's `setup` hook. |
| `harness` | `float \| None` | `None` | The harness run. |
| `finalize` | `float \| None` | `None` | The taskset's `finalize` hook. |
| `scoring` | `float \| None` | `None` | Verify + rewards/metrics. |

### `Task` (frozen)
| Field | Type | Default | Notes |
|---|---|---|---|
| `idx` | `int` | — | Stable integer index within its taskset. |
| `name` | `str \| None` | `None` | Optional human-readable label. |
| `description` | `str \| None` | `None` | Optional human-readable description. |
| `prompt` | `str \| Messages \| None` | — | The user message. A `Messages` list seeds a full initial conversation (only accepted by harnesses with `SUPPORTS_MESSAGE_PROMPT`). `None` → the user simulator opens. |
| `system_prompt` | `str \| None` | `None` | Optional system prompt. Harnesses with `APPENDS_SYSTEM_PROMPT` emit it as a real system message; others prepend to `prompt` (with a warning). |
| `image` | `str \| None` | `None` | Container image this task needs. When set, the runtime must be a container (docker/prime); subprocess is refused. |
| `workdir` | `str \| None` | `None` | Working directory for the harness and scoring. Injected where the runtime supports one. |
| `timeout` | `TaskTimeout` | `TaskTimeout()` | Per-task timeout overrides. |
| `resources` | `TaskResources` | `TaskResources()` | Runtime resources requested. |

---

## Toolset config

`verifiers/v1/mcp/toolset.py` — `ToolsetConfig(BaseConfig)`. Where one tool server runs (placement). A taskset declares `Toolset`s from `Taskset.tools`; each carries its `config`. **Subclass to add the server's own knobs** (the data its `@vf.tool` methods read).

| Field | Type | Default | Notes |
|---|---|---|---|
| `colocated` | `bool` | `False` | Run inside the harness's OWN runtime (reached in-sandbox, no tunnel). In a sandbox this uploads + installs the env package + `verifiers` (a per-rollout cost). Mutually exclusive with `shared`. |
| `shared` | `bool` | `False` | One instance for the whole eval, in its own `runtime` (pays an expensive `setup` once). Each rollout still reads/writes its OWN `self.state`. Mutually exclusive with `colocated`. |
| `fork` | `bool` | `False` | For a `shared` server: fork a child per rollout (copy-on-write), isolating module globals / mutated in-memory objects / relative-path on-disk writes. Requires `shared`; Linux/fork only; not for CUDA/GPU state or background threads. |
| `runtime` | `RuntimeConfig` | `SubprocessConfig()` | The server's own runtime, used unless `colocated`. Host/subprocess by default (always reachable from any harness). See [Runtime configs](#runtime-configs). |
| `url` | `str \| None` | `None` | An already-running streamable-HTTP MCP endpoint to connect to instead of launching a server. When set, placement is ignored and the toolset needs no `@vf.tool` methods. |

Validators: `colocated` + `shared` mutually exclusive; `fork` requires `shared`.

---

## User config

`verifiers/v1/mcp/user.py` — `UserConfig(BaseConfig)`. Where the user simulator runs (placement). The framework always drives it from the host. **Subclass to add the user's own knobs.**

| Field | Type | Default | Notes |
|---|---|---|---|
| `colocated` | `bool` | `False` | Run the user simulator inside the harness's runtime (its port is published back to the host so the framework can still drive it). |
| `runtime` | `RuntimeConfig` | `SubprocessConfig()` | The user simulator's own runtime, used unless `colocated`. See [Runtime configs](#runtime-configs). |

---

## Judge config

`verifiers/v1/judge.py`. A reusable per-task LLM judge. A taskset holds a `JudgeConfig` (or subclass) and constructs a `vf.Judge` from it.

### `JudgeConfig(BaseClientConfig)`
Inherits `base_url` / `api_key_var` / `headers` (with the Prime auto-config) from [`BaseClientConfig`](#client-config); adds the model + sampling. Subclass to add taskset-specific fields.

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | `str` | `"openai/gpt-5-mini"` | Judge model id. |
| `sampling` | `JudgeSamplingConfig` | `JudgeSamplingConfig()` | Per-call sampling knobs. |

### `JudgeSamplingConfig(SamplingConfig)`
Same shape as the rollout's [`SamplingConfig`](#sampling-config) — `temperature` / `top_p` / `reasoning_effort` / `max_tokens` (+ provider-specific keys via `extra='allow'`). Set e.g. `judge.sampling.max_tokens`.

`Judge` itself is configured by **class attributes** (not config fields):
- `prompt: str | None` — default template for `build_messages`, formatted with the `evaluate` kwargs.
- `schema: type[BaseModel] | None` — pydantic schema for OpenAI structured outputs (when set, `JudgeResponse.parsed` is the validated object).

---

## Rollout limits

`verifiers/v1/interception/server.py` — `RolloutLimits` (frozen dataclass). Not directly user-settable; built from the `max_*` fields of [`EnvConfig`](#envconfig--the-environment). Checked before each turn is served; the first limit reached refuses the turn (the same mechanism as a `@stop`) and becomes the trace's stop condition. Token caps are **soft by one turn** (the turn that crosses a cap still completes).

| Field | Type | Default | Maps from | Caps |
|---|---|---|---|---|
| `max_turns` | `int \| None` | `None` | `EnvConfig.max_turns` | `trace.num_turns` |
| `max_input_tokens` | `int \| None` | `None` | `EnvConfig.max_input_tokens` | `trace.num_input_tokens` |
| `max_output_tokens` | `int \| None` | `None` | `EnvConfig.max_output_tokens` | `trace.num_output_tokens` |
| `max_total_tokens` | `int \| None` | `None` | `EnvConfig.max_total_tokens` | `trace.num_total_tokens` |

---

## ServeConfig — the env-server CLI

`verifiers/v1/configs/serve.py` — `ServeConfig(EnvServerConfig)`. The env-server CLI. Inherits the full env + pool, so `--taskset.*` / `--harness.*` / `--pool.*` are the same flags as eval. Adds only CLI-specific serving knobs.

| Field | Type | Default | Aliases | Notes |
|---|---|---|---|---|
| `address` | `str` | `"tcp://127.0.0.1:5000"` | `address`, `a` | ZMQ address the ROUTER binds. |
| `verbose` | `bool` | `False` | `verbose`, `v` | Log at debug level. |
| `dry_run` | `bool` | `False` | — | Resolve + validate and dump, then exit. |

Plus all inherited `EnvServerConfig` fields (`taskset`, `harness`, `timeout`, `retries`, `max_*`, `multiplex`, `pool`, legacy).

---

## ValidateConfig — the validate CLI

`verifiers/v1/configs/validate.py` — `ValidateConfig(BaseConfig)`. Per-task, **model-free** validation: runs the taskset's `validate` hook (apply a gold solution, run a verifier) in a runtime. No harness, model, or sampling; fire-and-forget (nothing written to disk).

| Field | Type | Default | Aliases | Notes |
|---|---|---|---|---|
| `taskset` | `TasksetConfig` | `TasksetConfig()` | — | Resolved to its concrete subclass by `--taskset.id` (or a bare positional). `SerializeAsAny`. |
| `runtime` | `RuntimeConfig` | `DockerConfig()` | — | Where each task's `validate` hook runs. Docker by default (a gold check often needs the task's container); use `--runtime.type subprocess` for a check that needs no container. |
| `setup_timeout` | `float \| None` | `None` | — | Max wall-clock for the taskset's `setup` hook per task. |
| `validate_timeout` | `float \| None` | `None` | — | Max wall-clock for the `validate` hook per task. |
| `num_tasks` | `int \| None` | `None` | `num_tasks`, `n`, `num_examples`, `batch_size` | How many tasks to validate (None = all). |
| `shuffle` | `bool` | `False` | `shuffle`, `s` | Shuffle before taking the first `num_tasks`. |
| `max_concurrent` | `int \| None` | `128` | `max_concurrent`, `c` | Max tasks validated in flight at once (and, for a container runtime, live sandboxes). |
| `verbose` | `bool` | `False` | `verbose`, `v` | Log at debug level. |
| `rich` | `bool` | `True` | — | Live dashboard (one row per task) instead of per-task log lines. |

---

## Notes & conventions

- **Plugin resolution.** `taskset` and `harness` are generic base fields; `EnvConfig._resolve_plugins` narrows each to its concrete config type by `id` *before* validation, so a taskset/harness's own fields validate against the real subclass (no untyped args dict). The same narrowing happens for `ValidateConfig` (taskset only) and `ServeConfig` (inherited).
- **Dotted flags.** Every nested field is a CLI flag (`--harness.runtime.type docker`, `--taskset.split test`, `--pool.max_workers 8`, `--retries.rollout.max_retries 2`). Use `@ file.toml` for the same tree.
- **Precedence.** Runtime resources/timeouts: cli/toml > task > runtime default. Timeouts: cli/toml > per-task `TaskTimeout` > `TimeoutConfig` default. A field left at its default may be overridden by a task; a field the CLI changed wins.
- **Discriminated unions** are selected by their `type` field: `client.type` (eval|train), `pool.type` (static|elastic), `harness.runtime.type` / `runtime.type` (subprocess|docker|prime|modal).
- **Frozen models.** `Task`, `TaskResources`, `TaskTimeout`, `RolloutLimits` are immutable — they're the wire input / framework limits, not mutated at runtime.
- **Legacy v0.** Set `EnvConfig.id` (leave `taskset` unset) to run a classic `load_environment` env through the bridge. `--resume` is not supported for legacy evals.
