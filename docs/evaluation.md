# Evaluation

For new model evaluations, use [`uv run eval` in prime-rl](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/eval.md).

To evaluate an installed taskset, run `vf-eval`:

```bash
uv run vf-eval primeintellect/terminal-bench-2
```

The taskset package must already be installed in your environment (passing an ID like `owner/name@version` loads the installed `name` package rather than downloading or pinning that version). Because tasksets and harnesses expose their own CLI flags, include their IDs when passing `--help` to see all available options:

```bash
uv run vf-eval my-task --env.agent.harness.id codex --help
```

You can also configure a run with a `.toml` file:

```toml
model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"

[sampling]
temperature = 1.0

[env.taskset]
id = "primeintellect/terminal-bench-2"

[env.agent.harness]
id = "codex"
version = "0.116.0"

[env.agent.runtime]
type = "docker"
```

Validate the config with `uv run vf-eval @ config.toml --dry-run`, and run the evaluation with `uv run vf-eval @ config.toml`. Dotted CLI flags such as `--sampling.temperature 0.5` override values from the `.toml` file when both are present.

By default, evaluation outputs are written to `output_dir / run.dir`, which resolves to `outputs/<env>--<model>--<harness>--<short-id>/` (where `<env>` is the taskset name, prefixed by the paired env ID when `--env.id` sets one). Each run directory contains:

```text
configs/eval.toml           # launch TOML, when supplied
configs/resolved/eval.json  # full resolved config, usable with @
traces.jsonl               # one complete episode per line
logs/attempt_1/eval.log     # run and worker logs for this launch
logs/latest                # symlink to the latest attempt
```

## Model endpoints

By default, the client connects to Prime Inference and authenticates with `PRIME_API_KEY`, falling back to your active Prime CLI credentials. To point `vf-eval` at another compatible endpoint, set `base_url` and the name of the environment variable holding your API key:

```toml
[client]
base_url = "http://localhost:8000/v1"
api_key_var = "MODEL_API_KEY"
```

The endpoint must support whichever API the harness uses. In multi-agent runs, each agent can override `model`, `client`, and `sampling` under `env.<agent>`, while task [judges](tasksets.md#using-judges) configure their own client settings independently.

To pass secret headers from environment variables or customize HTTP timeouts, use `client.headers_from_env` and `client.timeout`:

```toml
[client.headers_from_env]
"X-Project-Token" = "PROJECT_TOKEN"

[client.timeout]
connect = 30
read = 300
```

Unlike static `client.headers` (which are written into the saved resolved config), `headers_from_env` reads environment variables when the client starts and overrides any matching static headers. HTTP timeouts are in seconds: `connect` defaults to `30`, while `read`, `write`, and `pool` have no limit by default. Because `read` only bounds the interval between received chunks rather than the total duration of a response, use agent stage timeouts to cap a full rollout.

To cap concurrent requests per model across every process on the host (env-server workers and concurrent runs alike), set `client.request_gate`. Requests beyond the cap wait in one first-come, first-served queue per endpoint, API key and model:

```toml
[client.request_gate]
max_concurrent = 32
```

By default the gate is adaptive: a 429 or 529 from the provider lowers the gate's limit by a quarter (at most once every 2 seconds), and the limit climbs back to `max_concurrent` over the next 60 seconds. The refused request waits (`Retry-After`, else exponential backoff) and queues again, up to `max_rejections = 10` times. Set `adaptive = false` for a provider that queues requests instead of refusing them.

`uv run vf-gate` shows each gate's cap, current limit, refusals and requests in flight and waiting; `uv run vf-gate --resize 48 --model M` changes the cap of running gates until no process uses them.

## Sampling settings

Shared generation parameters go under `[sampling]`, with per-agent overrides under `[env.<agent>.sampling]`. Overrides merge field by field, so setting an agent's `temperature` preserves the shared token limit and provider options (pass `None` in Python or `null` in JSON to clear an inherited field).

Alongside standard parameters like `temperature`, `top_p`, `reasoning_effort`, and `max_tokens` (or its alias `max_completion_tokens`), you can pass provider-specific options either directly in `[sampling]` or inside `extra_body`.

## Resuming evaluations

To resume an interrupted or partially failed evaluation, re-run its saved resolved config with `--resume`:

```bash
uv run vf-eval @ <run-dir>/configs/resolved/eval.json --resume
```

Completed episodes are preserved in the run directory, while missing, errored, or malformed episodes run again in place. To discard the existing run directory and start over instead, pass `--clean` (or write to a new directory to keep the previous results).

Because `--resume` matches completed episodes to tasks by their `TaskData` hash (filling each occurrence's `num_rollouts` slots in order when identical task data appears more than once), the installed taskset, code, and dataset must stay unchanged across runs. Overriding `Task.key` does not replace hash matching: any change to `TaskData`, including its assigned `idx`, produces a new hash and causes the task to run again.

## Timeouts and retries

Full-rollout retries are disabled by default (separate from HTTP retries inside the provider SDK). You can configure retry budgets per error type under `[env.agent.retries]`, or under `[env.retries]` for environments that manage shared resources such as Harbor Compose:

```toml
[env.agent.retries]
max_retries = 0

[[env.agent.retries.rules]]
type = "ProviderError"
status_code = [429, "5xx"]
max_retries = 2

[[env.agent.retries.rules]]
type = "SandboxError"
max_retries = 2
```

When an agent rollout fails, `rules` are checked in order and the first rule where all fields match sets the retry budget for that error; unmatched errors use the top-level `max_retries` budget. Each retry starts a fresh attempt from the beginning. To bound how long individual stages or rollouts can run, configure stage timeouts and token limits on the [runtime](runtimes.md); see [debugging](debugging.md) for diagnosing failed episodes.

## Disabling tools

Most harnesses expose a `disabled_tools` list that you can use to turn off specific built-in tools:

```toml
[env.agent.harness]
disabled_tools = ["shell_tool"]
```

Tool names and support for disabling them depend on the harness; check the harness documentation for the exact names it accepts.

## Skills

Harnesses whose underlying program natively supports `SKILL.md` files (such as Claude Code and Codex) accept a `skills` list. Passing a local skill directory uploads it to the runtime, while `{runtime = "..."}` copies skills from a directory already inside the runtime into the harness's skill discovery path:

```toml
[env.agent.harness]
skills = [{runtime = "/opt/skills"}, "path/to/my-skill"]
```

Tasks can also declare skills via `TaskData.skills` (for example, `skills=[{"runtime": "/opt/skills"}]`). Task skills are installed first, followed by harness skills, with later files overwriting earlier ones at the same path. Each rollout gets its own copy after task setup completes, and missing source directories (or configuring `skills` on a harness that does not support them) will fail the run.
