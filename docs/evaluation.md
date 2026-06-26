# Evaluation

Verifiers owns local evaluation configuration and execution. The Prime CLI forwards
`prime eval run` arguments unchanged to Verifiers, so the same command model is available
through the `eval` entrypoint.

## Local evaluation

A v1 evaluation combines a taskset plugin with a harness:

```bash
prime eval run owner/my-taskset \
  --harness.id default \
  --model openai/gpt-4.1-mini \
  --num-tasks 10 \
  --num-rollouts 2
```

The positional value is the taskset id. It can be an importable local package or an
`owner/name[@version]` Hub reference. Hub packages, including private packages available to
the current Prime account, are installed on demand. Install local packages explicitly:

```bash
prime env install my-taskset
prime eval run my-taskset --harness.id default
```

Use `--dry-run true` to resolve and print the typed configuration without running rollouts:

```bash
prime eval run owner/my-taskset --harness.id default --dry-run true
```

Run `prime eval run --help` for the complete typed surface. Important groups include:

- `--taskset.*`: taskset plugin configuration
- `--harness.*`: harness and runtime configuration
- `--client.*`: model endpoint configuration
- `--sampling.*`: sampling configuration
- `--timeout.*`: setup, rollout, finalization, and scoring limits
- `--retries.*`: whole-rollout retry policy
- `--pool.*`: static or elastic env-server workers

### V0 environments

V0 remains available through the explicit `--id` bridge:

```bash
prime eval run \
  --id owner/my-v0-env \
  --args '{"split":"test"}' \
  --model openai/gpt-4.1-mini \
  --num-tasks 20
```

`--args` is passed to `load_environment()`. `--extra-env-kwargs` is applied after loading.
V0 Hub references use the same authenticated acquisition path as v1 plugins.

## Configuration files

Prefix a TOML path with `@` to load the same typed configuration accepted by the CLI:

```toml
# configs/eval/math.toml
model = "openai/gpt-4.1-mini"
num_tasks = 50
num_rollouts = 2
max_concurrent = 32

[taskset]
id = "primeintellect/math-python"

[harness]
id = "default"

[sampling]
temperature = 0.2
max_tokens = 2048
```

```bash
prime eval run @ configs/eval/math.toml
```

CLI values override file values. Pydantic rejects unknown fields and invalid types before
execution.

V0 uses the same file format with a top-level `id` and optional `args`:

```toml
id = "owner/my-v0-env"
model = "openai/gpt-4.1-mini"
num_tasks = 20

[args]
split = "test"
```

## Runtimes

The built-in harness can run locally, in Docker, in a Prime sandbox, or in Modal. Select a
runtime with its discriminator and then set its typed fields:

```bash
prime eval run owner/my-taskset \
  --harness.runtime.type prime \
  --harness.runtime.cpu 2 \
  --harness.runtime.memory 4
```

Prime runtime authentication comes from the active Prime CLI context. The Prime wrapper
materializes that context for the Verifiers child process, including API, team, user, and
inference settings.

## Output and resume

Every run writes exactly two native artifacts:

```text
outputs/<taskset>--<model>--<harness>/<run-id>/
├── config.toml
└── results.jsonl
```

`config.toml` is the resolved typed configuration. `results.jsonl` receives one complete
trace per line as rollouts finish. There is no run manifest or `run.json`.

Resume an interrupted run by passing only its output directory:

```bash
prime eval run --resume outputs/my-taskset--openai--gpt-4.1-mini--default/<run-id>
```

Resume reuses the saved config and appends only missing or failed rollouts.

Browse or publish results through Prime:

```bash
prime eval view
prime eval push outputs/my-taskset--openai--gpt-4.1-mini--default/<run-id>
```

Verifiers exports typed artifact readers from `verifiers.v1.cli.output`; hosts should use
those readers instead of reimplementing the file contract.

## Hosted V0 evaluation

Hosted evaluation is a Prime-owned API surface and is intentionally separate from local
Verifiers execution:

```bash
prime eval submit owner/my-v0-env \
  --model openai/gpt-4.1-mini \
  --num-examples 20 \
  --rollouts-per-example 3 \
  --follow
```

The environment must already be published. A local id is accepted only when its
`.prime/.env-metadata.json` records a published upstream.

For batches, use Prime's strict hosted format:

```toml
# configs/eval/hosted.toml
model = "openai/gpt-4.1-mini"
num_examples = 20
rollouts_per_example = 3

[[eval]]
env_id = "owner/first-env"

[[eval]]
env_id = "owner/second-env"
num_examples = 50
```

```bash
prime eval submit configs/eval/hosted.toml
```

Shared top-level values apply to every `[[eval]]`; table values override them. Targets with
identical settings are submitted together. This hosted schema is not parsed through the V0
Verifiers CLI.
