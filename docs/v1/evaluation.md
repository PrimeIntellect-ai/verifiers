# Evaluation

To evaluate any taskset, use the `eval` entrypoint:

```bash
uv run eval primeintellect/terminal-bench-2
```

You can also use `.toml` files for configuration:

```toml
model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"

[sampling]
temperature = 1.0

[taskset]
id = "primeintellect/terminal-bench-2"

[harness]
id = "codex"
version = "0.116.0"
```

Validate the config by using `uv run eval @ config.toml --dry-run`. To run the evaluation, use `uv run eval @ config.toml`.

Use dotted arguments to set values using the CLI, e.g. `--sampling.temperature 0.5`. CLI arguments overwrite toml arguments when both are present.

The output from evaluations are written into `outputs/<taskset>--<model>--<harness>/<uuid>/` by default (use `output_dir` to overwrite the folder). The folder contains the used `config.toml`, all the traces in `results.jsonl`, as well as logs of the run and workers in `eval.log`.

## Common config values

- `model` — the model id to evaluate, e.g. `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`
- `sampling` — generation params passed to the model, e.g. `sampling.temperature`
- `taskset.id` / `harness.id` — pick the taskset and harness
- `num_tasks` — how many tasks to evaluate. Not setting a value means all tasks; an
  infinite taskset (a procedural generator, e.g. `wordle-v1`) requires it
- `num_rollouts` — rollouts per task
- `max_concurrent` — caps how many rollouts are in flight at once
- `verbose` — log at debug instead of info
- `shuffle` — randomizes the order of tasks (fixed seed); a no-op on an infinite taskset

## External candidate optimization (Weco)

[Weco](https://github.com/wecoai/weco-cli) optimizes a candidate artifact by rewriting it
and re-running an eval command. `weco-eval` is the verifiers-side adapter: one fixed v1
evaluation of the configured taskset + harness (it is not a harness itself) whose stdout
ends in a parseable `reward: <mean>` line. The candidate must be a *declarative* local
file the taskset or harness actually loads in each fresh evaluation process — a prompt,
template, or config, never Python the taskset imports (code candidates execute inside the
`weco-eval` process itself, which no output sealing or container around the harness can
make safe; supporting them would need a separate scoring trust domain). Use Weco's
`--sources` for a deliberately separated multi-file candidate surface — verifiers neither
receives nor manages Weco's source paths:

```bash
weco run --source <candidate-artifact> \
  --eval-command "uv run weco-eval <taskset-id> -n 20" \
  --metric reward --goal maximize \
  --steps 10 --eval-timeout 1800 --apply-change --output plain --no-open \
  --additional-instructions weco-instructions.md
```

Author `weco-instructions.md` yourself — what may change and what behavior must remain —
and pass the *path*: inline instruction text that starts with `-` breaks argument parsing,
and text that happens to equal an existing filename is read as one. Check the file exists
before launching (Weco silently treats a missing path as literal instruction text).
Instructions steer the optimizer, so they must come from you, never
from candidate or task output. `--metric` may
be the aggregate `reward` or one of the emitted `reward/<name>` / `metric/<name>`
components; the selected objective must stay fixed across the run. Any errored rollout
fails the step — `weco-eval` exits non-zero without metric lines rather than scoring a
partial eval (configure `--retries.rollout.*` to absorb transient provider errors). The
flags after `--goal` keep headless runs bounded and noninteractive: an explicit
`--steps` budget (Weco defaults
to 100 steps — thousands of rollouts at `-n 20`), an `--eval-timeout` so a pathological
candidate can't hang the run forever, `--apply-change` to write the winner back without
the interactive confirmation `weco run` otherwise ends with, plain output, and no browser
tab.

### System-prompt convenience

When the candidate is the taskset's system prompt, `--system-prompt-path` overrides every
selected task's prompt with the file contents, and `--init-prompt` seeds the file from the
taskset's own baseline. Seeding scans the same task selection the eval scores
(`-n`/`--shuffle`; an infinite taskset requires `-n`) and is refused when those tasks carry
differing system prompts:

```bash
uv run weco-eval <taskset-id> \
  --system-prompt-path prompt.txt --init-prompt -n 20

weco run --source prompt.txt \
  --eval-command "uv run weco-eval <taskset-id> --system-prompt-path prompt.txt -n 20" \
  --metric reward --goal maximize \
  --steps 10 --eval-timeout 1800 --apply-change --output plain --no-open \
  --additional-instructions weco-instructions.md
```

Describing the task and its required output format in your instructions file keeps the
task's intent visible to the optimizer even after it has rewritten `prompt.txt` (Weco
already sees the current prompt as the source file). Each run snapshots the exact
evaluated prompt to `<run-dir>/system_prompt.txt` and points its saved `config.toml` at
that snapshot, so a run replays against the candidate it actually scored even after Weco
restores `prompt.txt`; an explicit `-o` gains a per-run leaf so successive candidates
never overwrite each other.

### Benchmark integrity

Weco may edit only the intended candidate surface. Keep outside `--source`/`--sources`: the
eval command and its config, `@reward`/`@metric` implementations, reference answers and
dataset selection, tests/validators/correctness gates, and held-out tasks or data. Keep
fixed across candidates: taskset configuration and seed, the selected tasks, harness, model
and sampling, rollout count, and the scoring implementation. Evaluate the winner on a
disjoint held-out split or selection — a higher optimization reward on the selected tasks is
not by itself an improvement, and candidate artifacts that define or generate tasks must
be optimized against a frozen task-quality metric with fixed validity gates, or Weco can raise reward by weakening
the task.

Keep held-out answers and scoring assets *inaccessible* to the candidate's runtime, not
merely unwritable — a tool-capable harness will read whatever files a candidate prompt asks
it to. Note that `--apply-change` trusts the file set Weco's service returns (its source
allowlist is not yet enforced client-side), and that seeded prompts containing Markdown
backticks can trip current weco-cli handling of uploaded content. Expect data egress: `weco run`
uploads the source contents, additional instructions, the evaluation command, each step's
evaluation output, and any `--api-key` provider keys you pass to the Weco service. The
`--source` list is not a security boundary on its own, so stick to declarative candidates;
if you ever experiment beyond them, run the entire Weco + evaluator stack inside an
isolated container or VM with scoped credentials and controlled data/network access — a
merely "disposable" directory still inherits your environment variables, credentials, and
filesystem, and in-process candidate code could tamper with scorers or data in memory
regardless of sandboxing.

## Resuming evaluations

`--resume <output-dir>` re-runs only the rollouts a previous run left missing or errored, appending to that run's own `results.jsonl`. It reloads the run's saved `config.toml` verbatim, so it takes no other arguments. Good rollouts are kept, while errored ones are dropped and redone.

## Disabling tools

Almost every harness comes with a `disabled_tools` list, which can be used to disable one or multiple tools:

```toml
[harness]
disabled_tools = ["shell_tool"]
```

The names of these tools are set by the respective harness. Consult the relevant documentation for the given harness for the relevant name(s). Some harnesses do not offer support to disable tools.
