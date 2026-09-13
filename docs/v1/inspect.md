# Inspect tasks

The built-in `inspect` taskset runs the portable subset of an Inspect task through
Verifiers v1. Inspect supplies a finite dataset and a supported deterministic scorer;
Verifiers owns the harness, runtime, model calls, trace, and scalar reward.

Install Inspect and the package that owns the task for the loader and scoring workers.
Keep these dependencies outside the root Verifiers project:

```bash
uv run \
  --with 'inspect-ai>=0.3.263,<0.4' \
  --with 'task-provider-package==1.2.3' \
  eval @ inspect-task.toml --dry-run
```

An example config is:

```toml
model = "openai/gpt-5"

[env.taskset]
id = "inspect"
source = "some_package/some_task"
source_args = { split = "test" }
solver_policy = "replace"

[env.agent.harness]
id = "bash"
```

After validating the TOML, remove `--dry-run` to evaluate it. Local Inspect task files
use their normal `file.py@task` reference as `source`. One source must resolve to exactly
one Inspect task.

`solver_policy = "replace"` is deliberate: the selected Verifiers harness replaces the
Inspect task's main solver. The source solver name and replacement policy are recorded on
every task trace, so this run must not be presented as official runner parity.

## Supported subset

- Finite Inspect `Dataset` / `Sequence[Sample]` task data.
- String input or chat history containing text and HTTP(S)/data-URL images.
- Sample ids, targets, choices, and JSON-serializable metadata.
- A bare Inspect `generate()` solver, which the selected v1 harness replaces.
- Exactly one output-only built-in scorer: `answer`, `exact`, `f1` (without a custom
  `answer_fn`), `includes`, `match`, or `pattern`.
- Exact sample-id selection with `sample_ids`; use normal Verifiers `-n` for truncation.
- Installed `package/task` and local `file.py@task` references.

Scoring calls Inspect's tested built-in scorer implementation against the final raw
Verifiers completion, then records the scalar as `inspect/<scorer>`. The complete Inspect
score value, answer, explanation, reason, and metadata are retained under
`trace.info.inspect.score`.

Task data records the adapter version, Inspect/task package provenance, task version and
arguments, sample id, source solver, and replacement policy. Absolute local task-file
paths are reduced to `filename.py@task` in traces; the resolved run config still retains
the user-supplied source needed to reproduce the run.

## Needs a native task-specific port

The generic taskset fails before inference for behavior it cannot preserve:

- Inspect `Task.setup`, cleanup callbacks, or dynamic `SampleSource` data.
- Task/sample sandboxes, sample files/setup scripts, and checkpoint callbacks.
- Custom, model-graded, sandbox/state-dependent, perplexity, `choice`, or multiple scorers.
- Custom or composed solvers, including solver-owned prompt formatting, tools, user
  simulation, agents, or multi-agent branches.
- Audio, video, documents, provider-internal content, and annotated image detail.

For these tasks, define a normal v1 `TaskData`/`Task`/`Taskset` package and port the
dataset, prompt setup, runtime, tools, and scoring semantics explicitly. The original
Inspect runner remains the parity oracle.

Inspect does not currently document a standalone task resolver, so the adapter's use of
its private loader is isolated in `verifiers.v1.tasksets.inspect.compat` and version-gated.
Registered task packages are never downloaded automatically; install and pin trusted
packages yourself.
