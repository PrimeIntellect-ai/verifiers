# Eval process protocol

Verifiers owns eval argument parsing, dynamic taskset/harness resolution, execution, and run
artifacts. Process consumers should execute the Verifiers interpreter installed in the target
workspace instead of importing CLI internals:

```bash
python -m verifiers.v1.cli.eval.main --protocol-version
python -m verifiers.v1.cli.eval.main resolve --format json <eval-argv...>
python -m verifiers.v1.cli.eval.main run <eval-argv...>
```

The installed `eval` console script accepts the same arguments. Protocol stdout is JSON; normal
run output and direct `run ... --help` remain human-oriented.

## Capabilities

`--protocol-version` takes no other arguments and returns:

```json
{
  "protocol_version": 1,
  "trace_schema_version": 1,
  "manifest_schema_version": 1,
  "operations": ["run", "resolve"]
}
```

## Resolution

`resolve --format json` uses the same parser and dynamic config narrowing as `run`, but does not
execute an eval or write artifacts. Its response includes all config defaults:

```json
{
  "operation": "resolve",
  "protocol_version": 1,
  "trace_schema_version": 1,
  "manifest_schema_version": 1,
  "run_id": "3b246a70-2f0c-4b03-a3fe-b9a16be0e194",
  "output_dir": "outputs/my-task--provider--model--default/3b246a70-2f0c-4b03-a3fe-b9a16be0e194",
  "resume": false,
  "config": {}
}
```

Relative output paths are relative to the child process working directory. A resume resolution
loads the existing run ID and output directory from the run artifact. The public Python equivalent
is `verifiers.v1.cli.eval.resolve_eval(argv)`; it passes explicit `args` and `prog` to
`prime-pydantic-config` and does not read or mutate `sys.argv`.

## Run artifact

Every run, including `--dry-run`, owns one directory with:

```text
manifest.json
config.toml
results.jsonl
eval.log
```

`manifest.json` is replaced atomically at each lifecycle transition:

```json
{
  "schema": "verifiers.eval-run/v1",
  "protocol_version": 1,
  "trace_schema_version": 1,
  "run_id": "3b246a70-2f0c-4b03-a3fe-b9a16be0e194",
  "status": "running",
  "attempt": 1,
  "created_at": "2026-06-24T12:00:00Z",
  "updated_at": "2026-06-24T12:00:00Z",
  "started_at": "2026-06-24T12:00:00Z",
  "finished_at": null,
  "artifacts": {
    "config": "config.toml",
    "results": "results.jsonl",
    "log": "eval.log"
  },
  "error": null
}
```

`status` is one of `running`, `completed`, `failed`, or `cancelled`. A failed run records
`{"type": "ExceptionClass", "message": "..."}` in `error`. Resume preserves `run_id` and
`created_at`, increments `attempt`, clears the previous terminal fields, and transitions through
`running` to a new terminal state in the same directory. Runs created before `manifest.json` was
introduced remain resumable: their first resume derives the run ID from the directory name and
creates the manifest at attempt 2 without rewriting `config.toml` or `results.jsonl`.
