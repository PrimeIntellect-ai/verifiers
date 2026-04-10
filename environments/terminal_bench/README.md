# Terminal-Bench Sandbox

This example wraps a [Terminal-Bench](https://github.com/laude-institute/terminal-bench) task inside a `SandboxEnv`. The loader
dynamically downloads the requested Terminal-Bench task from the official registry, prepares the docker image specified by the
task, and exposes a persistent bash tool so an agent can solve the task interactively. After the rollout finishes the official
Terminal-Bench test runner (`run-tests.sh`) is executed inside the sandbox and the cached exit status is used as the reward.

- **Default task**: `hello-world` – the canonical introductory Terminal-Bench task that is referenced throughout their docs.
- **Dataset size**: 1 example (just the selected task).
- **Reward**: 1.0 if the Terminal-Bench tests pass, 0.0 otherwise. The raw stdout/stderr from the test run is attached to the
  rollout state for inspection.

## Usage

```python
import verifiers as vf

env = vf.load_environment(
    "terminal-bench-sandbox",
    task_id="hello-world",  # any Terminal-Bench task id available in the registry
)
```

The environment automatically downloads the chosen task (via `terminal-bench`'s registry client), uploads it to a sandbox using
the specified docker image, and exposes a `bash` tool. Agents can run `bash` commands such as:

```text
bash: cd /workspace/task && ls
bash: cd /workspace/task && TEST_DIR=/tests bash ./run-tests.sh
```

When the rollout ends we execute `run-tests.sh` ourselves with the proper environment variables (`TEST_DIR`/`T_BENCH_TEST_DIR`)
and store the reward for the rubric.

## Configuration

`load_environment` accepts the following keyword arguments:

- `task_id` (str): Terminal-Bench task identifier (default: `hello-world`).
- `dataset_name` / `dataset_version`: dataset coordinates in the Terminal-Bench registry (defaults to `terminal-bench-core`
  version `head`).
- `registry_url` or `local_registry_path`: override the registry source.
- `max_turns`: maximum agent turns before the environment stops the rollout (default: 20).

The sandbox is configured with the docker image declared in the task's `Dockerfile` and keeps the task assets under
`/workspace/task` inside the container. Tests are mirrored to `/tests` for compatibility with the upstream harness.
