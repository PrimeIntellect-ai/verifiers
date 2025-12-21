# dummy-harbor-env

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/dummy_harbor_env">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview

- **Environment ID**: `dummy-harbor-env`
- **Short description**: Minimal Harbor environment for testing the CLI agent interception framework
- **Tags**: `dummy`, `testing`, `cli-agent`, `harbor`

### Datasets

- **Primary dataset**: Harbor-format tasks in `tasks/` directory
- **Source**: Bundled with environment
- **Tasks**: 1 dummy task (`hello-world`)

### Task

- **Type**: single-turn (via HarborEnv)
- **Base class**: `HarborEnv` (extends `CliAgentEnv`)
- **Rubric overview**:
  - Reward computed by `tests/test.sh` which runs pytest on `test_state.py`
  - Returns 1.0 if `/app/hello.txt` contains "Hello, world!", 0.0 otherwise

### Quickstart

Run an evaluation with default settings:

```bash
uv run vf-eval dummy-harbor-env
```

Configure model and sampling:

```bash
uv run vf-eval dummy-harbor-env -m gpt-4.1-mini -n 1 -r 1
```

### How It Works

This environment demonstrates the HarborEnv/CliAgentEnv data flow:

1. **Harbor Task Loading**: Task is loaded from `tasks/hello-world/` with `task.toml`, `instruction.md`, and `tests/`
2. **Sandbox Creation**: A Docker sandbox is created with the task instruction uploaded to `/task/`
3. **Agent Execution**: A Python script reads the instruction and makes an OpenAI API call
4. **Interception**: The API call is intercepted by CliAgentEnv's HTTP proxy server (via Cloudflare tunnel)
5. **LLM Response**: The LLM returns a bash command to complete the task
6. **Execution**: The agent executes the command in `/app`
7. **Testing**: Harbor's `tests/test.sh` runs pytest to verify the result

### Agent Script Details

The embedded agent script:

- Reads task instruction from `/task/instruction.md`
- Asks the LLM for a bash command to complete the task
- Executes the returned command in `/app`

For the `hello-world` task, the LLM should respond with something like:
```bash
echo "Hello, world!" > hello.txt
```

### Environment Arguments

| Argument          | Type                | Default            | Description                              |
| ----------------- | ------------------- | ------------------ | ---------------------------------------- |
| `dataset_path`    | `str \| Path`       | `./tasks`          | Path to Harbor-format tasks directory    |
| `tasks`           | `list[str] \| None` | `None`             | Specific task names to load (None = all) |
| `agent_workdir`   | `str`               | `/app`             | Working directory for agent in sandbox   |
| `docker_image`    | `str`               | `python:3.11-slim` | Docker image for sandbox                 |
| `timeout_seconds` | `float`             | `300.0`            | Overall rollout timeout                  |
| `max_turns`       | `int`               | `-1`               | Max turns (-1 = unlimited)               |

### Metrics

| Metric   | Meaning                                              |
| -------- | ---------------------------------------------------- |
| `reward` | 1.0 if pytest passes (hello.txt correct), 0.0 otherwise |
