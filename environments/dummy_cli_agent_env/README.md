# dummy-cli-agent-env

### Overview
- **Environment ID**: `dummy-cli-agent-env`
- **Short description**: Test environment that validates CliAgentEnv data flow by running a simple Python agent script in a sandbox
- **Tags**: `test`, `cli-agent`, `eval`

### Datasets
- **Primary dataset**: Synthetic dataset with 5 examples
- **Source**: Generated in `load_environment()`
- **Split sizes**: 5 examples

### Task
- **Type**: multi-turn
- **Parser**: Default parser
- **Rubric overview**: 
  - `completion_reward`: Returns 1.0 if `/tmp/vf_complete` marker file exists, 0.0 otherwise

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval dummy-cli-agent-env
```

Configure model and sampling:

```bash
uv run vf-eval dummy-cli-agent-env -m gpt-4.1-mini -n 5 -r 1
```

### How It Works

This environment validates the CliAgentEnv data flow:

1. **Agent Script**: A Python script (`_AGENT_SCRIPT`) is embedded as base64 and written to the sandbox
2. **Interception**: The script makes OpenAI API calls which are intercepted by CliAgentEnv's HTTP proxy server
3. **Data Flow**: Each intercepted API call triggers one rollout iteration:
   - Agent makes HTTP request → intercepted by proxy server
   - Request queued → `get_prompt_messages()` processes it
   - Response generated → cached in intercept
   - `get_model_response()` returns cached response
   - Trajectory step added
4. **Completion**: Agent writes `/tmp/vf_complete` when done
5. **Stop Condition**: `agent_signaled_completion` detects the marker file and stops rollout

### Agent Script Details

The agent script:
- Reads `OPENAI_BASE_URL` and `OPENAI_MODEL` from environment variables
- Makes 4 OpenAI API calls in a loop
- Prints hardcoded "env response" messages between calls
- Writes `/tmp/vf_complete` when complete

### Environment Arguments

No custom arguments supported. Uses default CliAgentEnv configuration:
- `docker_image`: `python:3.11-slim`
- `max_turns`: 10
- `timeout_seconds`: 300.0
- `request_timeout`: 60.0

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if completion marker exists, 0.0 otherwise |
