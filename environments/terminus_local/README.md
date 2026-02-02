# terminus_local

Example environment using `LocalHarborEnv` to run Terminus agent on Harbor tasks.

## Overview

This environment demonstrates the simplified Harbor integration pattern that:

1. Uses Harbor SDK's `Trial` class directly for agent execution and verification
2. Passes a tunnel URL as `api_base` to intercept agent API calls
3. Collects trajectory from intercepted requests/responses
4. Extracts reward from Harbor's `TrialResult.verifier_result`

## Usage

### Basic Evaluation

```bash
prime eval run terminus_local
```

### Programmatic Usage

```python
from verifiers.environments.terminus_local import load_environment

env = load_environment(
    dataset_path="./tasks",
    agent_name="terminus-2",
    model_name="anthropic/claude-sonnet-4",
    max_turns=50,
)

# Run evaluation
results = await env.evaluate(
    client=client,
    model="anthropic/claude-sonnet-4",
    num_examples=5,
)
```

## LocalHarborEnv vs CliAgentEnv + HarborEnv

| Aspect             | CliAgentEnv + HarborEnv                              | LocalHarborEnv                                  |
| ------------------ | ---------------------------------------------------- | ----------------------------------------------- |
| Sandbox management | Manual (CliAgentEnv creates sandbox, installs agent) | Harbor SDK handles it via `Trial.run()`         |
| Task loading       | HarborEnv uploads files to sandbox                   | Harbor SDK reads task directory directly        |
| Agent execution    | Custom script in sandbox                             | Harbor's built-in agent runners                 |
| Verification       | Custom verifier integration                          | Harbor's standard `TrialResult.verifier_result` |
| API interception   | Same (Prime Tunnel + HTTP server)                    | Same (Prime Tunnel + HTTP server)               |

**When to use LocalHarborEnv:**

- Running Harbor-format tasks with standard Harbor agents (terminus-2, etc.)
- Want simpler setup with less custom code

**When to use CliAgentEnv + HarborEnv:**

- Need custom sandbox setup or agent installation
- Running non-Harbor agents on Harbor tasks

## Task Format

Tasks follow the standard Harbor format:

```
tasks/
└── my-task/
    ├── task.toml           # Task configuration
    ├── instruction.md      # Task instruction for the agent
    ├── environment/        # Environment setup (Dockerfile, etc.)
    │   └── Dockerfile
    ├── tests/              # Verification scripts
    │   └── test.sh         # Must write reward to /logs/verifier/reward.txt
    └── solution/           # Optional reference solution
        └── ...
```
