# Experimental Environments

Newer and more experimental environment classes that may have some sharper edges + change more frequently.

## ComposableEnv — Task / Agent decomposition

ComposableEnv separates **what to solve** (the task) from **how to solve it** (the agent) by reusing the battle-tested `CliAgentEnv` and delegating task-specific behavior to a `TaskSpec`.

### Core concepts

**TaskSpec** — the shared behavior for a problem type. One per domain. Defines docker image, sandbox setup, evaluation, and validation logic.

**Task** — one problem instance. Has a prompt, metadata (`info`), and a reference to its TaskSpec.

**TaskSet** — a collection of Tasks backed by a TaskSpec and an HF Dataset. The dataset is the backbone — it's what the framework iterates over for training and evaluation. The TaskSpec gives each row its meaning (how to create the sandbox, how to score the result).

```
TaskSet = Dataset + TaskSpec
         (the data)  (the behavior)
```

**ComposableEnv** — a `CliAgentEnv` subclass that delegates its hooks to a TaskSpec. It overrides three methods (`get_docker_image`, `post_sandbox_setup`, `post_rollout`) and inherits everything else: tunnel, HTTP interception, background job polling, streaming, TITO caching.

### Usage

```python
from swe_tasksets import R2ETaskSet
from opencode_agent import build_install_script, build_opencode_run_command
from verifiers.envs.experimental.composable_env import ComposableEnv

# Create a taskset
taskset = R2ETaskSet()                    # 4578 SWE instances

# Explore instances
task = taskset[0]                         # one Task
task.prompt                               # the problem statement
task.get_image()                          # per-instance docker image

# Slice
small = taskset.take(100)                 # first 100
filtered = taskset.filter(lambda ex: ...) # custom filter

# Validate (are gold solutions correct?)
results = await taskset.take(10).validate_taskset(concurrency=5)

# Run with an agent
env = ComposableEnv(
    taskset=taskset,
    run_command=build_opencode_run_command(agent_workdir="/testbed"),
    install_script=build_install_script(),
)
```

### Writing a TaskSpec

Implement these methods to create a new task type:

```python
class MyTaskSpec:
    needs_sandbox = True

    def get_prompt(self, info: dict) -> Messages:
        """What the agent sees."""

    def get_image(self, info: dict) -> str:
        """Docker image for this instance."""

    def get_workdir(self, info: dict) -> str:
        """Working directory inside the sandbox."""

    def get_env_vars(self) -> dict[str, str]:
        """Environment variables for the sandbox."""

    async def setup(self, sandbox_client, sandbox_id, state) -> None:
        """Prepare the sandbox (install deps, write files, etc)."""

    async def evaluate(self, sandbox_client, sandbox_id, state) -> float:
        """Score the result (run tests, check answer). Returns 0.0-1.0."""

    async def validate(self, sandbox_client, sandbox_id, state) -> bool:
        """Verify this instance is solvable (apply gold solution, evaluate)."""

    def get_extra_tools(self) -> list:
        """Optional domain-specific tools (e.g. compile_proof for Lean)."""
```

Then create a TaskSet:

```python
from verifiers.envs.experimental.task import TaskSet

dataset = load_dataset("my-dataset", split="train")
taskset = TaskSet(spec=MyTaskSpec(), dataset=dataset, name="my-tasks")
```

### How ComposableEnv works

ComposableEnv subclasses `CliAgentEnv` without modifying it. It overrides three hooks:

- **`get_docker_image(state)`** — delegates to `spec.get_image(info)` for per-instance images
- **`post_sandbox_setup(state)`** — runs `spec.setup()`, uploads the task instruction, installs the agent binary
- **`post_rollout(state)`** — runs `spec.evaluate()` to compute the reward

Everything else — tunnel setup, HTTP interception, background job polling, request normalization, streaming, TITO caching — is inherited from `CliAgentEnv` unchanged.

---

## GymEnv

Universal runner for Gym-compatible environments. Wraps any environment that implements `reset(seed)` and `step(action)` methods (following the OpenAI Gym / Gymnasium API). Supports both old-style 4-tuple and new-style 5-tuple step returns.

## MCPEnv

Environment for integrating MCP (Model Context Protocol) servers as tools. Connects to one or more MCP servers via stdio transport and exposes their tools to the model. Useful for giving models access to external services like web search, file fetching, or any MCP-compatible tool server.

## CliAgentEnv

Environment for running custom agent code inside sandboxes. Intercepts the agent's OpenAI API requests via an HTTP proxy server, with each request triggering one `MultiTurnEnv` rollout step.

## HarborEnv

`CliAgentEnv` subclass that loads Harbor-format tasks. Harbor is a task format for agent benchmarks with structured task directories containing `task.toml` configuration and `instruction.md` prompts, along with test scripts for computing rewards.

## RLMEnv

Environment implementing [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) (RLMs), an inference strategy where language models can decompose and recursively interact with input context of unbounded length through REPL environments.
