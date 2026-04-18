# Composable Task / Agent Architecture

Separates **what to solve** (the task) from **how to solve it** (the agent) by reusing the battle-tested `CliAgentEnv` and delegating task-specific behavior to a `TaskSet`.

## Core concepts

**Task** — a single, fully-bound problem instance. Has a prompt, sandbox spec, metadata. Created via `TaskSet[i]`.

**TaskSet** — a collection of Tasks backed by an HF Dataset. Subclass this for pure-LLM tasks (math, QA) where no sandbox is needed.

**SandboxTaskSet** — a TaskSet that requires sandboxes. Subclass this for SWE, Lean, Harbor, or anything needing Docker containers.

**SandboxSpec** — describes sandbox requirements per instance (image, CPU, memory, GPU type, timeout).

**Harness** — agent-side configuration. Declares how to install and run an agent binary, and where it expects to find task-provided content (instruction, system prompt).

**ComposableEnv** — a `CliAgentEnv` subclass that wires a TaskSet + Harness. Inherits all interception machinery unchanged. Supports `install_env` for install-only environment variables, automatic upload of task-declared directories (via `TaskSet.get_upload_dirs()`), and harness-declared metrics collection (via `Harness.metrics_path`).

**Skills** are first-class: any taskset with a sibling `skills/` directory gets automatic upload for free. `TaskSet.get_skills_dir()` auto-discovers it, and `get_upload_dirs()` includes it under the `"skills"` key by default. The harness's `upload_dir_mapping` decides where skills land in the sandbox (e.g. RLM puts them at `/task/rlm-skills`).

**discover_sibling_dir(taskset_cls, dirname)** — utility to auto-discover a directory co-located with a TaskSet's module. Works with installed packages and filesystem paths.

## Usage

```python
from verifiers.envs.experimental.composable.tasksets.swe.r2e_gym import R2EGymTaskSet
from verifiers.envs.experimental.composable.harnesses.opencode import opencode_harness
from verifiers.envs.experimental.composable import ComposableEnv

# Create a taskset
taskset = R2EGymTaskSet()                 # 4578 SWE instances

# Explore instances
task = taskset[0]                          # one Task
task.prompt                                # the problem statement
task.sandbox_spec                          # SandboxSpec(image=..., cpu=4, ...)
task.sandbox_spec.image                    # per-instance docker image

# Slice and filter
small = taskset.take(100)
filtered = taskset.filter(lambda ex: ...)

# Validate gold solutions
results = await taskset.take(10).validate(concurrency=5)

# Run with an agent
harness = opencode_harness(system_prompt="You are a coding agent...")
env = ComposableEnv(taskset=taskset, harness=harness, keep_sandbox_for_scoring=True)
```

For RLM-backed agents, use `ComposableEnv` with `rlm_harness(...)`. If your taskset has a sibling `skills/` directory, it's uploaded automatically — no override needed:

```python
# my_taskset/
# ├── __init__.py
# ├── taskset.py     ← defines MySweTaskSet
# └── skills/
#     └── demo/
#         ├── SKILL.md
#         └── pyproject.toml

env = ComposableEnv(
    taskset=taskset,
    harness=rlm_harness(...),  # maps "skills" → "/task/rlm-skills"
    install_env={"GH_TOKEN": token},
)
```

## Writing a new TaskSet

For **sandbox-free tasks** (math, QA), subclass `TaskSet`:

```python
class MyMathTaskSet(TaskSet):
    def get_instruction(self, info: dict) -> str:
        return info["question"]

    def get_rubric(self) -> vf.Rubric:
        return MyMathRubric()
```

For **sandbox tasks** (SWE, Lean, Harbor), subclass `SandboxTaskSet`:

```python
class MySWETaskSet(SandboxTaskSet):
    def get_instruction(self, info: dict) -> str:
        return info["problem_statement"]

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return SandboxSpec(image=info["docker_image"], cpu_cores=4)

    def get_rubric(self) -> vf.Rubric:
        return MySWERubric(self)

    async def setup(self, state) -> None:
        # state["sandbox_client"], state["sandbox_id"] available
        ...

    async def validate_instance(self, state) -> bool:
        # Apply gold patch, run tests, return True if passes
        ...
```

## Scoring

Each taskset provides its own rubric via `get_rubric()`. The rubric owns **all
scoring logic** — running tests, reading files, computing rewards. It is NOT
just a state reader.

Use `keep_sandbox_for_scoring=True` on the environment so the sandbox stays
alive after the rollout for the rubric to use.

```
Rollout completes → sandbox kept alive → rubric.score_rollout() runs tests
    → rubric.cleanup() deletes sandbox
```

This enables **retrying scoring independently of the rollout**: if scoring
fails (transient sandbox error), the rubric can retry without re-running the
agent.

### Writing a rubric

```python
class MySWERubric(vf.Rubric):
    def __init__(self, taskset, **kwargs):
        super().__init__(**kwargs)
        self.taskset = taskset
        self.add_reward_func(self.solved)

    async def solved(self, state, info, **kwargs) -> float:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        # Run tests in the sandbox
        test_output = await self.taskset._run_tests(sandbox_client, sandbox_id, state, ...)
        return float(self.taskset._calculate_reward(test_output, info))

    @vf.cleanup
    async def cleanup_sandbox(self, state):
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            await sandbox_client.delete(sandbox_id)
```

## State context

All `SandboxTaskSet` methods (`setup`, `validate_instance`) and rubric reward
functions receive `state` which contains:

- `state["sandbox_client"]` — the async sandbox client
- `state["sandbox_id"]` — current sandbox ID
- `state["test_timeout"]` — evaluation timeout (default 900)
- `state["info"]` — task metadata from the dataset row
- `state["answer"]` — ground truth answer

## How ComposableEnv works

ComposableEnv subclasses `CliAgentEnv` without modifying it. It overrides these hooks:

- **`get_docker_image(state)`** — reads `SandboxSpec.image` from the taskset
- **`get_sandbox_resources(state)`** — reads CPU, memory, GPU from `SandboxSpec`
- **`build_env_vars(state)`** — merges task env vars and exports `AGENT_WORKDIR`
- **`post_sandbox_setup(state)`** — runs task setup, uploads instruction + system prompt, installs agent
- **`post_rollout(state)`** — collects agent logs (scoring is done by the rubric)

Everything else — tunnel, HTTP interception, background job polling, streaming, TITO caching — is inherited from `CliAgentEnv` unchanged.

## Limitations and future work

**Task → Harness tool channel.** There's currently no mechanism for tasks to inject domain-specific tools into binary agents like OpenCode. Tool configuration is harness-side only (`opencode_harness(disabled_tools=...)`). A future channel between task and harness would allow tasks to declare tools that the harness can expose to the agent.

**SandboxSpec overrides.** The `SandboxSpec` comes from the TaskSet, but users may want to override resources at runtime (e.g. more memory for a specific run). Currently this is done via `ComposableEnv` kwargs passed through to `CliAgentEnv`. A cleaner pattern would be `SandboxSpec` merging at the environment level.
