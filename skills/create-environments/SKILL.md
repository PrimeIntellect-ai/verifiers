---
name: create-environments
description: Create or migrate native verifiers.v1 taskset and harness packages. Use to build an environment, port a benchmark, add task tools or user simulation, package an agent harness, or migrate an existing v0 environment to the typed v1 trace model.
---

# Create Environments

## Goal

Create native V1 environments that are installable and runnable with verifiers.

To start, ALWAYS use the CLI to create a package with the correct files:

```bash
prime env init my-task-v1
```

Add only the components the contract needs:

```bash
prime env init my-task-v1 -T      # task toolset
prime env init my-task-v1 -U      # user simulator
prime env init my-agent-v1 -H     # custom reusable harness
```

Often, the user does not want nor need a custom reusable harness, as verifiers offer a lot of built-in ones.

## Re-use existing abstractions first

For some common tasks, there are existing, pre-built tasksets in the `verifiers.v1.tasksets` folder. These come with batteries included and should always be preferred. The most notable inclusion is the `HarborTaskset`, which allows the creation of Harbor-based tasksets within a few LoC (also see docs/harbor.md).

## Custom task images

When a task needs a custom container image (e.g. a Harbor task whose `task.toml` does not have `docker_image`), you can build and publish it with `prime images push` from the Prime CLI ([Documentation](https://docs.primeintellect.ai/sandboxes/images)). This builds in the cloud — no local Docker needed — and prints the full image reference to use as the task's `image` field.

Use the naming convention `<env>.x86.<task>:latest` for the image name (e.g. `abc.x86.xyz:latest`), where `<env>` is the taskset name and `<task>` is the individual task.

## Define the needed values first

Before starting with the implementation, think about the following things:
- What is the dataset about, which fields does it have?
- Does it come with custom tools that are strictly necessary and not added by common harnesses? For example, a lot of harnesses come with bash or web search tools, which makes custom tools obsolete. Always prefer harnesses over custom tools
- Does the taskset need a user simulator?
- Which rewards are needed for scoring? What additional metrics might be nice to have, either for debugging, training or potentially in the future?
- How should the tasks be scored, is a judge needed?

For a port, map source behavior one-to-one: rows, the exact prompts verbatim, harness restrictions, score extraction and exceptions.

Ask the user about unresolved semantic choices instead of inventing them. Present your evidence (both in code and in your questions) by commenting and linking to the exact source in the paper, the GitHub repo etc.

## Native package contract

A package exports one `vf.Taskset` subclass and optionally one `vf.Harness` subclass through `__all__`. This happens automatically when you bootstrap a new environment using `prime env init`.

Do not add `load_environment()`, `load_taskset()`, or `load_harness()` functions. The v1 loader
resolves classes and their config types from `__all__` and generic bases.

Use:

```python
import verifiers.v1 as vf
```

Never mix v0 `Environment`, `Rubric`, `Parser`, `SingleTurnEnv`, `MultiTurnEnv`, or `ToolEnv` objects into a v1 environment. Exclusively use functions, classes and objects from `verifiers.v1`.

## Minimal implementation

```python
import verifiers.v1 as vf 


# A task defines a single problem and is defined as a subclass of vf.Task
class AdditionTask(vf.Task):
    answer: int


# The taskset defines the dataset (collection of tasks) and needs a vf.TasksetConfig, which can be empty.
class AdditionTaskset(vf.Taskset[AdditionTask, vf.TasksetConfig]):
    def load_tasks(self) -> list[AdditionTask]:
        # Load the dataset, in this case we built it on the initial load
        return [
            AdditionTask(idx=i, prompt=f"What is {i} + {i}?", answer=2 * i)
            for i in range(100)
        ]

    # @vf.reward denotes the scoring function for the environment.
    # It needs the actual task (defined earlier) as well as the trace, which contains the whole message graph, including function calls, user messages etc.
    # It returns the reward for the rollout based on this function.
    @vf.reward
    async def exact_match(self, task: AdditionTask, trace: vf.Trace) -> float:
        return float(trace.last_reply == str(task.answer))


# Export the Taskset for verifiers to find it when loading
__all__ = ["AdditionTaskset"]
```

Do not override `Taskset.__init__`; use config, `load_tasks()`, lifecycle hooks, or an owning server.

## Ownership rules

The taskset owns:

- task loading and prompt framing;
- task-specific tools and user simulation;
- typed transient state;
- setup/finalize/validate hooks;
- stop conditions, metrics, rewards, and group rewards.

The harness owns:

- the reusable agent/chat program;
- how the program is provisioned and launched;
- wiring the program to the supplied interception endpoint;
- harness-generic execution metrics.

The runtime config owns where code executes. The task should not interfere with the runtime nor manipulate it directly.

## Scoring rules

- Prefer deterministic verification grounded in the task's actual artifact or answer.
- Use an LLM judge only when semantic judgment is unavoidable.
- Metrics are for observability and do not contribute to reward, but are useful. Use them deliberately and appropriately!
- Group rewards receive all traces for one task but no live runtime.
- Raise ordinary Python exceptions for invalid verifier execution. The framework records a `TasksetError` in this case.

## Validation and lifecycle

Implement `validate(task, runtime)` whenever ground truth can be checked without a model. Keep rollout work in the owning stage:

- `setup(task, runtime)` — prepare repo, files, or service.
- harness execution — let the agent act.
- `finalize(task, trace, runtime)` — capture diffs/artifacts needed for scoring.
- `@vf.reward` / `@vf.metric` — evaluate while the runtime is still live.

Persist inspectable artifacts in JSON-serializable `trace.info`. Put counters and live coordination in a typed `vf.State` subclass.

## Tools

Some environments require custom tools. These should be the exception as they don’t work with every harness and are registered as MCP servers.

You can create them like this (remember the bootstrapping with `prime env init MY_ENV -T`)
```python
class SearchToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "search"

    @vf.tool
    async def query(self, text: str) -> list[str]:
        """Search the task corpus."""
        return []


if __name__ == "__main__":
    SearchToolset.run()
```

Expose the tool to the config:

```python
class SearchConfig(vf.TasksetConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(shared=True)
```

`Taskset.tools()` returns `[]` by default, so the toolset is never launched unless you wire it. Return it from the hook:

```python
class SearchTaskset(vf.Taskset[vf.Task, SearchConfig]):
    def tools(self, task: vf.Task) -> list[vf.Toolset]:
        return [SearchToolset(self.config.tools)]
```

Tools can be placed in various ways. Think about what the tool is and how expensive its setup is:
- per-rollout for cheap setup MCPs;
- colocated for shared harness filesystem;
- shared for expensive read-only setup or writable state entirely in `self.state` (e.g. big databases);
- remote URL for an existing MCP service.

## User simulation

Use a `vf.User` when the environment, not the harness, is able to drive the conversation.

The selected harness must support user simulation, which a lot of the built-in, especially the CLI-based ones, don't. The built-in default harness does support user sim.

## Custom harnesses

Choose a built-in first (which you can find under `verifiers.v1.harnesses`). Add a custom harness only when desired or not currently implemented.

Its `launch()` **must** point every model request at the provided `endpoint` with `secret` (to work with the InterceptionServer); direct provider calls bypass trace capture and thus would break downstream usage.

Advertise capabilities accurately:

- `SUPPORTS_MCP`
- `SUPPORTS_USER_SIM`
- `SUPPORTS_MESSAGE_PROMPT`
- `APPENDS_SYSTEM_PROMPT`

Return the `vf.ProgramResult` from `runtime.run_program()` or `runtime.run_uv_script()`. Do not manually build trace nodes in the harness.

## Dependencies and credentials

- Add package import-time dependencies to the environment's own `pyproject.toml`.
- Never edit the repository root `pyproject.toml` or `uv.lock` for an environment dependency.
- Read required external credentials at the earliest owning boundary so missing values fail clearly.
- Do not require a user-managed background server unless the contract explicitly uses a remote URL.

## Migration from v0

Map concepts directly:

| V0 | Native v1 |
| --- | --- |
| Dataset row | Typed `vf.Task` subclass |
| `load_environment(**kwargs)` | Exported `vf.Taskset` class + typed config |
| `Rubric` reward function | Taskset `@vf.reward` method |
| Parser object | Ordinary parsing inside scoring |
| `ToolEnv` tools | `vf.Toolset` + `Taskset.tools()` |
| `MultiTurnEnv.env_response` | `vf.User`, tools, or a reusable harness |
| Dict state | Typed `vf.State` |
| Sandbox subclass | Harness runtime config + task hooks |

Preserve prompt, tool, and scoring equivalence before improving design. Compare representative v0 and v1 traces where feasible.

## Publish gate

After package installability, validation, and representative eval behavior are stable, ask the user whether Hub visibility should be `PUBLIC` or `PRIVATE`. Only then run:

```bash
prime env push my-task-v1 --visibility PRIVATE
```

Publishing is an external state change and requires the user's requested visibility. Do not publish merely because local verification passed.
