---
name: create-environments
description: Create or migrate native verifiers.v1 taskset, environment, and harness packages. Use to build a taskset, port a benchmark, add task tools, script or model a user, build a multi-agent environment, package an agent harness, or migrate an existing v0 environment to the typed v1 trace model.
---

# Create Tasksets

## Goal

Create native v1 tasksets that are installable and runnable with verifiers.

To start, ALWAYS use the CLI to create a package with the correct files:

```bash
prime env init my-task-v1
```

Add only the components the contract needs:

```bash
prime env init my-task-v1 -T      # task toolset
prime env init my-agent-v1 -H     # custom reusable harness
```

Often, the user does not want nor need a custom reusable harness, as verifiers offer a lot of built-in ones.

## Re-use existing abstractions first

For some common tasks, there are existing, pre-built tasksets in the `verifiers.v1.tasksets` folder. These come with batteries included and should always be preferred. The most notable inclusion is the `HarborTaskset`, which allows the creation of Harbor-based tasksets within a few LoC (also see docs/v1/harbor.md).

## Custom task images

When a task needs a custom container image (e.g. a Harbor task whose `task.toml` does not have `docker_image`), you can build and publish it with `prime images push` from the Prime CLI ([Documentation](https://docs.primeintellect.ai/sandboxes/images)). This builds in the cloud — no local Docker needed — and prints the full image reference to use as the task's `image` field.

Use the naming convention `<env>.x86.<task>:latest` for the image name (e.g. `abc.x86.xyz:latest`), where `<env>` is the taskset name and `<task>` is the individual task.

## Define the needed values first

Before starting with the implementation, think about the following things:
- What is the dataset about, which fields does it have?
- Does it come with custom tools that are strictly necessary and not added by common harnesses? For example, a lot of harnesses come with bash or web search tools, which makes custom tools obsolete. Always prefer harnesses over custom tools
- Is the conversation driven by a user (scripted turns, a game engine, a modeled user)? That is env control flow (`user=` / `agent.chat()`), not a server.
- Does one rollout involve more than one agent run (attempts, judge, seats)? Then the package also exports an `Environment` subclass — or an existing bundled env (`--env.id best-of-n|judge|user-sim`) already covers it.
- Which rewards are needed for scoring? What additional metrics might be nice to have, either for debugging, training or potentially in the future?
- How should the tasks be scored, is a judge needed?

For a port, map source behavior one-to-one: rows, the exact prompts verbatim, harness restrictions, score extraction and exceptions.

Ask the user about unresolved semantic choices instead of inventing them. Present your evidence (both in code and in your questions) by commenting and linking to the exact source in the paper, the GitHub repo etc.

## Native package contract

A package exports one `vf.Taskset` subclass — and optionally one `vf.Environment` subclass (multi-agent control flow) and/or one `vf.Harness` subclass — through `__all__`. The taskset export happens automatically when you bootstrap a new taskset using `prime env init`.

Do not add `load_environment()`, `load_taskset()`, or `load_harness()` functions. The v1 loader
resolves classes and their config types from `__all__` and generic bases.

Use:

```python
import verifiers.v1 as vf
```

Never mix v0 `Environment`, `Rubric`, `Parser`, `SingleTurnEnv`, `MultiTurnEnv`, or `ToolEnv` objects into a v1 taskset. Exclusively use functions, classes and objects from `verifiers.v1`.

## Minimal implementation

```python
import verifiers.v1 as vf


# One row's serializable data. Add references or other task-specific fields here.
class AdditionData(vf.TaskData):
    answer: int


# The behavior for that row. Decorated methods may request only the values they need;
# `trace` contains the full message graph and `self.data` is this task's row.
class AdditionTask(vf.Task[AdditionData]):
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == str(self.data.answer))


# The taskset is the loader. Its config can be the empty base config.
class AdditionTaskset(vf.Taskset[AdditionTask, vf.TasksetConfig]):
    def load(self) -> list[AdditionTask]:
        # Construct one behavior object around each data row and the shared task config.
        return [
            AdditionTask(
                AdditionData(idx=i, prompt=f"What is {i} + {i}?", answer=2 * i),
                self.config.task,
            )
            for i in range(100)
        ]


# Export the taskset class so the v1 loader can discover it.
__all__ = ["AdditionTaskset"]
```

Do not override `Taskset.__init__`. Implement `load()` on the taskset and put hooks and scoring on the task.

## Ownership rules

`TaskData` owns the immutable, serializable values for one row:

- prompts and optional system prompts;
- reference answers or other ground truth;
- container image, workdir, resources, and timeout requests;
- any additional typed fields scoring or a user/tool server needs.

Only `TaskData` is stored on the trace. Do not put live clients, runtime handles etc. here.

`Task` owns the behavior applied to that row:

- `setup`, `finalize`, and model-free `validate` hooks;
- stop conditions, metrics, and rewards;
- task-scoped tool declarations;
- task-facing configuration read from `self.config`.

`Taskset` owns loading and selection-time concerns. Its `load()` constructs the tasks, its direct config fields hold dataset/split/seed/sample-count knobs, and `Taskset.tools` may declare task-agnostic servers shared by one environment worker's rollouts.

The harness owns:

- the reusable agent or chat program;
- how that program is provisioned and launched;
- wiring model requests to the supplied interception endpoint and secret;
- harness-generic execution metrics.

Runtime config chooses where code executes. Task hooks should use the `vf.Runtime` interface they receive instead of assuming Docker-, Prime-, Modal-, or host-specific implementation details.

## Scoring rules

- Prefer deterministic verification grounded in the task's actual artifact or answer.
- Use an LLM judge only when semantic judgment is unavoidable.
- Metrics are for observability and do not contribute to reward, but are useful. Use them deliberately and appropriately!
- Judgement that compares the sibling traces of one env-rollout (best-of-n selection, zero-sum payoffs) lives on `Environment.score(task, traces)` — attach via `trace.record_reward`/`record_metric`; no live runtime there.
- Raise ordinary Python exceptions from rollout hooks and scoring. The rollout records them as `TaskError`.

## Validation and lifecycle

Implement `Task.validate(self, runtime)` whenever ground truth can be checked without a model. Keep rollout work on the task:

- `setup(self, trace, runtime)` — prepare files or services.
- harness execution — let the agent act.
- `finalize(self, trace, runtime)` — capture artifacts needed for scoring.
- `@vf.reward` / `@vf.metric` — evaluate while the runtime is still live.

Persist inspectable artifacts in JSON-serializable `trace.info`. Put counters and live coordination in a typed `vf.State` subclass.

## Tools

Some tasksets require custom tools. These should be the exception as they don’t work with every harness and are registered as MCP servers.

```python
class SearchToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "search"

    @vf.tool
    async def query(self, text: str) -> list[str]:
        # Tool docstrings are exposed to the model as the MCP tool description.
        """Search the task corpus."""
        return []


class SearchTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class SearchTask(vf.Task[vf.TaskData, vf.State, SearchTaskConfig]):
    # Declaring the class on Task.tools gives it one-server-per-rollout scope.
    tools = (SearchToolset,)


if __name__ == "__main__":
    SearchToolset.run()
```

Choose placement from the tool's lifetime and filesystem needs:

- **Task-scoped, own runtime:** declare the class on `Task.tools` with `vf.ToolsetConfig`. One server is launched per rollout. The default subprocess runtime is inexpensive and host-side.
- **Task-scoped, colocated:** set `colocated = true` on its `ToolsetConfig` when the tool must see the harness's filesystem or processes. It still launches once per rollout.
- **Taskset-scoped, shared:** parameterize the toolset with `vf.SharedToolsetConfig`, put the matching config field directly on `TasksetConfig`, and declare the class on `Taskset.tools`.
- **Existing remote service:** set `url` on the matching toolset config. Verifiers connects to the streamable-HTTP MCP endpoint instead of launching the class locally.


## User simulation

There is one mechanism: `user=` on `Agent.run` — any async `str -> vf.Messages` callable whose replies are injected as user turns (returning no messages ends the exchange; a task with no `prompt` is opened by the user). There is no user server to declare or place; who computes the replies is env control flow:

- **Scripted user** (replay pre-generated turns, step a game engine): a plain closure inside an `Environment.rollout()` override — see `environments/alphabet_sort_v1` or the bundled `textarena` taskset.
- **Modeled user** (an LLM playing the user): another agent role, driven live via `agents["user"].chat(...)` and relayed into the assistant's run — or just use the bundled `user-sim` env (`--env.id user-sim`), which does exactly this from the task's prompt-as-scenario.

The harness running the *assistant* must support injected user turns (`SUPPORTS_USER_SIM`): `default`, `null`, and the in-process `direct` harness do; most CLI-agent harnesses don't.

## Multi-agent environments

When one rollout is more than one agent run, export an `Environment` subclass next to the taskset: declare roles as `vf.AgentConfig` fields on a `vf.EnvParams` subclass (bound via `Environment[YourParams]`, addressed as `--env.<role>.*`), override `roles()` to map seat names to those fields (`{"solver": self.params.solver, ...}` — without it the base's single `"main"` seat runs), `rollout(task, agents)` (imperative control flow), and optionally `score(task, traces)` (sibling-dependent judgement). Before writing one, check the bundled envs (`--env.id best-of-n | judge | user-sim`) and the reference implementations (`environments/code_golf_v1`, `environments/kuhn_poker_v1`). See docs/v1/tasksets.md.

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

- Add package import-time dependencies to the taskset's own `pyproject.toml`.
- Never edit the repository root `pyproject.toml` or `uv.lock` for a taskset dependency.
- Read required external credentials at the earliest owning boundary so missing values fail clearly.
- Do not require a user-managed background server unless the contract explicitly uses a remote URL.

## Migration from v0

Map concepts directly:

| V0 | Native v1 |
| --- | --- |
| Dataset row | Typed `vf.TaskData` subclass |
| `load_environment(**kwargs)` | Exported `vf.Taskset` class + typed config |
| `Rubric` reward function | Task `@vf.reward` method |
| Parser object | Ordinary parsing inside task scoring |
| `ToolEnv` tools | `vf.Toolset` declared on `Task.tools` or `Taskset.tools` |
| `MultiTurnEnv.env_response` | a user closure the env's `rollout()` passes via `user=` |
| Dict state | Typed `vf.State` |
| Sandbox subclass | Runtime config + task hooks |

Preserve prompt, tool, and scoring equivalence before improving design. Compare representative v0 and v1 traces where feasible.

## Publish gate

After package installability, validation, and representative eval behavior are stable, ask the user whether Hub visibility should be `PUBLIC` or `PRIVATE`. Only then run:

```bash
prime env push my-task-v1 --visibility PRIVATE
```

Publishing is an external state change and requires the user's requested visibility. Do not publish merely because local verification passed.
