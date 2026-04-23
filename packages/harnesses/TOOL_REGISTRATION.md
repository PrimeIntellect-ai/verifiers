# Task Capability Registration

Composable tasksets can provide tools and skills for a task, and harnesses
register each capability in the agent's native format. The current shared tool
transport is MCP; skills are passed as a sandbox directory containing
`*/SKILL.md` files.

## Lifecycle

`ComposableEnv` owns the wiring:

1. The sandbox is created.
2. The taskset runs its normal sandbox setup.
3. `taskset.get_tools(info)` declares task tools.
4. `taskset.prepare_tools(state, tools)` materializes runtime tools, if needed.
5. `harness.with_tools(tools)` creates a rollout-specific harness when tools
   need native registration.
6. `taskset.get_skills(info)` declares task skills.
7. `taskset.prepare_skills(state, skills)` materializes runtime skills, if
   needed.
8. If the taskset has a `skills/` source directory, `ComposableEnv` uploads it
   and resolves the sandbox path into `TaskSkills.skills_dir`.
9. `harness.with_skills(skills)` creates a rollout-specific harness when skills
   need native registration.
10. The agent is installed and launched with the rollout-specific harness.

Harnesses only receive resolved `TaskTools` and `TaskSkills`. They should not
parse taskset formats such as Harbor TOML, start sidecar servers, or know which
taskset produced the capabilities.

## TaskTools

Tasksets return `verifiers.envs.composable_tools.TaskTools`:

```python
TaskTools(
    mcp_servers=[
        {
            "name": "repo",
            "transport": "streamable-http",
            "url": "http://127.0.0.1:8000/mcp",
        }
    ],
    env_vars={"HARBOR_MCP_REPO_URL": "http://127.0.0.1:8000/mcp"},
)
```

`mcp_servers` are passed to harnesses. `env_vars` are passed only when launching
the agent background job, so tasksets can compute them after sandbox-side setup.

## TaskSkills

Tasksets return `verifiers.envs.composable_skills.TaskSkills`:

```python
TaskSkills(
    source_dir=Path(__file__).parent / "skills",
)
```

Tasksets usually do not need to set `skills_dir` manually. By default,
`TaskSet.get_skills_dir()` discovers a sibling `skills/` directory next to the
taskset module, `TaskSet.get_skills(info)` wraps it as `TaskSkills.source_dir`,
and `ComposableEnv` uploads it to the path declared by the harness's
`skills_path`.

## Harness Contract

Harnesses that support task tools set `configure_tools` on their `Harness`.
Harnesses that support task skills set `configure_skills`. Most native CLI
harnesses should use `make_native_harness`, which creates both callbacks and
rebuilds the run command with merged capabilities:

```python
from harnesses import make_native_harness


def my_harness(mcp_servers=None, skills_dir=None, instruction_path="/task/instruction.md"):
    return make_native_harness(
        build_run_command=build_run_command,
        run_kwargs={"instruction_path": instruction_path},
        instruction_path=instruction_path,
        default_skills_path="/task/skills",
        mcp_servers=mcp_servers,
        skills_dir=skills_dir,
    )
```

`configure_tools` must return a new `Harness`; it should not mutate the original.
Multiple rollouts share the environment object, so the effective harness is stored
on rollout state.

For custom harnesses that do not map cleanly to a run-command builder, use
`make_configurable_harness` around a callable that returns a fresh `Harness` for
`(effective_mcp_servers, effective_skills_dir)`.

If a taskset explicitly provides tools and the harness has no `configure_tools`
callback, `Harness.with_tools()` raises a clear `ValueError`.

If a taskset provides skills and the harness has neither a `skills_path` nor a
`configure_skills` callback, `Harness.with_skills()` raises a clear `ValueError`.

## Native Harnesses

Codex, Claude Code, OpenClaw, OpenCode, Pi Mono, and Terminus 2 implement native
skill registration. Codex, Claude Code, OpenClaw, OpenCode, and Terminus 2 also
implement MCP registration. Their callbacks merge factory-provided config with
task-provided config and rebuild each harness for that rollout.

This means both forms compose:

- static harness config: `codex_harness(mcp_servers=[...])`
- taskset-provided tools: `TaskSet.get_tools(...)`

The agent sees one MCP config containing both sets.
