## Shared Best Practices (All Contexts)

- Create new environments with `prime env init <MY_ENV_NAME>`.
- Use `prime <command> --plain` to get better formatted command outputs.
- Use the bundled skills first for create, browse, review, eval, optimization, training, and brainstorming workflows.
- Go through the verifiers code as the source of truth.
- Use toml files and validate them first before you run them.
- Always look at the verifiers v1 code first to see whether helper functions or harnesses exist for your given task.
- Never mix v1 and legacy verifiers functions in one environment. Always prefer and use v1.* code.

## Style Rules

Use these rules when shaping public v1 APIs, configs, and environment files.

- Prefer verifiers-native task, trace, server, harness, and runtime interfaces over repeated path/import/discovery plumbing in user packages.
- Expose as few knobs in the configs as possible, but as many as needed.
- Use strict Pydantic models for structured config, tasks, messages, and state.
- A basic taskset should fit in a few dozen idiomatic lines: typed data/task/config classes, `load()`, and decorated scoring on the task.
- Do not override `Taskset.__init__`; implement `load()`. Do not override `Harness.__init__` or `User.__init__`; use `setup()`.
- Refer to the code as the source of truth.
- Avoid mutable module globals. Process-level locks/rate limiters and immutable constants are the narrow exceptions.
