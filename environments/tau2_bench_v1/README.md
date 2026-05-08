# tau2-bench-v1

### Overview
- **Environment ID**: `tau2-bench-v1`
- **Short description**: v1 taskset-owned user/tool simulator pattern for tau2-style multi-turn benchmark tasks.
- **Tags**: v1, user, tools, multi-turn

### Task
- **Type**: multi-turn user simulation
- **Rubric overview**: Uses v1 rollout signals and taskset state to score task completion.

### Quickstart
```bash
prime eval run tau2-bench-v1
```

### Notes
- The user simulator and tools are scoped through v1 runtime objects rather than stored directly in state.
