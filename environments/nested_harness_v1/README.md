# nested-harness-v1

### Overview
- **Environment ID**: `nested-harness-v1`
- **Short description**: v1 example showing a tool that calls a child `Harness` as its own rollout scope.
- **Tags**: v1, nested-harness, tools

### Task
- **Type**: single-turn tool use
- **Rubric overview**: Checks whether the parent rollout returns the child harness answer.

### Quickstart
```bash
prime eval run nested-harness-v1
```

### Notes
- The child harness inherits model controls from the parent runtime and receives its own rollout-local state.
