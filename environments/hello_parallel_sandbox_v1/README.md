# hello-parallel-sandbox-v1

V1 example where a sandboxed harness program keeps its primary program sandbox
alive across rollout and update stages.

The parent harness runs the base loop inside its primary sandbox
(`program={"sandbox": true}`). The taskset contributes a rollout-scoped `bash`
tool bound with `sandbox="program"`, so tool calls execute in that primary
program sandbox instead of creating a separate tool sandbox. The parent writes
`/tmp/answer.txt`, then:

1. two update-stage audits run concurrently and store serializable findings in
   `state.extras` and `state.artifacts`;
2. reward scoring reads those findings and writes a scalar rollout reward.

```bash
prime eval run hello-parallel-sandbox-v1 -m openai/gpt-5.4-mini -n 3 -r 1 -t 4096
```

Environment args:

- `num_examples`: number of built-in tasks to expose, default all tasks.
- `max_turns`: max parent-loop turns, default `4`.
