# hello-parallel-sandbox-v1

V1 example where a sandboxed harness program keeps its primary program sandbox
alive across rollout and update stages.

The parent harness runs the base loop inside its primary sandbox
(`program={"sandbox": true}`). The taskset contributes a rollout-scoped `bash`
tool bound with `sandbox="program"`, so tool calls execute in that primary
program sandbox instead of creating a separate tool sandbox. The parent writes
`/tmp/answer.txt`, then:

1. two update-stage child harnesses run concurrently, borrow the live `model`
   and `bash` tool, append their model calls to the public trajectory, and
   inspect the same primary sandbox before rollout cleanup;
2. a reward-stage child harness borrows the same live `model` and `bash` tool,
   keeps its trajectory private, inspects the same sandbox, and returns a JSON
   score.

```bash
prime env install hello-parallel-sandbox-v1
prime eval run hello-parallel-sandbox-v1 -m openai/gpt-5.4-mini -n 3 -r 1 -t 4096
```

Environment args:

- `num_examples`: number of built-in tasks to expose, default all tasks.
- `max_turns`: max parent-loop turns, default `4`.
