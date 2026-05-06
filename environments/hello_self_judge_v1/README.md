# hello-self-judge-v1

V1 example where the answer rollout and the judge rollout share live runtime
resources.

The answer harness runs the base loop locally. The taskset contributes a
rollout-scoped, sandbox-backed `bash` tool. Each task asks the model to fetch
public web pages, write `/tmp/evidence.md`, and answer from the evidence. The
taskset then:

1. runs an update-stage judge harness that borrows the live `model` and
   `bash` tool, appends to the public trajectory, and uses the same
   tool-owned sandbox to inspect `/tmp/evidence.md`;
2. stores the update judge's findings under `state["update_judge"]`;
3. runs a reward-stage judge harness that borrows only the live `model`, keeps
   its trajectory private, stores JSON under `state["judge"]`, and returns its
   score.

```bash
prime env install hello-self-judge-v1
prime eval run hello-self-judge-v1 -m openai/gpt-5.4-mini -n 3 -r 1 -t 4096
```

Environment args:

- `num_examples`: number of built-in tasks to expose, default all tasks.
- `max_turns`: max answer-loop turns, default `8`.
