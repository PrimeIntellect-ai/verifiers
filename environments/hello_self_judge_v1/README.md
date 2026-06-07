# hello-self-judge-v1

V1 example where the answer rollout is reviewed by taskset-owned update logic.

The answer harness runs the base loop. Each task asks the model to answer with a
sources line. The taskset then:

1. stores judge findings under `state.extras["judge"]` and
   `state.artifacts["judge_findings"]`;
2. reports source-mention metrics;
3. computes the reward from the serialized findings.

```bash
prime eval run hello-self-judge-v1 -m openai/gpt-5.4-mini -n 3 -r 1 -t 4096
```

Environment args:

- `num_examples`: number of built-in tasks to expose, default all tasks.
- `max_turns`: max answer-loop turns, default `8`.
