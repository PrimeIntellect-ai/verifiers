# terminal-bench-2-v1

[Terminal-Bench 2](https://www.tbench.ai/) as a v1 taskset — a thin wrapper over `harbor-v1`
that pins `dataset` to `terminal-bench/terminal-bench-2` (89 agentic terminal tasks). Each
task runs in its own container; the model drives a shell to solve it and is scored pass/fail.
Needs the `harbor` CLI (`uv tool install harbor`) and a container runtime (docker/prime/modal).

## Baseline results

Run with the `rlm` harness on the **modal** runtime, one rollout per task, models served via
[Prime Intellect inference](https://pinference.ai):

```bash
uv run eval terminal-bench-2-v1 --harness.runtime.type modal --harness.id rlm \
  --max-turns 100 --timeout.rollout 3600 -m <model>
```

| Model | Accuracy (solved) | Corrected¹ | Mean reward | Errored | Capped² | Model-call 400s³ | Wall time |
|-------|------------------:|-----------:|------------:|--------:|--------:|-----------------:|----------:|
| `deepseek/deepseek-v4-flash` | **48.3%** (43/89) | 59.7% (43/72) | 0.483 | 0 | 17 | 23 | ~60 min |
| `z-ai/glm-4.7` | **29.2%** (26/89) | 33.8% (26/77) | 0.292 | 0 | 12 | 72 | ~60 min |
| `qwen/qwen3.6-27b` | **18.0%** (16/89) | 18.0% (16/89) | 0.180 | 1 | 0 | 72 | ~48 min |

<sub>89 tasks, `n=1` rollout each, concurrency 128. **Accuracy** = fraction of tasks solved
(reward `1.0`). **Errored** = rollouts that ended in a terminal error.
<br>¹ **Corrected** = accuracy over only the rollouts that finished within budget, i.e.
excluding the ones that hit a cap. None of the capped rollouts solved in any run, so this only
drops out-of-budget failures from the denominator.
<br>² **Capped** = rollouts that hit the 100-turn or 3600 s limit instead of the agent stopping
on its own (split deepseek 10/7, glm 9/3, qwen 0/0 — qwen always `agent_completed`).
<br>³ **Model-call 400s** = transient inference errors (mostly context-length on the long
rollouts) that the rollout retried/recovered from — a reliability signal, not counted as
errored.</sub>
