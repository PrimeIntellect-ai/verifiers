# interception-v1

An example v1 environment that defines deterministic and judge-based policies directly
with `@vf.intercept`.

The task class installs two response handlers in priority order:

1. `deterministic_policy` uses a typed `vf.AssistantMessage` to select the response
   boundary and returns replacement text when an exact rule matches.
2. `judge_policy` receives the candidate, request prompt, and trace, then calls
   `vf.Judge.complete()` for semantic classification. Passing `trace` records the judge
   call and its usage.

The higher-priority deterministic handler always runs first. The lower-priority judge
sees any replacement it produced, so it classifies the cleaned result rather than the
original candidate. The reward checks `trace.interceptions` to show which decorated
handler rewrote the exchange.

## Develop

Install the package and evaluate both examples:

```bash
uv pip install -e environments/interception_v1
uv run eval interception-v1 -n 2
```

The example uses `vf.Judge()` and its default model. A handler can select another model
by constructing the judge with a config:

```python
judge = vf.Judge(vf.JudgeConfig(model="openai/gpt-5.4-nano"))
```
