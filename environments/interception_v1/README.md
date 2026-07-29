# interception-v1

An example v1 environment that composes deterministic interception with an ordinary
LLM judge.

The task class installs three policies in priority order:

1. `intercept_provider_tools("web_search*")` removes matching provider-hosted tools
   from the request before inference.
2. `intercept_shell_commands("curl", "wget")` rewrites matching assistant tool calls
   before the harness can execute them.
3. `intercept_with_judge(...)` checks the remaining content after the cheaper
   deterministic rules have run.

The first task triggers the deterministic shell policy. The second task emits a marker
that the example judge rubric blocks. Their reward checks `trace.interceptions` to show
which policy actually rewrote the exchange.

## Develop

Install the package and evaluate both examples:

```bash
uv pip install -e environments/interception_v1
uv run eval interception-v1 -n 2
```

The judge uses `vf.Judge()` and its default model. To select another ordinary judge,
pass one explicitly:

```python
vf.intercept_with_judge(
    rubric,
    judge=vf.Judge(vf.JudgeConfig(model="openai/gpt-5.4-nano")),
)
```
