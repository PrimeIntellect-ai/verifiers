# swebench-pro

### Overview
- **Environment ID**: `swebench-pro`
- **Short description**: SWE-bench Pro composed from `tasksets` and `harnesses`
- **Tags**: swe-bench, cli_agent, composable

### Quickstart

```bash
prime eval run swebench-pro -n 1 -r 1
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `ScaleAI/SWE-bench_Pro` | Hugging Face dataset |
| `dataset_split` | str | `test` | Dataset split |
| `max_examples` | int | `-1` | Limit examples; `-1` uses all |
| `limit` | int | unset | Legacy alias for `max_examples` |
| `harness` | str/table | unset | Harness name or generic harness factory config |
| `harness_config` | table | `{agent = "openclaw"}` | Harness factory config |

Harness config is resolved by `harnesses.build_harness_from_config`. For example,
`agent = "codex"` resolves to `harnesses.codex.codex_harness`; `factory =
"pkg.module.make_harness"` calls a custom factory. A harness can also be passed
as a table, such as `harness = { factory = "pkg.module.make_harness" }`.
Existing eval TOMLs may put harness options directly under `[eval.env_args]`,
such as `harness = "opencode"` with `opencode_disabled_tools = [...]`;
selected-harness prefixes and the generic `agent_` prefix are stripped before
the factory is called.
