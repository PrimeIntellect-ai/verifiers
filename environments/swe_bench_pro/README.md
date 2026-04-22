# swebench-pro

### Overview
- **Environment ID**: `swebench-pro`
- **Short description**: SWE-bench Pro composed from `tasksets` and `harnesses`
- **Tags**: swe-bench, cli_agent, composable

### Quickstart

```bash
prime eval run swebench-pro -n 1 -r 1
```

Example TOML for the GLM-5.1 run:

```toml
model = "z-ai/glm-5.1"
provider = "prime"
api_client_type = "openai_chat_completions"
api_key_var = "PRIME_API_KEY"
api_base_url = "https://api.pinference.ai/api/v1"
save_results = true
num_examples = 3
rollouts_per_example = 1
max_concurrent = 3
max_tokens = 65536
disable_tui = true

[[eval]]
env_id = "swebench-pro"

[eval.env_args]
harness = "mini-swe-agent"
auto_download = true
limit = 3
timeout_seconds = 3600.0
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
