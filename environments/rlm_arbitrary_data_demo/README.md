# RLM Arbitrary Data Demo

This toy environment exercises the custom data serialization path in `RLMEnv`.

The environment supports multiple `context_dtype` options so you can test different
serializers from a single entrypoint:

- `text` → string (default serializer)
- `list` → builtin serializer (random list of ints)
- `tuple` → builtin serializer (random tuple of ints)
- `nested_list` → builtin serializer (nested list of ints)
- `nested_dict` → builtin serializer (nested dict/list of ints)
- `mixed` → builtin serializer (mixed builtin containers)
- `large_list` → builtin serializer (larger list of ints)
- `polars` → custom serializer/deserializer in this environment

Run:

```bash
prime eval run rlm-arbitrary-data-demo -n 1
```

To pick a dtype explicitly:

```bash
prime eval run rlm-arbitrary-data-demo -n 1 --env-arg context_dtype=polars
```
