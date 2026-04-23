# renderers

Per-model, Python-native chat templates + a fallback `DefaultRenderer` that
wraps `tokenizer.apply_chat_template`. Used by the verifiers renderer client
to tokenize conversations client-side before hitting vLLM `/v1/generate`.

## Picking a renderer

```python
from transformers import AutoTokenizer
from renderers.base import create_renderer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
r = create_renderer(tok, renderer="auto")  # → Qwen3Renderer
```

`renderer="auto"` looks up the tokenizer's `name_or_path` in
`MODEL_RENDERER_MAP` by **exact match**. Prefix matching is intentionally
not used — models with the same architecture can ship different chat
templates (base vs instruct, fine-tune renames), and routing them by
prefix would silently pick a renderer that doesn't produce
template-parity output. Fine-tunes and renamed checkpoints must pass
`renderer=<name>` explicitly; anything unknown falls back to
`DefaultRenderer`.

To force a specific renderer:

```python
r = create_renderer(tok, renderer="qwen3")
```

Available: `qwen3`, `qwen3_vl`, `qwen3.5`, `glm5`, `glm4.5`, `minimax-m2`,
`deepseek_v3`, `kimi_k2`, `kimi_k25`, `nemotron3`, `gpt_oss`, `default`.

## DefaultRenderer knobs

Only `DefaultRenderer` reads these. Hand-coded renderers hard-code their own
behavior.

| kwarg | default | purpose |
|---|---|---|
| `tool_parser` | `None` | Name of a tool parser in `renderers.parsers`. |
| `reasoning_parser` | `None` | Name of a reasoning parser in `renderers.parsers`. Falls back to a `<think>…</think>` sniff. |
| `synthesize_close_on_truncation` | `False` | See below. |

### `synthesize_close_on_truncation`

When a turn hits `max_tokens` in vLLM, its `completion_ids` have no
end-of-turn marker. The multi-turn bridge (`bridge_to_next_turn`) can't
find a turn-close boundary and returns `None`, forcing a full
`apply_chat_template` re-render. That re-render runs the raw completion
text back through BPE, which often doesn't round-trip cleanly at the
truncation boundary — the extension property breaks and
`interleave_rollout` fragments the rollout into multiple
`TrainingSample`s. Observable symptom: `samples_per_rollout` creeps
above 1.0 roughly in step with the truncation rate.

With `synthesize_close_on_truncation=True`, the bridge appends the
renderer's canonical turn-close (for `DefaultRenderer`: the tokenizer's
`eos_token_id`; for hand-coded renderers: the template's `<|im_end|>`,
`<|endoftext|>`, or equivalent) to the end of the truncated completion
and hand-emits the new messages on top. Returned `new_prompt_ids` start
with the exact `prev_prompt_ids + prev_completion_ids`, so extension
holds and the rollout stays one sample. The synthetic token lands in
`prompt_ids` of the merged sample with `prompt_mask=False`, so loss
and KL never see it. Hand-coded renderers default this to `True`
because they know their template; `DefaultRenderer` keeps it opt-in
because it doesn't.

**Enable when:** you know `tokenizer.eos_token_id` is the canonical
end-of-turn marker for the template you're using. This is true for all
chatml-family fine-tunes (Qwen3, GLM, DeepSeek, Kimi, MiniMax, Nemotron,
GPT-OSS) — if you're using one of those, prefer the model-specific
renderer, which sets this on by construction. Use the knob on
`DefaultRenderer` for fine-tunes whose `name_or_path` isn't in
`MODEL_RENDERER_MAP` (e.g. `your-org/Qwen3-something`).

**Leave off when:** the template's end-of-turn marker isn't
`eos_token_id` (rare but possible), or you want to exactly mirror main's
TITO-on-truncation behavior for A/B parity. With the flag off,
`DefaultRenderer` falls back to `apply_chat_template` re-render on
truncation, matching the pre-renderers behavior.

```python
r = create_renderer(tok, renderer="default",
                    synthesize_close_on_truncation=True)
```

Or with a pool:

```python
from renderers.base import create_renderer_pool

pool = create_renderer_pool(
    "your-org/Qwen3-1.7B-SomeFineTune",
    renderer="auto",
    size=16,
    synthesize_close_on_truncation=True,
)
```
