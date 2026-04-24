# renderers

A per-model, Python-native chat-template layer that moves prompt assembly, tool-call parsing, and reasoning parsing out of vLLM and into our own code. Lives in `packages/renderers/` in the verifiers repo; consumed by `verifiers.clients.renderer_client`. Replaces the old TITO client.

---

## 1. Why renderers?

Previously, most of "what the prompt actually looks like" was delegated to vLLM: its chat-template application, its tool-call parser, its reasoning parser. That is a fruitful source of bugs, but the reason we need something like renderers is more fundamental.

For RL we need **Token-In, Token-Out (TITO)**: the trainer must see the exact token ids the sampler saw, with an exact loss mask. This is essential for keeping train-inference KL mismatch low and keeping the trainer stepping fast — rollouts whose per-turn prefixes drift under re-tokenization cannot be stitched into a single training sample and fragment into many partial ones.

Renderers turn vLLM into a dumb TITO inference engine. Every prompt manipulation happens client-side:

- **Chat template application** — we render `messages → token ids` ourselves, either with a hand-coded Python renderer that mirrors the Jinja template or with `DefaultRenderer`, which wraps `tokenizer.apply_chat_template` but still owns the tokenization boundary.
- **Tool-call parsing** — `ToolParser` implementations scan token ids for the model's special delimiter tokens (e.g. token id `151657` for `<tool_call>` on Qwen3). A regex-on-decoded-text parser can mistake a literal `"<tool_call>"` inside user content for a real tool-call opener; matching by id can't, because regular text never tokenizes to the special-token id.
- **Reasoning parsing** — `ReasoningParser` implementations split `<think>…</think>` with a tested whitespace-preservation contract (bit-exact newline handling around the closing tag — a spot where we had a real bug that silently broke the extension property at the think boundary).

### What we gain

- **RL correctness.** A prompt/completion split we control, which is exactly what `bridge_to_next_turn` relies on to keep rollouts from fragmenting under truncation or re-tokenization.
- **Testable parity.** Per-model renderers are plain Python. We can render the same conversation through the renderer and through HF's `apply_chat_template` and assert token-level parity. Every edge case (empty thinking, multiple tool calls, truncated turns) becomes a unit test instead of undefined behavior buried inside Jinja.
- **Escape hatch.** Anything without a hand-coded renderer falls back to `DefaultRenderer` (a generic `apply_chat_template` wrapper), which mirrors the previous TITO path.

---

## 2. API

### Renderer protocol

Every renderer implements this (`renderers.base.Renderer`):

```python
render(messages, *, tools=None, add_generation_prompt=False) -> RenderedTokens
render_ids(messages, *, tools=None, add_generation_prompt=False) -> list[int]
parse_response(token_ids) -> ParsedResponse
get_stop_token_ids() -> list[int]
bridge_to_next_turn(prev_prompt_ids, prev_completion_ids, new_messages, *, tools=None) -> list[int] | None
```

`RenderedTokens` carries `token_ids` **and** `message_indices` — one entry per token attributing each token to its source message, so the loss mask can be built from message roles. This lets `build_supervised_sample` assemble samples in O(1) renders instead of O(n).

`ParsedResponse` is `(content, reasoning_content, tool_calls)`.

Round-trip invariant: rendering `[user, assistant(content=X, reasoning=Y, tool_calls=[T])]` to token ids, slicing out the assistant completion, and feeding it through `parse_response` returns an equivalent structured assistant message (modulo field formatting). Tested per-renderer in `tests/test_roundtrip.py`.

### Picking a renderer

```python
from transformers import AutoTokenizer
from renderers.base import create_renderer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
r = create_renderer(tok, renderer="auto")   # → Qwen3Renderer
```

`renderer="auto"` matches `tokenizer.name_or_path` against `MODEL_RENDERER_MAP` by **exact match**. Prefix matching is intentionally off: two models with the same architecture can ship different chat templates (base vs instruct, fine-tune renames), and prefix routing would silently pick a renderer that doesn't produce template-parity output. Fine-tunes must pass `renderer=<name>` explicitly; unknown names fall back to `DefaultRenderer`.

Available hand-coded renderers: `qwen3`, `qwen3_vl`, `qwen3.5`, `glm5`, `glm4.5`, `minimax-m2`, `deepseek_v3`, `kimi_k2`, `kimi_k25`, `nemotron3`, `gpt_oss`, `default`.

### Pools

For multi-threaded pre-tokenization, use `RendererPool`:

```python
from renderers.base import create_renderer_pool

pool = create_renderer_pool("Qwen/Qwen3-8B", renderer="auto", size=16)
with pool.checkout() as r:
    ids = r.render_ids(messages)
```

Each slot owns its own tokenizer copy. Construction fans out across a thread pool.

---

## 3. The bridge (the core contract)

After rollout step *N* the agent has produced `prompt_ids_N` and `completion_ids_N`. Call their concatenation *stream_N* — the exact bytes vLLM saw at sampling time plus what it emitted. For step *N+1* we require: **`prompt_ids_{N+1}` starts with *stream_N* token-for-token**, then continues with new tokens for the incoming tool result / user message / generation-prompt opener. This is the **extension property**.

```
stream_N       = [t0, t1, t2, …, tK]                    # prompt_N + completion_N, length K+1
prompt_ids_N+1 = [t0, t1, t2, …, tK, tK+1, tK+2, …]     # identical prefix, then new turn bytes
```

The next turn's prompt must **extend** the previous turn's tokens; it must never re-tokenize them.

If it holds, `interleave_rollout` merges the two steps into one training sample with one clean `completion_mask`. If it doesn't, the rollout fragments into multiple samples and `samples_per_rollout` drifts above 1.0 — significantly slowing the trainer without adding signal.

The bridge is what produces the tokens needed to uphold the extension property in accordance with the chat template the model was trained on.

### Per-renderer bridges

Each hand-coded renderer implements `bridge_to_next_turn` directly for its model's chat template — no shared generic helper, just Python that knows what tokens the template would insert between turns. Qwen3's bridge knows about `<|im_start|>role\n … <|im_end|>\n`; GLM's bridge knows that turns end when the next role marker appears; DeepSeek V3, Kimi K2/K2.5, Nemotron-3, GPT-OSS, MiniMax each have their own. On a clean stop, vLLM's `completion_ids` already includes the template's close token; on truncation, the renderer synthesizes the canonical close (`<|im_end|>`, `<|endoftext|>`, or the equivalent for that model) so the extension invariant still holds, and the synthetic close is masked out of the loss because the model didn't produce it.

`DefaultRenderer.bridge_to_next_turn` returns `None` by default (forcing a full `apply_chat_template` re-render, which is where modes a–e in §4 strike). Hand-coded bridges are what actually close the gap.

### Turn truncation

When a turn hits `max_tokens`, its `completion_ids` have no end-of-turn marker. A hand-coded renderer's bridge appends the template's canonical turn-close (`<|im_end|>`, `<|endoftext|>`, or equivalent) to the truncated completion and emits new messages on top. The synthetic close lands in `prompt_ids` of the merged sample with `prompt_mask=False`, so loss and KL never see it, and the extension invariant still holds.

---

## 4. Breaking behaviors you might not expect

Everything in this section has been observed in production RL runs on Qwen3, Qwen3.5, GLM-4.5, and opencode-scaffolded environments. Each is a concrete reason why the "render full history through `apply_chat_template` every turn" pattern breaks the extension property. Hand-coded renderers sidestep all of them because `bridge_to_next_turn` never re-renders prior turns.

### a. Boolean type round-trip

vLLM emits a literal `false` inside a parameter block; the client parses `<parameter=dry_run>false</parameter>` into a Python `bool(False)`; `apply_chat_template` re-renders via `str(False)` → `"False"`. Capital F. Every rollout with a boolean parameter breaks on re-render.

```
prev stream: '<parameter=dry_run>\nfalse\n</parameter>'
re-rendered: '<parameter=dry_run>\nFalse\n</parameter>'
```

Reproducible on Qwen3.5-35B-A3B + mini-swe-agent-plus: roughly 50% break rate per rollout (32 of 64 rollouts in a single step).

### b. BPE retokenization drift

The BPE tokenizer is context-sensitive. The same substring tokenizes differently depending on neighboring bytes, which can shift by one whitespace or boundary character between the raw completion and the re-rendered history. Example around `jsonp` inside a Python snippet:

```
prev ids: [..., 2164, 79, 50586, ...]   # 'json' + 'p' + 'enderer'  (3 tokens)
cur  ids: [..., 55137,    50586, ...]   # 'jsonp' + 'enderer'       (2 tokens)
```

Same text, different token ids, different token count — from that point on every subsequent token id is shifted. Same class of bug we hit when `json.dumps` (Python, `{"k": "v"}` with spaces) vs. `JSON.stringify` (JS, `{"k":"v"}` compact) produced different raw arg bytes in an opencode scaffold and cascaded BPE drift through thousands of tokens of Python code.

### c. Tool-call XML structure drift

vLLM emits a no-arg tool call with a stylistic empty `</parameter>`; the Jinja template re-renders the reconstructed dict without it:

```
prev stream: '<function=echo ...>\n</parameter>\n</function>\n</tool_call>'
re-rendered: '<function=echo ...>\n</function>\n</tool_call>'
```

The `</parameter>\n` vanishes on re-render. Extension property broken at the close of every such call.

### d. Thinking stripped from non-latest assistant turns

Some chat templates strip `<think>…</think>` blocks out of prior assistant turns when re-rendering history. The rollout's recorded stream has the thinking content; the next turn's re-rendered prompt does not. Applies across the Qwen3-series under certain message shapes.

### e. Max-seq-len truncation zeroing the anchor

This one is client-side. When `parse_response_tokens` enforces the trainer's `max_seq_len`, it zeros out `completion_ids` whenever `prompt_len > max_seq_len`. On the next turn, the bridge's anchor (`prev_prompt_ids + prev_completion_ids`) is empty, so the bridge returns `None` and the caller falls back to full re-render — triggering modes a–d for every rollout that truncated. `_step_token_ids` now prefers populated `step.tokens` and falls back to the raw, un-truncated `step.response.message.tokens` specifically to defend against this.

### f. Scaffold-level history rewriting

Some agent scaffolds (e.g. opencode's AI-SDK `experimental_repairToolCall` hook) rewrite tool calls on the client before sending them back as history. If the model emits `Bash` (capital B), the hook rewrites it to a synthetic `{name: "invalid", input: {tool: "Bash", error: "Model tried to call unavailable tool Bash"}}`. The next turn's prompt now contains a tool call that the model never actually emitted — an unfixable extension break as long as the scaffold is in the loop. Renderers' bridge cannot help here because the drift is not in rendering; it's in the history the scaffold handed us.

### How renderers avoids the client-side ones

`bridge_to_next_turn(prev_prompt_ids, prev_completion_ids, new_messages, ...)` constructs the next-turn prompt as `prev_prompt_ids + prev_completion_ids + render_only_new_messages(...)`. It never re-parses prior tool call arguments, never re-tokenizes prior turns, never re-serializes prior tool-call XML. Modes a–e disappear by construction. Mode f is structural and outside the renderer's reach.

On Qwen3.5-35B-A3B + mini-swe-agent-plus step 0, the delta with renderers-v2 vs. `apply_chat_template`-based TITO:

| client path                                | breaks per step 0 | samples from 64 rollouts |
|--------------------------------------------|-------------------|---------------------------|
| `apply_chat_template` (full re-render)     | 32                | 77                        |
| renderers `bridge_to_next_turn`            | 0                 | 64                        |

---

## 5. `DefaultRenderer`

Fallback for anything without a hand-coded renderer. Wraps `apply_chat_template` and accepts optional `tool_parser` / `reasoning_parser` kwargs (mirroring vLLM's convention) — it does its best to keep the Renderer contract even without model-specific knowledge.

That said, **prefer a hand-coded renderer** for any model you actually train on. Hand-coded renderers are the only path that closes all the extension-property gaps in §4 by construction. `DefaultRenderer` is there so an unknown model doesn't block you from running — not as the recommended steady state. If your model doesn't have one yet, implementing a renderer for it is a few hundred lines of Python (`render_ids` + `parse_response` + `bridge_to_next_turn`).

---

## 6. VLM support

Renderers currently do not support the VLM (image) path. `Qwen3VLRenderer` exists and works for text-only conversations against Qwen3-VL checkpoints, but `ImagePart` is not plumbed through render / parse / bridge or the client. Adding real VLM support requires (a) carrying image payloads through the content-part model end-to-end and (b) deciding how image tokens interact with the extension invariant across multi-turn steps. Out of scope for this iteration.

---

## 7. Patched vs. "correct" chat templates

Many chat templates are poorly suited to RL — they re-tokenize history on every turn, or normalize boolean/JSON values, or auto-strip thinking content from past turns, each of which breaks the extension property.

There is a balance to strike between exact parity with the shipped template — which some downstream consumers expect — and giving users faster, more correct training.

Roadmap: add a `use_patched` flag on renderers whose shipped template is sub-optimal for RL. The patched variant would render the same surface form but avoid the known extension-breaking patterns.

### Training-inference parity in production

Downstream customers typically host the trained model on vLLM. Training with renderers (client-side TITO) and then serving through vLLM's default `/v1/chat/completions` path is a source of subtle drift.

Approach: train with renderers (Renderer client), but run evals during training through the normal `/v1/chat/completions` endpoint — that is the most prod-like setting and surfaces any train-to-serve mismatch before it ships.

---

## 8. Validation surface

Environments used to validate renderer parity and break behavior end-to-end:

- Reverse-Text — single-turn, no tools
- Wordle — multi-turn, deterministic feedback
- OpenCode-Math — multi-turn tool-calling scaffold with invalid-tool rewrites
- RLM-SWE — multi-turn SWE tool-calling against remote sandbox

Renderer-by-renderer: round-trip parity tests (`render` → `parse_response` returns the original messages) and token-level parity against `apply_chat_template` live in `packages/renderers/tests/`.

---

## 9. Resources

- **verifiers PR** — Renderers package + replace TITO client with Renderer client: [PrimeIntellect-ai/verifiers#1068](https://github.com/PrimeIntellect-ai/verifiers/pull/1068)
- **prime-rl PR** — switch RL trainer to renderers: [PrimeIntellect-ai/prime-rl#2278](https://github.com/PrimeIntellect-ai/prime-rl/pull/2278)
