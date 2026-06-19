# EmptyModelResponseError Repro Notes

This branch captures an opt-in raw-response dump hook for renderer-client
responses and a saved-rollout replay that reproduces:

```text
EmptyModelResponseError('Model returned reasoning but no content and did not call any tools')
```

## Dump Hook

Set `VF_RENDERER_RESPONSE_DUMP_DIR` to write raw native renderer responses when
the parsed response has neither content nor tool calls. Set
`VF_RENDERER_RESPONSE_DUMP_ALL=1` to dump every renderer response.

The dump is written before `RendererClient.raise_from_native_response()` raises,
so it preserves the native parsed response shape, prompt, tools, state, and
sampling parameters.

## Local Repro

The repro uses the saved eval rollout row that originally failed:

```text
/root/outputs/nemotron-nano-swe/run_default/rollouts/step_10/eval_rollouts_swe-bench-verified-quick.jsonl
example_id = 2
rollout_id = 672db80b-05bb-4c62-8193-077617a38bef
```

Run against the standalone Nemotron Nano inference endpoint:

```bash
cd /root/git/research-prod/prime-rl

VLLM_API_KEY=EMPTY uv run python deps/verifiers/scripts/repro_empty_model_response.py \
  --input /root/outputs/nemotron-nano-swe/run_default/rollouts/step_10/eval_rollouts_swe-bench-verified-quick.jsonl \
  --output /root/outputs/nemotron-nano-empty-repro/probes/repro_empty_model_response.jsonl \
  --dump-dir /root/outputs/nemotron-nano-empty-repro/raw-response-dumps/repro-empty-model-response \
  --base-url http://slinky-3:18080/v1 \
  --example-id 2 \
  --rollout-id 672db80b-05bb-4c62-8193-077617a38bef \
  --seed 123 \
  --discover-max-tokens 512
```

Expected behavior:

1. The discovery request is valid. It finds the first `</think>` boundary in
   the completion and suggests `max_tokens = 132` for this seed/prompt.
2. The capped request with `max_tokens = 132` raises
   `EmptyModelResponseError`.
3. The dump directory contains a JSON payload with this summary:

```json
{
  "has_content": false,
  "has_tool_calls": false,
  "has_reasoning_content": true,
  "finish_reason": "length"
}
```

The confirmed local dump from the investigation is:

```text
/root/outputs/nemotron-nano-empty-repro/raw-response-dumps/saved-row-seed123-max132-20260619T1446Z/20260619T143615Z-empty-2-672db80b-05bb-4c62-8193-077617a38bef.json
```

## Root Cause Seen In The Dump

The native response is not a transport-level empty response. vLLM produced a
completion, but the completion ended exactly at the reasoning boundary:

```text
reasoning_content: non-empty
content: ""
tool_calls: []
finish_reason: "length"
```

For the same prompt and seed with a larger token cap, the model continues
immediately after `</think>` with a `bash` tool call. With the smaller cap, the
renderer parser extracts everything before `</think>` as reasoning, then sees no
post-thinking content or tool call. `RendererClient.raise_from_native_response()`
then raises the observed `EmptyModelResponseError`.

So the reproduced failure is generation-budget exhaustion at the
reasoning/action boundary, not a sandbox failure and not an empty HTTP response.
