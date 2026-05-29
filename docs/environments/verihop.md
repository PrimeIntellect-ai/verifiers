# VeriHop

VeriHop is a **multi-hop multimodal** environment: one synthetic image per episode, a chain of
dependent questions (count red circles → count blue → sum), and a rubric that can score both
**final boxed answers** and **per-hop** responses.

## Install

From the `verifiers` repository:

```bash
uv pip install -e environments/verihop
```

## API

- `verihop.synthesize(...)` → `datasets.Dataset` with `prompt`, `answer`, `info`.
- `verihop.load_environment(...)` → `VeriHopEnv` or `VeriHopToolEnv`.
- `vf.add_image(message, image)` (core `verifiers`) appends an OpenAI-style `image_url` block to a user message dict.

## Message contract

Models should emit `<hop_answer>...</hop_answer>` on **every** hop. The final hop must also
include `\boxed{n}` for the outcome reward.

Optional grounding for denser process scores:

```text
<grounding bbox="x0,y0,x1,y1" desc="red circles"/>
```

## Tools mode

`load_environment(use_tools=True)` registers PIL helpers (`crop_region`, `zoom_center`,
`count_color_blobs`). The rollout image is injected as `_pil_image` (hidden from the JSON
schema). Hops advance when the assistant sends a **non-tool** message that closes the hop.

## Further reading

See `environments/verihop/README.md` for motivation and defaults.
