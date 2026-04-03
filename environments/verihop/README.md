# VeriHop

Multi-hop **visual** RLVR tasks with **procedural** scenes, optional **PIL-based tools**
(`crop_region`, `zoom_center`, `count_color_blobs`), and a **dense** rubric (per-hop +
boxed outcome).

## Layout

- `verihop.synthesize()` — builds a Hugging Face `Dataset` with `prompt`, `answer`, `info['verihop']`.
- `VeriHopEnv` — multi-turn chain without tools (image on the first user turn only).
- `VeriHopToolEnv` — same chain, but the model may call tools between hops; hops advance on a **text-only** assistant turn that includes `<hop_answer>...</hop_answer>`.
- `VeriHopRubric` — `outcome_weight` on final `\boxed{}` match, `process_weight` on hop answers (+ optional grounding IoU vs metadata).

## Hop format

Each hop should include an answer in `<hop_answer>...</hop_answer>`. The **last** hop must also put the final integer in `\boxed{n}`.

Grounding tags (optional, for process rewards):

`<grounding bbox="x0,y0,x1,y1" desc="..."/>`

Boxes use **pixel coordinates** in the original image space (same as PIL).

## Usage

```python
from verihop import load_environment

env = load_environment(num_samples=1024, seed=0, use_tools=False)
```

With tools:

```python
env = load_environment(num_samples=512, use_tools=True)
```

## Prime RL

Point your orchestrator env at this package after `uv pip install -e environments/verihop`
(or publish to the hub). Use `name = "verihop"` / your env id and pass `use_tools` via the
environment args your runner supports.

See `examples/train_with_prime_rl.py` for a commented template.
