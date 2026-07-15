# Textify: image → ASCII/braille at the interception server

Working plan for the `sebastian/textify` feature branch. This document is the living design
doc — it is kept up to date as the design and implementation iterate. The initial commit of
this file preserves the original intent.

> Supersedes an earlier, uncommitted draft (`verifiers/v1/TEXTIFY_PLAN.md`, still sitting
> untracked in the `main` worktree) that designed textify as a *builder-applied* static
> transform on `task.prompt` and explicitly scoped out the interception-server version.
> The requirements have changed: the transform must apply to **any** vision environment
> with a config switch — no env code changes — which is exactly the interception-server
> chokepoint. The render core, config surface, and benchmarks from that draft carry over.

## 1. Motivation

Observation (Slack): gpt-5.6 and fable jump massively on maze-bench in ASCII mode but not in
image mode — evidence that labs train on games with images rendered to text. Two uses:

1. **RL / evals**: flip any vision-language environment into a pure-language environment with
   one config value (`--textify.enabled true`). Expands the set of trainable environments to
   text-only models, and creates deliberately-hard text-space perception tasks.
2. **Pretraining / SSL**: `(image, conversion params, rendered text)` triples are an automatic,
   self-supervised task on any image corpus. Emitting the conversion parameters in prose ties
   the text-space output to a describable transform. (This use needs only the pure render
   core + a config-to-prose helper, both exposed as utils.)

Honest framing: image→ASCII/braille is **lossy and hard**. ASCII-Eval / ViTC (arXiv
2410.01733) has frontier models at ~70% recognizing concepts in ASCII art; open models are
well below. This is a feature (hard benchmarks, RL signal from partial perception), not a
transparent vision↔text equivalence — docstrings must say so.

## 2. The design decision: transform at the interception server

Every model call a harness makes goes through the interception server: the harness POSTs its
native request (chat / responses / anthropic wire), the server parses the body, records the
turn onto the trace, and forwards it (eval proxy) or renders it (train client). **Images only
ever enter a rollout through request bodies** — task-prompt images serialized by the harness,
tool-result images the harness folds into its next request, user-sim images injected by
`dialect.extend`. Model *outputs* are text in every dialect (assistant bodies flatten to
text). So one transform applied to the request body at receipt covers every image source in
the system, for every harness, with zero env cooperation.

The one insertion point (`InterceptionServer.handle_request`):

```python
raw = await request.read()
body = from_json(raw)
body = textify_body(body, dialect, self.textify)   # <— HERE, before parse_request
...
prompt, tools = dialect.parse_request(body)         # trace sees textified content
if dialect.streaming(body):
    return await self._stream(...)                  # stream path shares it
```

Placing it **before** `parse_request` makes everything downstream consistent for free:

- the **trace** records the textified text (what the model actually saw — exactly right for
  training data; `is_content`/token attribution work unchanged, and the renderer path has no
  image parts left to attribute, so no multi-modal bridging is needed),
- the **eval proxy** forwards the textified native JSON to the provider,
- the **train client** re-parses the (already textified) body for the renderer,
- both the **non-streaming and streaming** paths are covered (the streaming check happens
  after body parse),
- the **user-sim loop** re-enters `get_response` with `dialect.extend(body, ...)`-built
  bodies; injected user messages are textified at injection (one extra call site, message-level).

Out of scope at this chokepoint: `aux_routes` (e.g. Anthropic `count_tokens`) stay native —
token counts may be slightly off under textify; acceptable and documented.

### Why not the alternatives

- **Builder-applied (`textify_messages` in `load()`)**: only catches `task.prompt`; misses
  tool results and user-sim turns unless every env plumbs it everywhere. Doesn't satisfy
  "any vision task with the turn of a switch". (Still available as a util for builders who
  want static control.)
- **In the eval/train clients**: two implementations instead of one, and the trace would
  record un-textified content (the trace is built from the body before the client runs).
- **Message-graph post-processing**: the provider would still receive images — no good.

## 3. Architecture: two layers

### Layer 1 — pure render core: `verifiers/v1/utils/textify.py`

No framework imports beyond `types`. Independently useful (pretraining/SSL, builder-side
static transforms, tests).

```python
class TextifyConfig(BaseConfig):
    enabled: bool = False        # identity when false; no existing eval changes
    mode: Literal["ascii", "braille"] = "ascii"  # ascii default when enabled
    width: int = 160             # output columns
    height: int | None = None    # explicit rows; None preserves image aspect
    char_aspect: float = 0.5     # cell height/width correction
    gamma: float = 1.0           # brightness curve (lum **= gamma)
    invert: bool | None = None   # None auto-inverts predominantly-light diagrams
    ramp: str = " .:-=+*#%@"     # ascii glyph ramp, dark -> light
    threshold: float = 0.5       # braille on/off cutoff
    max_chars: int | None = 40_000 # hard budget; clamp width/height to fit

def image_to_text(image, cfg) -> str            # pure numpy render, no fence
def describe(cfg) -> str                        # config as prose (SSL use-case)
def textify_messages(messages, cfg) -> Messages # typed-Messages transform (builders, extend path)
```

`BaseConfig` (pydantic-config), so it is CLI/TOML-tunable. Note vs the old draft: no
`placement` field — see §5.

Core algorithm (benchmarked in the earlier draft, carried over): decode/resize directly toward
the target grid (before float conversion) → RGB uint8 →
`height = round(width * (H/W) * char_aspect)` (braille: 2×4 sub-dots per cell) →
nearest-neighbour integer subsample → Rec.601 luminance → gamma/invert → ramp index (ascii)
or dot-bit packing to `U+2800+code` (braille). Pure numpy, deterministic, sub-ms even at 4K
(ASCII 224²: 0.53ms, 4K: 0.13ms; braille similar). Pillow is a core dependency (imported lazily) because interception receives encoded data-URI
images; ndarray input itself needs only numpy.

Token-economics note (goes in the docstring): braille packs 8 dots/char but braille
codepoints are 3 UTF-8 bytes and tokenize badly (often ~1–3 tokens/char); the default ASCII
ramp is single-byte chars that tokenize cheaply. Braille wins dots-per-token only modestly
and loses perceptual grouping — measure before assuming braille is "better for details" in
token space.

### Layer 2 — interception integration

**Dialect hook.** Each dialect knows its own image shape; add one method to `Dialect`:

```python
def textify_body(self, body: ReqT, render: Callable[[str], str]) -> ReqT:
    """Replace this format's image parts with rendered text parts, in place in the
    content structure. `render(url) -> fenced text`. Default: return body unchanged
    (a dialect without image parts needs nothing)."""
```

- chat: `{"type": "image_url", ...}` parts inside `messages[*].content` lists
- responses: `{"type": "input_image", ...}` inside `input[*].content`
- anthropic: `{"type": "image", "source": {...}}` blocks inside `messages[*].content`

Each replaced image part becomes a text part in the same position (order preserved). A
message whose content collapses to all-text stays a content list — no structural surgery.

**Config flow.** Textify is a framework concern (same argument as `max_turns`: it must apply
to any harness, enforced where model traffic passes):

- `EnvConfig.textify: TextifyConfig = TextifyConfig()` → `--textify.enabled true`,
  `--textify.width 160`, ... on eval/validate/serve; ascii is the default mode once enabled;
  rides the env-server wire via
  `EnvServerConfig` automatically; TOML `[textify]`.
- `Environment` passes it to each `Rollout`, which puts it on the `RolloutSession`
  (alongside `RolloutLimits`) — the server already routes every request to its session, so
  per-rollout config is naturally in scope at the insertion point, and a shared interception
  server multiplexing many rollouts stays correct even if future callers mix configs.

**Insertion points** (both in `interception/server.py`):
1. `handle_request`: `body = dialect.textify_body(body, render)` right after body parse,
   gated on `session.textify.enabled`; covers stream + non-stream + every turn of
   the user-sim loop (the loop reuses `body`).
2. user-sim injection: textify `user_messages` (typed `Messages` → `textify_messages`)
   before `dialect.extend(...)` and before `prompt` extension, so simulator-emitted images
   are covered and wire/trace stay consistent.

**URL policy.** Data URIs are decoded and rendered. Plain `http(s)://` image URLs are passed
through unchanged with a rate-limited warning — the interception server must not do network
fetches on the hot path (latency, credentials, nondeterminism). If real demand appears,
fetching becomes an explicit opt-in later (`fetch: bool = False`), not a default.

**Concurrency / safety.** Decode + render runs in `asyncio.to_thread` so one large image does not
stall the shared aiohttp/interception loop. Encoded images have a 25 MP decode ceiling, output has
a 40k-character default budget and 1M hard ceiling, and the per-rollout resend cache is bounded to
32 renderings.

**Failure policy.** A decode/render failure must not kill the rollout silently-weirdly:
raise `TaskError` through the existing `_fail` path (the rollout records it as data, like
every other boundary error). Malformed images are a data problem the env author must see.

## 4. Output framing (fixed, not configurable)

One fenced block per image, language tag derived from mode:

    ```image[ascii]
    <art lines>
    ```

No dimension headers, no prose labels — the literature's "parsing help" belongs in the task
prompt (builder's job). `image_to_text` returns raw art; the fence is added at the
message/body layer. `describe(cfg)` exists for the SSL use-case where the caller *wants*
the parameters in text, by choice.

## 5. What happened to `placement` (replace / append / new_message)

The Slack sketch and the old draft floated merging subsequent same-role messages or emitting
the art as a new message. At the wire level this dissolves: an image is a *content part
inside* a message, so replacing the part in place preserves message structure (tool_call_id
pairing, role alternation, harness-side assumptions) and interleaving (text–image–text order
survives). Append/new-message variants would perturb structures that harness SDKs and
providers validate. Dropped — one behavior, in-place replacement. Revisit only with concrete
evidence that placement matters for model performance.

## 6. Trace & training semantics

- The trace records textified content: `trace` = what the model saw. Replay and offline
  re-scoring see the same text. `Trace.tools`, branching, token attribution: unchanged.
- Rewards that read `trace` see text, as they should. A reward that needs the *original*
  image reads it from `self.data` (task data is untouched — textify acts on the wire, never
  on `TaskData`).
- Determinism: render is a pure function of (image bytes, config) — same rollout, same text.

## 7. Text-heavy images: future hybrid mode

Plain luminance ASCII preserves coarse geometry but destroys small text, labels, equations,
and chemical notation — exactly what the blind width-160 description experiment showed. Merely
raising width helps only until token cost dominates. Future work should be a **hybrid layout
renderer**, not a larger ramp:

1. detect text regions (OCR/layout model, opt-in dependency),
2. OCR their contents verbatim with bounding boxes/confidence,
3. render non-text regions as ASCII,
4. place OCR strings approximately at their image coordinates or emit a compact legend such as
   `[text x=0.42 y=0.18: "NAD+"]`,
5. preserve equations via OCR/LaTeX when available.

This is explicitly deferred: network-free deterministic rendering is phase one. The extension
belongs behind a separate mode/config (`ascii_ocr` or nested `ocr.enabled`) because OCR is slower,
non-deterministic across model/version changes, and introduces a new dependency/licensing surface.

## 8. Implementation plan (incremental, each step green)

1. `utils/textify.py`: config + render core (`image_to_text`, ascii + braille) + fence +
   `describe` + `textify_messages`; Pillow core dependency. Unit tests (deterministic known-array renders, braille
   bit-packing, config edge cases, no-Pillow ndarray path).
2. `Dialect.textify_body` default + chat implementation. Unit tests on wire dicts.
3. responses + anthropic implementations. Unit tests.
4. `EnvConfig.textify` → `Rollout` → `RolloutSession`; insertion in `handle_request` +
   user-sim injection. E2E: a small vision env (e.g. `mmmu-v1`) with `--textify.enabled true`
   on subprocess + docker; `mode=image` byte-identical passthrough.
5. Exports (`vf.TextifyConfig`, `vf.image_to_text`, `vf.textify_messages`, `vf.describe`),
   docs page.

## 9. Open questions (co-design)

- **Config granularity**: one global `EnvConfig.textify` for everything, or allow per-source
  overrides later (prompt vs tool-result images)? Start global; per-source only with a use case.
- **`describe()` output format**: currently one deterministic key=value-like prose line;
  whether the fence tag should carry mode only or dimensions too (lean: mode only).
- **Color**: luminance-only for now; ANSI color codes explode token counts. Revisit never?
- **Braille default params**: threshold vs adaptive (Otsu)? Start fixed threshold, keep the
  config door open.
- **Default width evidence**: width 160 is the current starting default, not settled. A blind
  six-image MMMU test at 160 matched all originals to shuffled ASCII correctly, but ASCII-only
  semantic descriptions lost text and misidentified anatomy/physics symbols; coarse geometry
  survived. An initial 5-question Physics sweep favored wider renderings but was confounded by
  large sampling/reasoning variance; a controlled sweep is required before claiming a threshold.
- **Where the eval CLI surfaces it**: `--textify.*` top-level (current plan, mirrors
  `max_turns`) vs nested under something. Lean top-level on `EnvConfig`.
- **Old draft cleanup**: delete the untracked `verifiers/v1/TEXTIFY_PLAN.md` from the main
  worktree once this branch exists (it is superseded by this document).

## 10. Log

- v1 (initial commit): refined from the Slack sketch + earlier builder-level draft; decided
  interception-server chokepoint, dialect `textify_body` hook, in-place replacement only,
  no network fetch, framework-level `EnvConfig.textify`.
- v2 (implementation): textify remains disabled by default but ascii is the default enabled mode;
  width 160 initial default; explicit height + arbitrary aspect ratio; auto-invert predominantly
  light diagrams/documents; off-event-loop decode/resize with 25 MP / 40k char safety budgets;
  bounded per-rollout render cache; nested tool-result images and aux token-count routes covered;
  fresh-process 4K PNG render measured ~65 MB above import baseline (vs reviewer-observed ~536 MB
  before resize-first refactor); text-heavy OCR/layout rendering deferred to future work.
- v3 (initial experiments): blind shuffled matching of six MMMU originals to ASCII was 6/6 at
  widths 40, 120, and 160, but 4/6 at 80 (two sparse horizontal diagrams swapped — evidence that
  nearest-neighbor phase/alignment matters, not just width). Width-160 blind descriptions preserved
  geometry but hallucinated semantics (leg anatomy -> Saturn, current wires -> frying pans) and
  could not read labels/equations. Live deepseek-v4-flash evals ran end-to-end through subprocess
  and Docker (final Docker smoke reward 1.0); tiny MMMU slices were highly noisy across generation
  budgets/settings, so no monotonic resolution threshold is claimed. Width 160 remains a conservative
  initial default; 120-160 is the plausible coarse-geometry range pending larger experiments.
