"""Per-token attribution invariants for ``RenderedTokens.message_indices``.

Why this file exists
--------------------
The test barrage in ``test_render_ids`` checks token-id parity against
``apply_chat_template`` — useful, but only for token *bytes*. The
per-token attribution (``RenderedTokens.message_indices``) was never
covered, even though it directly drives the loss mask in
``build_training_sample``.

That gap surfaced through two real bugs:

- ``KimiK2Renderer.render``'s unknown-role fallback emitted the closing
  ``<|im_end|>`` with the post-normalisation index ``i`` instead of the
  caller-relative ``oi``.
- ``KimiK25Renderer.render`` used the raw post-normalisation index
  everywhere, shifting *every* index by one whenever auto-system
  injection happened.

In both cases token IDs were unchanged (so render-parity still passed),
but ``message_indices`` pointed at wrong (or out-of-range) messages.

Tests in this file:

1. ``test_message_indices_in_range`` — parametrised via the conftest
   matrix. Renders a no-system multi-role conversation and asserts the
   contract: every entry in ``message_indices`` is either ``-1``
   (structural scaffolding) or a valid caller-relative index, and every
   caller message contributes at least one token. The no-system input
   forces renderers that auto-inject default system messages
   (Kimi K2 / K2.5 / K2.6) into the injection path — the path where the
   post-normalisation index can shift away from the caller-relative one.

2. ``test_kimi_k2_unknown_role_message_indices`` — Kimi-K2-specific
   regression: triggers auto-system injection plus an unknown role,
   which is exactly the path the original bug was on. Without the fix
   this test catches a stray index that points past the caller's last
   message.
"""

from __future__ import annotations


def test_message_indices_in_range(model_name, renderer):
    """Every emitted token's ``message_indices`` must be in
    ``[-1, len(messages))``. ``-1`` is the documented sentinel for
    structural scaffolding (e.g. trailing generation prompt).

    Uses a no-system input so renderers that auto-inject default system
    messages (Kimi K2 / K2.5 / K2.6) actually take the injection path —
    that's the path where the post-normalisation index can shift away
    from the caller-relative one and the bug surfaces."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs, add_generation_prompt=True)
    n = len(msgs)

    assert len(rendered.token_ids) == len(rendered.message_indices), (
        f"{model_name}: token_ids and message_indices length mismatch"
    )
    bad = [
        (k, idx)
        for k, idx in enumerate(rendered.message_indices)
        if not (idx == -1 or 0 <= idx < n)
    ]
    assert not bad, (
        f"{model_name}: out-of-range message_indices entries (k, idx): {bad[:8]}"
    )

    # Every caller message must contribute at least one token.
    seen = set(rendered.message_indices)
    missing = [k for k in range(n) if k not in seen]
    assert not missing, (
        f"{model_name}: messages not represented in message_indices: {missing}"
    )


def test_kimi_k2_unknown_role_message_indices():
    """Regression for the ``i`` vs ``oi`` mistake in
    ``KimiK2Renderer.render``'s unknown-role fallback. To surface, we
    need (a) auto-system injection (no system in the input) and (b) an
    unknown role hitting the fallback, and (c) the unknown role at the
    *last* caller position so the post-normalisation index points past
    the caller's input length.

    Layout under test (caller indices on the right):

        normalized[0] = auto-injected system     (oi = -1)
        normalized[1] = user                     (oi = 0)
        normalized[2] = developer  (unknown)     (oi = 1)   ← bug emits i=2

    With the bug, ``<|im_end|>`` on the developer message is emitted
    with the post-normalisation index ``i=2`` — out of range for a
    caller list of length 2.
    """
    from transformers import AutoTokenizer

    from renderers import create_renderer

    tok = AutoTokenizer.from_pretrained(
        "moonshotai/Kimi-K2-Instruct", trust_remote_code=True
    )
    renderer = create_renderer(tok, renderer="auto")

    msgs = [
        {"role": "user", "content": "hi"},
        # Unknown role hits the system-style fallback in KimiK2Renderer.render.
        {"role": "developer", "content": "internal note"},
    ]
    rendered = renderer.render(msgs)

    n = len(msgs)
    bad = [
        (k, idx)
        for k, idx in enumerate(rendered.message_indices)
        if not (idx == -1 or 0 <= idx < n)
    ]
    assert not bad, (
        f"out-of-range message_indices on Kimi K2 unknown-role fallback "
        f"(k, idx): {bad[:8]}"
    )
