"""Shared bridge_to_next_turn implementations.

A "bridge" computes the new step's ``prompt_ids`` by extending
``previous_prompt_ids + previous_completion_ids`` with whatever tokens
the next turn(s) add, without re-rendering the tokens the model already
emitted. The invariant the caller relies on is:

    new_prompt_ids[: len(prev_prompt + prev_completion)] == prev_prompt + prev_completion

which keeps ``interleave_rollout`` from fragmenting the rollout.

Two shapes cover everything we ship today:

* ``chatml_bridge`` — templates that wrap each message as
  ``<|start|>role\\ncontent<|end|>\\n``. vLLM emits the close token on a
  clean stop and includes it in ``completion_ids``. Truncation is
  handled by synthesizing the template's canonical close
  (``get_stop_token_ids()[0]``).

* ``glm_bridge`` — templates that open the next turn with a
  ``<|user|>`` / ``<|observation|>`` marker instead of closing the
  current one. vLLM's stop_token_ids includes those markers so they
  land in ``completion_ids`` naturally; truncation just means they
  aren't there yet, and the bridge appends them as the next turn
  renders.

Both walk the renderer via a dummy-assistant render to find where
``render([dummy_assistant])`` ends relative to
``render([dummy_assistant, *new_messages], add_generation_prompt=True)``.
"""

from __future__ import annotations

from renderers.base import Message, ToolSpec


def chatml_bridge(
    renderer,
    previous_prompt_ids: list[int],
    previous_completion_ids: list[int],
    new_messages: list[Message],
    *,
    tools: list[ToolSpec] | None = None,
) -> list[int] | None:
    """Bridge for chatml-style templates with explicit per-turn close tokens.

    Synthesizes ``get_stop_token_ids()[0]`` when ``previous_completion_ids``
    doesn't end with a stop token (i.e. vLLM hit ``max_tokens``), then
    uses the dummy-assistant render to recover any "between-turn"
    tokens (typically the trailing ``\\n`` that chatml includes in a
    message's rendering).
    """
    previous_ids = list(previous_prompt_ids) + list(previous_completion_ids)
    if not previous_ids or not new_messages:
        return None

    try:
        stop_token_ids_list = list(renderer.get_stop_token_ids())
    except Exception:
        stop_token_ids_list = []
    stop_token_ids = set(stop_token_ids_list)

    # Find where prev_completion's close actually lives — scan backwards
    # through completion_ids only (not prompt_ids).
    boundary_idx: int | None = None
    if stop_token_ids:
        for idx in range(len(previous_ids) - 1, len(previous_prompt_ids) - 1, -1):
            if previous_ids[idx] in stop_token_ids:
                boundary_idx = idx
                break

    if boundary_idx is None:
        # Truncation: synthesize the template's canonical close.
        if not stop_token_ids_list:
            return None
        previous_ids = previous_ids + [stop_token_ids_list[0]]
        boundary_idx = len(previous_ids) - 1

    previous_ids = previous_ids[: boundary_idx + 1]
    boundary_token_id = previous_ids[-1]

    dummy_assistant: Message = {"role": "assistant", "content": "x"}
    try:
        bridge_full_ids = renderer.render_ids(
            [dummy_assistant, *new_messages], tools=tools, add_generation_prompt=True
        )
        bridge_base_ids = renderer.render_ids(
            [dummy_assistant], tools=tools, add_generation_prompt=False
        )
    except Exception:
        return None

    if bridge_full_ids[: len(bridge_base_ids)] != bridge_base_ids:
        return None

    # Locate the boundary token's last occurrence inside bridge_base so we
    # can start slicing bridge_full right after it — this recovers the
    # between-turn tokens that live AFTER the close in the template
    # (chatml's trailing ``\n``, etc.).
    gap: int | None = None
    for idx in range(len(bridge_base_ids) - 1, -1, -1):
        if bridge_base_ids[idx] == boundary_token_id:
            gap = len(bridge_base_ids) - idx - 1
            break
    if gap is None:
        return None

    bridge_ids = list(bridge_full_ids[len(bridge_base_ids) - gap :])
    if bridge_ids and bridge_ids[0] == boundary_token_id:
        bridge_ids = bridge_ids[1:]
    return previous_ids + bridge_ids


def glm_bridge(
    renderer,
    previous_prompt_ids: list[int],
    previous_completion_ids: list[int],
    new_messages: list[Message],
    *,
    tools: list[ToolSpec] | None = None,
) -> list[int] | None:
    """Bridge for GLM-style templates that use next-turn markers.

    GLM has no per-turn close token: an assistant turn ends when the
    next ``<|user|>`` / ``<|observation|>`` marker appears. The append
    delta is just ``render([dummy_assistant, *new_messages], gen=True)``
    minus ``render([dummy_assistant])``. If ``previous_ids`` already
    ends with the next-turn marker (vLLM stopped on it), dedup the
    leading marker in the delta.
    """
    previous_ids = list(previous_prompt_ids) + list(previous_completion_ids)
    if not previous_ids or not new_messages:
        return None
    dummy_assistant: Message = {"role": "assistant", "content": "x"}
    try:
        base = renderer.render_ids(
            [dummy_assistant], tools=tools, add_generation_prompt=False
        )
        full = renderer.render_ids(
            [dummy_assistant, *new_messages], tools=tools, add_generation_prompt=True
        )
    except Exception:
        return None
    if full[: len(base)] != base:
        return None
    bridge_ids = list(full[len(base) :])
    if bridge_ids and previous_ids[-1] == bridge_ids[0]:
        bridge_ids = bridge_ids[1:]
    return previous_ids + bridge_ids
