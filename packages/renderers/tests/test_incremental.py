"""Unit tests for ``trim_to_turn_close`` and the bridge contract invariants
that every renderer's ``bridge_to_next_turn`` must uphold.

Cross-renderer parity is validated in test_render_ids.py and test_roundtrip.py;
this file exercises the shared primitive + protocol-level guarantees with a
fake renderer so we get fast, deterministic coverage of the tricky corners
(truncation opt-in, assistant-in-extension rejection, empty inputs).
"""

from renderers.base import (
    ParsedResponse,
    RenderedConversation,
    reject_assistant_in_extension,
    trim_to_turn_close,
)


def test_rendered_conversation_keeps_exact_token_tape():
    parsed = ParsedResponse(content="done")
    conv = RenderedConversation(
        prompt_ids=[1, 2], messages=[{"role": "user", "content": "hi"}]
    )

    next_conv = conv.with_completion(
        [3, 99], completion_logprobs=[-0.1, -0.2], parsed_completion=parsed
    )

    assert next_conv.token_ids == [1, 2, 3, 99]
    assert next_conv.completion_logprobs == [-0.1, -0.2]
    assert next_conv.parsed_completion is parsed
    assert conv.completion_ids == []


# ---------------------------------------------------------------------------
# trim_to_turn_close
# ---------------------------------------------------------------------------


def test_trim_to_turn_close_trims_to_last_close_in_completion():
    # prev = [1, 2] + [3, 99, 30] (stop token 99). Trim to the 99 boundary,
    # drop the [30] that the model sampled after the stop token.
    result = trim_to_turn_close([1, 2], [3, 99, 30], {99})
    assert result == [1, 2, 3, 99]


def test_trim_to_turn_close_ignores_close_in_prompt():
    # A stop-token id that happens to appear in prev_prompt (as structural
    # template scaffolding) must not be treated as a turn boundary.
    result = trim_to_turn_close([99, 1], [3, 4, 5], {99})
    assert result is None


def test_trim_to_turn_close_synthesises_when_truncated():
    # Truncation: no stop token in completion. With synthesize_close=99,
    # append the synthetic close and return prev + [99].
    result = trim_to_turn_close([1, 2], [3, 4, 5], {99}, synthesize_close=99)
    assert result == [1, 2, 3, 4, 5, 99]


def test_trim_to_turn_close_returns_none_on_truncation_without_synth():
    # Truncation without synth opt-in → caller falls back to fresh render.
    result = trim_to_turn_close([1, 2], [3, 4, 5], {99})
    assert result is None


def test_trim_to_turn_close_accepts_multiple_close_tokens():
    # Multiple close tokens: pick the LAST one that appears in completion.
    result = trim_to_turn_close([1], [3, 50, 4, 99, 30], {50, 99})
    assert result == [1, 3, 50, 4, 99]


# ---------------------------------------------------------------------------
# reject_assistant_in_extension
# ---------------------------------------------------------------------------


def test_reject_assistant_in_extension_true_when_assistant_present():
    assert reject_assistant_in_extension(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
    )


def test_reject_assistant_in_extension_false_for_tool_user_only():
    assert not reject_assistant_in_extension(
        [{"role": "tool", "content": "result"}, {"role": "user", "content": "next"}]
    )


# ---------------------------------------------------------------------------
# Contract tests against a minimal fake renderer
# ---------------------------------------------------------------------------


class _FakeRenderer:
    """Minimal Renderer whose bridge exercises the contract:

    - Extension token is a single sentinel ID 42.
    - ``<|im_end|>`` ≡ 99 is the canonical close.
    """

    synthesize_close_on_truncation = True

    def __init__(self):
        self._im_end = 99

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        raise NotImplementedError

    def parse_response(self, token_ids):
        return ParsedResponse(content="")

    def get_stop_token_ids(self):
        return [self._im_end]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
    ):
        if not previous_prompt_ids or not new_messages:
            return None
        if reject_assistant_in_extension(new_messages):
            return None
        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._im_end},
            synthesize_close=(
                self._im_end if self.synthesize_close_on_truncation else None
            ),
        )
        if previous_ids is None:
            return None
        return previous_ids + [42]


def test_fake_bridge_extends_verbatim_on_clean_stop():
    renderer = _FakeRenderer()
    prev_prompt = [1, 2]
    prev_completion = [3, 99]
    result = renderer.bridge_to_next_turn(
        prev_prompt, prev_completion, [{"role": "user", "content": "next"}]
    )
    assert result == [1, 2, 3, 99, 42]
    assert (
        result[: len(prev_prompt) + len(prev_completion)]
        == prev_prompt + prev_completion
    )


def test_fake_bridge_synthesises_on_truncation():
    renderer = _FakeRenderer()
    result = renderer.bridge_to_next_turn(
        [1, 2], [3, 4, 5], [{"role": "user", "content": "next"}]
    )
    # Truncated prev; synth-close appends 99 then extension 42.
    assert result == [1, 2, 3, 4, 5, 99, 42]


def test_fake_bridge_rejects_assistant_in_extension():
    renderer = _FakeRenderer()
    result = renderer.bridge_to_next_turn(
        [1], [99], [{"role": "assistant", "content": "x"}]
    )
    assert result is None


def test_fake_bridge_returns_none_without_synth_opt_in():
    renderer = _FakeRenderer()
    renderer.synthesize_close_on_truncation = False
    result = renderer.bridge_to_next_turn(
        [1], [3, 4, 5], [{"role": "user", "content": "next"}]
    )
    assert result is None


def test_fake_bridge_rejects_empty_inputs():
    renderer = _FakeRenderer()
    assert (
        renderer.bridge_to_next_turn([], [99], [{"role": "user", "content": "x"}])
        is None
    )
    assert renderer.bridge_to_next_turn([1], [99], []) is None
