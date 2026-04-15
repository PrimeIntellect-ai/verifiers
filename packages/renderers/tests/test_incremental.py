from renderers.base import (
    ParsedResponse,
    RenderedConversation,
    build_incremental_prompt_ids,
)


class _BridgeRenderer:
    supports_tools = True

    def __init__(self, bridge_base=None, bridge_full=None):
        self.bridge_base = bridge_base or [10, 99, 30]
        self.bridge_full = bridge_full or [10, 99, 30, 40, 50]
        self.calls = []

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        self.calls.append((messages, tools, add_generation_prompt))
        if len(messages) == 1 and add_generation_prompt is False:
            return list(self.bridge_base)
        if len(messages) > 1 and add_generation_prompt is True:
            return list(self.bridge_full)
        raise AssertionError((messages, tools, add_generation_prompt))

    def parse_response(self, token_ids):
        return ParsedResponse(content="")

    def get_stop_token_ids(self):
        return [99]


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


def test_build_incremental_prompt_ids_keeps_bridge_after_dummy_boundary():
    renderer = _BridgeRenderer(
        bridge_base=[10, 99, 30], bridge_full=[10, 99, 30, 40, 50]
    )

    result = build_incremental_prompt_ids(
        renderer,
        previous_prompt_ids=[1, 2],
        previous_completion_ids=[3, 99],
        new_messages=[{"role": "tool", "content": "result", "tool_call_id": "call_0"}],
    )

    assert result == [1, 2, 3, 99, 30, 40, 50]


def test_build_incremental_prompt_ids_skips_duplicate_stop_token():
    renderer = _BridgeRenderer(bridge_base=[10, 99], bridge_full=[10, 99, 99, 40])

    result = build_incremental_prompt_ids(
        renderer,
        previous_prompt_ids=[1],
        previous_completion_ids=[99],
        new_messages=[{"role": "user", "content": "next"}],
    )

    assert result == [1, 99, 40]


def test_build_incremental_prompt_ids_trims_post_stop_scaffolding():
    renderer = _BridgeRenderer(bridge_base=[10, 99, 30], bridge_full=[10, 99, 30, 40])

    result = build_incremental_prompt_ids(
        renderer,
        previous_prompt_ids=[1],
        previous_completion_ids=[3, 99, 30],
        new_messages=[{"role": "tool", "content": "result", "tool_call_id": "call_0"}],
    )

    assert result == [1, 3, 99, 30, 40]


def test_build_incremental_prompt_ids_falls_back_when_stop_boundary_missing():
    renderer = _BridgeRenderer(bridge_base=[10, 11], bridge_full=[10, 11, 40])

    assert (
        build_incremental_prompt_ids(
            renderer,
            previous_prompt_ids=[1],
            previous_completion_ids=[99],
            new_messages=[{"role": "user", "content": "next"}],
        )
        is None
    )
