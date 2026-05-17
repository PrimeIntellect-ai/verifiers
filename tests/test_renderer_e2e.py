"""Tier-1 end-to-end tests for the renderer → RendererClient → Environment path.

Exercises the full TITO wire protocol with a scripted stand-in for vLLM:
messages → Renderer.render_ids → /inference/v1/generate → Renderer.parse_response → trajectory.

Asserts the four load-bearing invariants:

1. vLLM sees exactly what the client rendered (``returned_prompt_ids == sent_prompt_ids``).
2. Multi-turn prompts extend the previous turn's ``prompt_ids + completion_ids``
   bitwise (no re-tokenization across turns — the per-renderer
   ``bridge_to_next_turn`` method).
3. Sampler-side completion tokens appear verbatim in ``trajectory[step]['tokens']``;
   the trainer sees exactly what the sampler produced.
4. Completion IDs we scripted survive round-trip through the full rollout.

Parametrized over five model families so each renderer's render/parse paths
are exercised. Tokenizers come from the local HF cache; no network.
"""

import logging
from typing import Any

import pytest

import verifiers as vf
from datasets import Dataset
from renderers import create_renderer
from verifiers.clients.renderer_client import RendererClient, _to_renderer_message
from verifiers.types import Messages, State


def _renderer_has_extension_property(renderer) -> bool:
    """Runtime probe: does ``renderer.bridge_to_next_turn`` actually stitch?

    Simulates one turn of sampling and asks the renderer's bridge to extend
    it. Returns True iff the bridge returns tokens that extend prev (not
    None). This is the operational definition of "extension property" —
    whether the client will take the bridge path or the re-render fallback.
    """
    prev_prompt = renderer.render_ids(
        [{"role": "user", "content": "u"}], add_generation_prompt=True
    )
    prev_full = renderer.render_ids(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "x"}],
        add_generation_prompt=False,
    )
    n = 0
    while n < min(len(prev_prompt), len(prev_full)) and prev_prompt[n] == prev_full[n]:
        n += 1
    stop_ids = renderer.get_stop_token_ids()
    prev_completion = list(prev_full[n:])
    if stop_ids:
        prev_completion.append(stop_ids[0])

    result = renderer.bridge_to_next_turn(
        prev_prompt,
        prev_completion,
        [{"role": "user", "content": "next"}],
        tools=None,
    )
    return result is not None


# ── Model families under test ───────────────────────────────────────
# Each entry: (HF model id, renderer name or "auto"). One representative
# per distinct renderer code path (JSON tool calls, XML tool calls,
# arg_key/arg_value tokens, unique special tokens, Jinja fallback).
_MODEL_FAMILIES = [
    ("Qwen/Qwen3-0.6B", "auto"),
    ("Qwen/Qwen3.5-9B", "auto"),
    ("THUDM/GLM-4.5-Air", "auto"),
    ("MiniMaxAI/MiniMax-M2.5", "auto"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "default"),
]


_renderer_cache: dict[tuple[str, str], tuple[Any, Any]] = {}


def _load(model_name: str, renderer_name: str):
    key = (model_name, renderer_name)
    if key not in _renderer_cache:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        renderer = create_renderer(tokenizer, renderer=renderer_name)
        _renderer_cache[key] = (tokenizer, renderer)
    return _renderer_cache[key]


def pytest_generate_tests(metafunc):
    if "model_family" in metafunc.fixturenames:
        metafunc.parametrize(
            "model_family",
            _MODEL_FAMILIES,
            ids=[m for m, _ in _MODEL_FAMILIES],
        )


@pytest.fixture
def tokenizer_and_renderer(model_family):
    model_name, renderer_name = model_family
    return _load(model_name, renderer_name)


# ── Scripted vLLM stand-in ───────────────────────────────────────────


class ScriptedVLLM:
    """Fake ``AsyncOpenAI``-compatible client serving canned
    /inference/v1/generate responses (vllm 0.20 wire shape).
    """

    def __init__(self, completions: list[list[int]]):
        self._completions = list(completions)
        self.requests: list[dict[str, Any]] = []
        self.base_url = "http://fake-host:8000/v1"

    async def post(self, path: str, *, cast_to=None, body=None, options=None):
        assert path.endswith("/inference/v1/generate"), f"unexpected endpoint {path!r}"
        assert body is not None
        self.requests.append(body)

        assert self._completions, "ScriptedVLLM ran out of canned completions"
        completion_ids = self._completions.pop(0)

        return {
            "request_id": f"resp-{len(self.requests)}",
            "choices": [
                {
                    "index": 0,
                    "token_ids": list(completion_ids),
                    "logprobs": {
                        "content": [
                            {"token": f"token_id:{tid}", "logprob": -0.1}
                            for tid in completion_ids
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    async def close(self):
        pass


def _make_renderer_client(
    renderer, completions: list[list[int]]
) -> tuple[RendererClient, ScriptedVLLM]:
    """Build a RendererClient wired to a ScriptedVLLM."""
    scripted = ScriptedVLLM(completions)
    client = object.__new__(RendererClient)
    client.logger = logging.getLogger("test_renderer_e2e.RendererClient")
    client._config = None
    client._client = scripted
    client._renderer = renderer
    client._pool_size = 1
    return client, scripted


def _canonical_completion(
    renderer,
    history: list[dict],
    assistant_content: str,
) -> list[int]:
    """Return completion_ids for an assistant saying ``assistant_content``.

    Computed as ``render_ids(history + [asst]) − render_ids(history, add_gen=True)``,
    so the completion is guaranteed to round-trip: parsing it back and re-rendering
    the assistant message in history produces the same tokens. Avoids test noise
    from each renderer's idiosyncratic thinking prefill / history-injection
    conventions (Qwen3.5 prefills ``<think>\\n`` in the gen prompt; GLM-4.5 always
    injects ``<think></think>`` in assistant history; etc).

    Appends the renderer's first stop token so the completion also looks like
    what vLLM actually returns after a stop condition fires.
    """
    with_asst = list(history) + [{"role": "assistant", "content": assistant_content}]
    prompt_ids = renderer.render_ids(history, add_generation_prompt=True)
    full_ids = renderer.render_ids(with_asst, add_generation_prompt=False)

    # Longest common prefix between gen-prompt form and history form.
    n = 0
    max_n = min(len(prompt_ids), len(full_ids))
    while n < max_n and prompt_ids[n] == full_ids[n]:
        n += 1
    completion = list(full_ids[n:])

    stop_ids = renderer.get_stop_token_ids()
    if stop_ids:
        completion.append(stop_ids[0])
    return completion


# ── Test 1: Reverse-Text (single-turn, no tool calls) ───────────────


@pytest.mark.asyncio
async def test_reverse_text_single_turn(tokenizer_and_renderer, model_family):
    """Single-turn TITO: prompt → one completion → terminate.

    Asserts invariants 1, 3, and parse correctness.
    """
    tokenizer, renderer = tokenizer_and_renderer
    model_name, _ = model_family

    system_text = "Reverse the text. Wrap in <reversed_text> tags."
    user_text = "Reverse this: hello"
    expected_output = "<reversed_text>olleh</reversed_text>"
    input_messages_tmp = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    completion_ids = _canonical_completion(
        renderer, input_messages_tmp, expected_output
    )

    client, scripted = _make_renderer_client(renderer, [completion_ids])

    input_messages = input_messages_tmp
    dataset = Dataset.from_dict(
        {
            "prompt": [input_messages],
            "answer": ["olleh"],
            "example_id": [0],
        }
    )
    parser = vf.XMLParser(["reversed_text"], answer_field="reversed_text")

    def lcs(completion, answer, **kwargs):
        from difflib import SequenceMatcher

        got = parser.parse_answer(completion) or ""
        return SequenceMatcher(None, got, answer).ratio()

    env = vf.SingleTurnEnv(
        client=client,
        model=model_name,
        dataset=dataset,
        parser=parser,
        rubric=vf.Rubric(funcs=[lcs], weights=[1.0]),
    )

    state = await env.rollout(
        input={
            "prompt": input_messages,
            "answer": "olleh",
            "example_id": 0,
            "info": {"env_id": "reverse-text"},
        },
        client=client,
        model=model_name,
        sampling_args={"temperature": 0.0},
    )

    # --- Invariant 1: vLLM saw exactly what the renderer produced --------
    assert len(scripted.requests) == 1
    sent_prompt_ids = scripted.requests[0]["token_ids"]

    expected_prompt_ids = renderer.render_ids(
        input_messages,
        add_generation_prompt=True,
    )
    assert sent_prompt_ids == expected_prompt_ids, (
        f"Client-rendered prompt diverged from what vLLM would have received for {model_name}"
    )

    # --- Invariant 3: trajectory tokens match sampler output --------------
    trajectory = state["trajectory"]
    assert len(trajectory) == 1
    step = trajectory[0]
    assert step["tokens"] is not None
    assert step["tokens"]["completion_ids"] == completion_ids
    assert step["tokens"]["prompt_ids"] == sent_prompt_ids

    # --- parse_response extracted the content we scripted ----------------
    # Use the renderer directly to avoid relying on the XMLParser for
    # default-path models that might strip tokens differently.
    parsed = renderer.parse_response(completion_ids)
    assert "olleh" in parsed.content or "reversed_text" in parsed.content


# ── Test 2: AlphabetSort (multi-turn, user/assistant back-and-forth) ─


@pytest.mark.asyncio
async def test_alphabet_sort_multi_turn(tokenizer_and_renderer, model_family):
    """Three-turn rollout: user → asst → user follow-up → asst → user follow-up → asst.

    Asserts invariants 1, 2, 3. Invariant 2 (prefix stability across turns)
    is the critical one — it validates each renderer's ``bridge_to_next_turn``.
    """
    tokenizer, renderer = tokenizer_and_renderer
    model_name, _ = model_family

    turn1_prompt = "Sort alphabetically: Charlie, Alice, Bob"
    turn1_answer = "<alphabetical_sorted>\nAlice\nBob\nCharlie\n</alphabetical_sorted>"
    turn2_followup = "Now include Dave"
    turn2_answer = "<combined_alphabetical_sorted>\nAlice\nBob\nCharlie\nDave // new name!\n</combined_alphabetical_sorted>"
    turn3_followup = "Now include Eve"
    turn3_answer = "<combined_alphabetical_sorted>\nAlice\nBob\nCharlie\nDave\nEve // new name!\n</combined_alphabetical_sorted>"

    # Build canonical completions by simulating what the renderer would have
    # produced at each turn. Each completion is guaranteed round-trip stable
    # so re-rendering history matches vLLM's sampler-side view.
    turn1_history = [{"role": "user", "content": turn1_prompt}]
    turn1_completion = _canonical_completion(renderer, turn1_history, turn1_answer)

    turn2_history = turn1_history + [
        {"role": "assistant", "content": turn1_answer},
        {"role": "user", "content": turn2_followup},
    ]
    turn2_completion = _canonical_completion(renderer, turn2_history, turn2_answer)

    turn3_history = turn2_history + [
        {"role": "assistant", "content": turn2_answer},
        {"role": "user", "content": turn3_followup},
    ]
    turn3_completion = _canonical_completion(renderer, turn3_history, turn3_answer)

    completions = [turn1_completion, turn2_completion, turn3_completion]

    client, scripted = _make_renderer_client(renderer, completions)

    followups = [turn2_followup, turn3_followup]

    class SortingEnv(vf.MultiTurnEnv):
        @vf.stop
        async def after_three_turns(self, state: State) -> bool:
            return len(state["trajectory"]) >= 3

        async def env_response(
            self, messages: Messages, state: State, **kwargs
        ) -> Messages:
            assistant_count = sum(
                1 for m in messages if getattr(m, "role", None) == "assistant"
            )
            idx = assistant_count - 1
            return [vf.UserMessage(content=followups[idx])]

    env = SortingEnv(
        client=client,
        model=model_name,
        max_turns=3,
        dataset=Dataset.from_dict(
            {"prompt": [[{"role": "user", "content": turn1_prompt}]], "example_id": [0]}
        ),
        parser=vf.Parser(),
        rubric=vf.Rubric(),
    )

    state = await env.rollout(
        input={
            "prompt": [{"role": "user", "content": turn1_prompt}],
            "example_id": 0,
            "info": {"env_id": "alphabet-sort"},
        },
        client=client,
        model=model_name,
        sampling_args={"temperature": 0.0},
    )

    trajectory = state["trajectory"]
    assert len(trajectory) == 3, f"expected 3 turns, got {len(trajectory)}"
    assert len(scripted.requests) == 3

    # --- Invariant 3 (universal): sampler ids preserved in trajectory ----
    for turn_idx, step in enumerate(trajectory):
        assert step["tokens"] is not None
        assert step["tokens"]["completion_ids"] == completions[turn_idx], (
            f"turn {turn_idx}: trainer's completion_ids diverged from sampler's for {model_name}"
        )

    has_extension = _renderer_has_extension_property(renderer)

    # --- Invariant 1 (non-extension): sent prompt == naive re-render of
    # that turn's reconstructed history. When the bridge trick doesn't apply,
    # the client falls back to ``render_ids(messages, add_generation_prompt=True)``;
    # this check guards that fallback path against silent divergence.
    if not has_extension:
        for turn_idx, step in enumerate(trajectory):
            step_prompt_dicts = [_to_renderer_message(m) for m in step["prompt"]]
            expected = renderer.render_ids(
                step_prompt_dicts, add_generation_prompt=True
            )
            assert scripted.requests[turn_idx]["token_ids"] == expected, (
                f"turn {turn_idx}: client sent prompt_ids that don't match "
                f"render_ids(step['prompt']) for {model_name}. "
                f"Re-render path diverged."
            )
            assert step["tokens"]["prompt_ids"] == expected

    # --- Invariant 2 (extension): bitwise extension across turns. Only
    # holds for renderers whose history form of an assistant message contains
    # the generation stop token (the anchor ``bridge_to_next_turn`` trims to).
    # When this holds, the client stitches the previous turn's exact tokens
    # (including stop) with the renderer's hand-emitted extension — which
    # necessarily diverges from a naive full re-render, so invariant 1 is
    # skipped.
    if has_extension:
        for turn_idx in range(1, 3):
            prev_prompt = trajectory[turn_idx - 1]["tokens"]["prompt_ids"]
            prev_completion = trajectory[turn_idx - 1]["tokens"]["completion_ids"]
            this_prompt = trajectory[turn_idx]["tokens"]["prompt_ids"]

            prev_combined = list(prev_prompt) + list(prev_completion)
            assert this_prompt[: len(prev_combined)] == prev_combined, (
                f"turn {turn_idx}: prompt_ids is NOT a bitwise extension of "
                f"turn {turn_idx - 1}'s prompt+completion for {model_name}. "
                f"bridge_to_next_turn should have stitched, but didn't."
            )
            assert len(this_prompt) > len(prev_combined), (
                f"turn {turn_idx}: prompt didn't grow past previous combined for {model_name}"
            )
