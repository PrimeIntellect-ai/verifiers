"""Barrage test: build_supervised_sample and build_trajectory_step.

Runs against every (model, renderer) pair.
"""

import pytest

from renderers import build_supervised_sample, build_trajectory_step


def _expected(tokenizer, messages, **kwargs):
    result = tokenizer.apply_chat_template(messages, return_dict=False, **kwargs)
    if isinstance(result, dict):
        return list(result["input_ids"])
    return list(result)


def test_build_supervised_sample_ids_match(model_name, tokenizer, renderer):
    """Token IDs must match apply_chat_template."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    ids, mask = build_supervised_sample(
        renderer, msgs, role_to_mask=lambda m: m["role"] == "assistant"
    )
    assert ids == _expected(tokenizer, msgs)


def test_build_supervised_sample_has_trainable_tokens(model_name, tokenizer, renderer):
    """At least some tokens should be marked for training."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    ids, mask = build_supervised_sample(
        renderer, msgs, role_to_mask=lambda m: m["role"] == "assistant"
    )
    assert sum(mask) > 0
    assert len(mask) == len(ids)


def test_build_trajectory_step_reconstructs_full(model_name, tokenizer, renderer):
    """prompt_ids + completion_ids must equal the full rendered sequence."""
    prompt = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    completion = [{"role": "assistant", "content": "Hello!"}]
    step = build_trajectory_step(renderer, prompt, completion)
    full_ids = renderer.render_ids(prompt + completion)
    assert step["prompt_ids"] + step["completion_ids"] == full_ids


def test_build_trajectory_step_masks(model_name, tokenizer, renderer):
    """Prompt mask all False, completion mask all True."""
    prompt = [{"role": "user", "content": "Hi"}]
    completion = [{"role": "assistant", "content": "Hello!"}]
    step = build_trajectory_step(renderer, prompt, completion)
    assert all(m is False for m in step["prompt_mask"])
    assert all(m is True for m in step["completion_mask"])
    assert len(step["completion_logprobs"]) == len(step["completion_ids"])
