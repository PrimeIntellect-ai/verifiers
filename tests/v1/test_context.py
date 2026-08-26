import httpx
from openai import BadRequestError

from verifiers.v1.clients.context import (
    compaction_threshold,
    model_context_window,
)
from verifiers.v1.configs.harness import CompactionConfig
from verifiers.v1.harnesses.bash.harness import BashHarnessConfig
from verifiers.v1.harnesses.bash.program import context_error
from verifiers.v1.harnesses.rlm.harness import RLMHarnessConfig


def test_model_context_window_reads_vllm_extension() -> None:
    payload = {
        "data": [
            {"id": "other", "max_model_len": 1},
            {"id": "target", "max_model_len": 32_768},
        ]
    }

    assert model_context_window(payload, "target") == 32_768


def test_model_context_window_accepts_common_provider_extensions() -> None:
    payload = {"data": [{"id": "target", "context_length": 128_000}]}

    assert model_context_window(payload, "target") == 128_000


def test_model_context_window_is_unknown_for_standard_model_card() -> None:
    payload = {
        "data": [
            {
                "id": "target",
                "object": "model",
                "created": 1,
                "owned_by": "provider",
            }
        ]
    }

    assert model_context_window(payload, "target") is None


def test_compaction_threshold_reserves_ten_percent() -> None:
    assert compaction_threshold(32_768) == 29_491


def test_compaction_is_disabled_by_default_for_both_harnesses() -> None:
    assert BashHarnessConfig().compaction is None
    assert RLMHarnessConfig().compaction is None


def test_compaction_config_has_shared_automatic_default() -> None:
    assert CompactionConfig().summarize_at_tokens is None


def test_threshold_is_learned_from_provider_error() -> None:
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://provider/v1/chat/completions"),
    )
    error = BadRequestError(
        "maximum context length is 32,768 tokens",
        response=response,
        body={"error": {"message": "maximum context length is 32,768 tokens"}},
    )

    assert context_error(error) == (True, 29_491)
