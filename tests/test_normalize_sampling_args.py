"""Tests for _normalize_sampling_args in verifiers.envs.environment."""

from verifiers.envs.environment import _normalize_sampling_args


# ---------------------------------------------------------------------------
# Completions API: logprobs / top_logprobs translation
# ---------------------------------------------------------------------------


class TestCompletionLogprobs:
    """Completions API takes logprobs as an integer and ignores top_logprobs."""

    def test_bool_true_with_top_logprobs(self):
        """logprobs=True + top_logprobs=K → logprobs=K, no top_logprobs."""
        result = _normalize_sampling_args(
            {"logprobs": True, "top_logprobs": 5}, "completion"
        )
        assert result["logprobs"] == 5
        assert isinstance(result["logprobs"], int)
        assert not isinstance(result["logprobs"], bool)
        assert "top_logprobs" not in result

    def test_bool_true_without_top_logprobs(self):
        """logprobs=True alone → logprobs=1."""
        result = _normalize_sampling_args({"logprobs": True}, "completion")
        assert result["logprobs"] == 1
        assert isinstance(result["logprobs"], int)
        assert not isinstance(result["logprobs"], bool)

    def test_bool_false_removed(self):
        """logprobs=False → logprobs key removed entirely."""
        result = _normalize_sampling_args({"logprobs": False}, "completion")
        assert "logprobs" not in result

    def test_bool_false_with_top_logprobs(self):
        """logprobs=False + top_logprobs=K → both removed (False wins)."""
        result = _normalize_sampling_args(
            {"logprobs": False, "top_logprobs": 5}, "completion"
        )
        assert "logprobs" not in result
        assert "top_logprobs" not in result

    def test_top_logprobs_only_no_logprobs(self):
        """top_logprobs=K without logprobs → logprobs=K."""
        result = _normalize_sampling_args({"top_logprobs": 5}, "completion")
        assert result["logprobs"] == 5
        assert "top_logprobs" not in result

    def test_int_logprobs_preserved(self):
        """logprobs already an integer → kept as-is."""
        result = _normalize_sampling_args({"logprobs": 3}, "completion")
        assert result["logprobs"] == 3

    def test_int_logprobs_with_top_logprobs(self):
        """logprobs=int + top_logprobs present → logprobs kept, top_logprobs stripped."""
        result = _normalize_sampling_args(
            {"logprobs": 3, "top_logprobs": 5}, "completion"
        )
        assert result["logprobs"] == 3
        assert "top_logprobs" not in result

    def test_original_dict_top_logprobs_not_popped(self):
        """top_logprobs must NOT be popped from the input dict (used downstream)."""
        original = {"logprobs": True, "top_logprobs": 5}
        _normalize_sampling_args(original, "completion")
        assert "top_logprobs" in original


# ---------------------------------------------------------------------------
# Chat completions API: logprobs / top_logprobs should pass through
# ---------------------------------------------------------------------------


class TestChatLogprobs:
    """Chat completions API: logprobs (bool) + top_logprobs (int) pass through."""

    def test_bool_true_with_top_logprobs_passthrough(self):
        result = _normalize_sampling_args({"logprobs": True, "top_logprobs": 5}, "chat")
        assert result["logprobs"] is True
        assert result["top_logprobs"] == 5

    def test_bool_true_only(self):
        result = _normalize_sampling_args({"logprobs": True}, "chat")
        assert result["logprobs"] is True

    def test_bool_false_kept(self):
        """logprobs=False is a valid boolean for chat — kept in output."""
        result = _normalize_sampling_args({"logprobs": False}, "chat")
        assert result["logprobs"] is False


# ---------------------------------------------------------------------------
# max_tokens normalization (pre-existing behaviour, verify not broken)
# ---------------------------------------------------------------------------


class TestMaxTokens:
    def test_chat_renames_max_tokens(self):
        result = _normalize_sampling_args({"max_tokens": 100}, "chat")
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 100

    def test_completion_keeps_max_tokens(self):
        result = _normalize_sampling_args({"max_tokens": 100}, "completion")
        assert result["max_tokens"] == 100
        assert "max_completion_tokens" not in result

    def test_none_max_tokens_dropped(self):
        result = _normalize_sampling_args({"max_tokens": None}, "chat")
        assert "max_tokens" not in result
        assert "max_completion_tokens" not in result


# ---------------------------------------------------------------------------
# None-valued entries are always dropped
# ---------------------------------------------------------------------------


class TestNoneFiltering:
    def test_none_values_dropped(self):
        result = _normalize_sampling_args({"temperature": 0.7, "top_p": None}, "chat")
        assert "top_p" not in result
        assert result["temperature"] == 0.7
