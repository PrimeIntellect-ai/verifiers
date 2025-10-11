"""Tests for sampling argument utilities and merging logic."""

from verifiers.utils.sampling_utils import SamplingArgs, merge_sampling_args


class TestSamplingArgs:
    """Test the SamplingArgs dataclass and serialization."""

    def test_to_dict_empty(self):
        """Test that empty SamplingArgs converts to empty dict."""
        args = SamplingArgs()

        result = args.to_dict()

        assert result == {}

    def test_to_dict_separates_extra_body(self):
        """Test that extra_body fields are correctly separated from standard fields."""
        args = SamplingArgs(temperature=0.7, top_k=40, min_p=0.05)

        result = args.to_dict()

        assert result["temperature"] == 0.7
        assert result["extra_body"] == {"top_k": 40, "min_p": 0.05}
        assert "_EXTRA_BODY_FIELDS" not in result


class TestMergeSamplingArgs:
    """Test sampling argument merging behavior."""

    def test_merge_without_overrides(self):
        """Test that args are converted correctly when no overrides provided."""
        args = SamplingArgs(max_tokens=128)

        merged = merge_sampling_args(args)

        assert merged == {"max_tokens": 128}

    def test_overrides_take_precedence(self):
        """Test that override values replace base argument values."""
        args = SamplingArgs(temperature=0.4)
        overrides = {"temperature": 0.9, "max_tokens": 256}

        merged = merge_sampling_args(args, overrides)

        assert merged["temperature"] == 0.9
        assert merged["max_tokens"] == 256

    def test_extra_body_merges_correctly(self):
        """Test that extra_body fields from both sources are merged."""
        args = SamplingArgs(top_k=20)
        overrides = {"extra_body": {"top_k": 5, "min_p": 0.03}}

        merged = merge_sampling_args(args, overrides)

        assert merged["extra_body"]["top_k"] == 5
        assert merged["extra_body"]["min_p"] == 0.03
