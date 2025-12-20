"""Tests for verifiers.utils.error_utils.ErrorChain."""

import verifiers as vf
from verifiers.utils.error_utils import ErrorChain, get_vf_error_chain


class TestErrorChain:
    """Test cases for ErrorChain class."""

    def test_init_error_chain(self):
        """Test ErrorChain initialization."""
        outer = ValueError("outer")
        inner = KeyError("root")
        outer.__cause__ = inner

        error_chain = ErrorChain(outer)
        assert error_chain.root_error is outer
        assert len(error_chain.chain) == 2
        assert error_chain.chain[0] is outer
        assert error_chain.chain[1] is inner

    def test_init_vf_error_chain(self):
        """Test ErrorChain initialization with vf errors."""
        outer = vf.ToolCallError()
        inner = ValueError()
        outer.__cause__ = inner

        error_chain = ErrorChain(outer, get_vf_error_chain)
        assert error_chain.root_error is outer
        assert len(error_chain.chain) == 1
        assert error_chain.chain[0] is outer

    def test_init_custom_error_chain(self):
        """Test ErrorChain with custom build_error_chain function."""

        def custom_builder(error):
            return [error]  # Always return just the root error

        root = KeyError("root")
        outer = ValueError("outer")
        outer.__cause__ = root

        chain = ErrorChain(outer, build_error_chain=custom_builder)
        assert len(chain.chain) == 1
        assert chain.chain[0] is outer

    def test_hash_same_types(self):
        """Test that ErrorChains with same error types have same hash."""
        error1 = ValueError("first")
        error1.__cause__ = KeyError("cause1")

        error2 = ValueError("second")
        error2.__cause__ = KeyError("cause2")

        chain1 = ErrorChain(error1)
        chain2 = ErrorChain(error2)
        assert hash(chain1) == hash(chain2)

    def test_hash_different_types(self):
        """Test that ErrorChains with different error types have different hash."""
        error1 = ValueError("first")
        error2 = TypeError("second")

        chain1 = ErrorChain(error1)
        chain2 = ErrorChain(error2)
        assert hash(chain1) != hash(chain2)

    def test_eq_same_types(self):
        """Test equality for ErrorChains with same error types."""
        error1 = ValueError("first")
        error1.__cause__ = KeyError("cause1")

        error2 = ValueError("second")
        error2.__cause__ = KeyError("cause2")

        chain1 = ErrorChain(error1)
        chain2 = ErrorChain(error2)
        assert chain1 == chain2

    def test_eq_different_types(self):
        """Test inequality for ErrorChains with different error types."""
        error1 = ValueError("first")
        error2 = TypeError("second")

        chain1 = ErrorChain(error1)
        chain2 = ErrorChain(error2)
        assert chain1 != chain2

    def test_eq_different_chain_length(self):
        """Test inequality for ErrorChains with different chain lengths."""
        error1 = ValueError("single")

        error2 = ValueError("outer")
        error2.__cause__ = KeyError("cause")

        chain1 = ErrorChain(error1)
        chain2 = ErrorChain(error2)
        assert chain1 != chain2

    def test_eq_non_error_chain(self):
        """Test that comparing ErrorChain to non-ErrorChain returns NotImplemented."""
        error = ValueError("test")
        chain = ErrorChain(error)
        assert chain.__eq__("not an ErrorChain") is NotImplemented

    def test_contains_error_instance(self):
        """Test __contains__ checks if an error instance matches any type in the chain."""
        root = KeyError("root")
        outer = ValueError("outer")
        outer.__cause__ = root

        chain = ErrorChain(outer)
        # Check that instances of same types are found in chain
        assert KeyError in chain
        assert ValueError in chain
        assert TypeError not in chain

    def test_contains_subclass(self):
        """Test __contains__ works with subclasses."""
        outer = vf.ToolCallError()
        inner = vf.SandboxError()
        outer.__cause__ = inner

        chain = ErrorChain(outer)
        # ToolCallError is directly contained, in chain
        assert vf.ToolCallError in chain
        # ToolParseError is a sibling class, not in chain
        assert vf.ToolParseError not in chain
        # SandboxError is directly contained, in chain
        assert vf.SandboxError in chain
        # InfraError is parent of SandboxError, in chain
        assert vf.InfraError in chain
        # Error is parent of all vf errors, in chain
        assert vf.Error in chain

    def test_hashable_for_counter(self):
        """Test that ErrorChain can be used as Counter key."""
        from collections import Counter

        error1 = ValueError("first")
        error2 = ValueError("second")
        error3 = TypeError("third")

        chains = [ErrorChain(error1), ErrorChain(error2), ErrorChain(error3)]
        counter = Counter(chains)

        # error1 and error2 have same type, should be counted together
        assert counter[ErrorChain(ValueError("any"))] == 2
        assert counter[ErrorChain(TypeError("any"))] == 1
