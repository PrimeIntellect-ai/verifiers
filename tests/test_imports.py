import verifiers


class TestImports:
    """Test that all public API imports work correctly.
    This was inspired by issue #349.

    Timeline:
    - Aug 26, 2025: v0.1.3 released to PyPI (without StatefulToolEnv in __init__.py)
    - Sept 11, 2025: PR #306 fixed the missing import in __init__.py
    - No new PyPI release made with the fix
    - Impact: Users installing verifiers==0.1.3 from PyPI cannot import StatefulToolEnv even though the class exists in their installation.

    This test ensures that all items in verifiers.__all__ can be imported,
    catching issues like the one above before they reach users.
    """

    @staticmethod
    def _is_optional_dependency_error(error_msg: str) -> bool:
        """Check if an error indicates missing optional dependencies."""
        optional_dependency_patterns = [
            "install as `verifiers-rl`",
            "install as `verifiers[all]`",
            "install as `verifiers[math]`",
            "install as `verifiers[",  # catches any [extra] pattern
            "verifiers[browser]",  # browser extra
            "verifiers[ta]",  # textarena extra
            "verifiers[rg]",  # reasoning-gym extra
        ]
        return any(pattern in error_msg for pattern in optional_dependency_patterns)

    def test_all_items_are_importable(self):
        """Test that all items in __all__ can actually be imported."""
        for item_name in verifiers.__all__:
            try:
                # This should not raise AttributeError
                item = getattr(verifiers, item_name)
                assert item is not None, f"{item_name} in __all__ but is None"
            except (AttributeError, ImportError) as e:
                # Check if this is an expected optional dependency error
                if self._is_optional_dependency_error(str(e)):
                    # This is expected for items requiring optional dependencies
                    continue
                else:
                    # For non-optional items, this should not happen
                    raise AssertionError(
                        f"Required item '{item_name}' cannot be imported: {e}"
                    )

    def test_lazy_imports_work(self):
        """Test that lazy imports work correctly."""
        # Dynamically detect lazy imports by checking verifiers module
        lazy_imports = getattr(verifiers, "_LAZY_IMPORTS", {})

        for name in lazy_imports.keys():
            assert name in verifiers.__all__, f"Lazy import {name} not in __all__"

            # Try to access the lazy import - this might fail due to missing dependencies
            # but should not fail due to import errors in our code
            try:
                item = getattr(verifiers, name)
                assert item is not None
            except (AttributeError, ImportError) as e:
                # This is expected for lazy imports when dependencies are missing
                if self._is_optional_dependency_error(str(e)):
                    # This is the expected error for missing optional dependencies
                    pass
                else:
                    # This is an unexpected error, re-raise it
                    raise
