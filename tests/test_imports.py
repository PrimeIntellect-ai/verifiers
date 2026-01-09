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
        """Check if an AttributeError indicates missing optional dependencies."""
        optional_dependency_patterns = [
            "install as `verifiers[all]`",
            "install as `verifiers[math]`",
            "install as `verifiers[",  # catches any [extra] pattern
        ]
        return any(pattern in error_msg for pattern in optional_dependency_patterns)

    def test_all_items_are_importable(self):
        """Test that all items in __all__ can actually be imported."""
        for item_name in verifiers.__all__:
            try:
                # This should not raise AttributeError
                item = getattr(verifiers, item_name)
                assert item is not None, f"{item_name} in __all__ but is None"
            except AttributeError as e:
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
            except AttributeError as e:
                # This is expected for lazy imports when dependencies are missing
                if self._is_optional_dependency_error(str(e)):
                    # This is the expected error for missing optional dependencies
                    pass
                else:
                    # This is an unexpected AttributeError, re-raise it
                    raise

    def test_dir_includes_public_api(self):
        """Test that dir(verifiers) exposes the public API."""
        directory = dir(verifiers)
        for name in verifiers.__all__:
            assert name in directory, f"{name} missing from dir(verifiers)"

    def test_star_import_matches_all(self):
        """Test that star import exposes __all__ names."""
        namespace: dict[str, object] = {}
        try:
            exec("from verifiers import *", namespace)
        except AttributeError as e:
            if not self._is_optional_dependency_error(str(e)):
                raise

        for name in verifiers.__all__:
            if name in namespace:
                continue
            # If star import skipped a name, accessing it should only fail
            # for missing optional dependencies.
            try:
                _ = getattr(verifiers, name)
            except AttributeError as e:
                assert self._is_optional_dependency_error(str(e)), (
                    f"{name} missing from star import without optional-deps error"
                )
