"""The verifiers package root.

The v1 stack lives in `verifiers.v1`; the classic (v0) stack lives in
`verifiers.legacy`. The v0 modules keep answering at their pre-move paths —
`verifiers.envs`, `verifiers.types`, `import verifiers as vf`, ... — through the
alias finder below, so importing this package root stays side-effect free (the
v0 surface only loads when something touches it).
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING

try:
    __version__ = _version("verifiers")
except PackageNotFoundError:  # source tree without install metadata
    __version__ = "0.0.0+unknown"

# The v0 modules that moved under `verifiers.legacy` and stay importable at
# their old top-level paths.
_LEGACY_MODULES = frozenset(
    {
        "clients",
        "decorators",
        "envs",
        "errors",
        "gepa",
        "parsers",
        "rubrics",
        "scripts",
        "serve",
        "types",
        "utils",
    }
)


class _LegacyAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve `verifiers.<v0 module>` to `verifiers.legacy.<module>`.

    The loaded module is the legacy module object itself (one instance, aliased
    in `sys.modules` under both names), so `verifiers.types.State is
    verifiers.legacy.types.State` and pre-move imports — including the absolute
    ones inside the moved code — keep working. Installed at the *front* of
    `sys.meta_path` so the path finder never re-executes an aliased package's
    children under the old name.
    """

    _prefix = "verifiers."
    _legacy_prefix = "verifiers.legacy."

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._prefix) or fullname.startswith(
            self._legacy_prefix
        ):
            return None
        tail = fullname.removeprefix(self._prefix)
        if tail.split(".", 1)[0] not in _LEGACY_MODULES:
            return None
        try:
            legacy_spec = importlib.util.find_spec(self._legacy_prefix + tail)
        except ModuleNotFoundError:
            legacy_spec = None
        if legacy_spec is None:  # unknown submodule: fail like any missing import
            return None
        spec = importlib.machinery.ModuleSpec(
            fullname,
            self,
            origin=legacy_spec.origin,
            is_package=legacy_spec.submodule_search_locations is not None,
        )
        return spec

    def create_module(self, spec):
        # The already-initialized legacy module keeps its own __name__/__spec__
        # (`module_from_spec` only fills attributes that are missing).
        return importlib.import_module(
            self._legacy_prefix + spec.name.removeprefix(self._prefix)
        )

    def exec_module(self, module):
        pass  # imported (or in progress) under its legacy name already


if not any(isinstance(finder, _LegacyAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _LegacyAliasFinder())


def __getattr__(name: str):
    """`vf.XMLParser`-style access forwards to the v0 surface in
    `verifiers.legacy` (loaded on first touch). `__all__` forwards too, so
    `from verifiers import *` keeps its pre-move meaning."""
    if name.startswith("_") and name != "__all__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    legacy = importlib.import_module("verifiers.legacy")
    return getattr(legacy, name)


if TYPE_CHECKING:  # static view of the forwarded surface
    from verifiers.legacy import *  # noqa: F403
