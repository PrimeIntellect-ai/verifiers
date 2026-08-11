"""The verifiers package root.

The v1 stack lives in `verifiers.v1`; the classic (v0) stack lives in
`verifiers.legacy`. The v0 modules also answer at their historical top-level
paths — `verifiers.envs`, `verifiers.types`, `import verifiers as vf`, ... —
through the alias finder below, so importing this package root stays
side-effect free (the v0 surface only loads when something touches it).
"""

import pkgutil
import sys
from functools import cache
from importlib import import_module
from importlib.abc import InspectLoader, Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from importlib.util import find_spec
from typing import TYPE_CHECKING

try:
    __version__ = _version("verifiers")
except PackageNotFoundError:  # source tree without install metadata
    __version__ = "0.0.0+unknown"


@cache
def _legacy_modules() -> frozenset:
    """Top-level modules of `verifiers.legacy`, addressable at `verifiers.<name>`.
    Read off the package directory, without executing its `__init__`."""
    spec = find_spec("verifiers.legacy")
    if spec is None or spec.submodule_search_locations is None:
        return frozenset()
    return frozenset(
        module.name for module in pkgutil.iter_modules(spec.submodule_search_locations)
    )


class LegacyAliasFinder(MetaPathFinder, Loader):
    """Resolve `verifiers.<v0 module>` to `verifiers.legacy.<module>`.

    The loaded module is the legacy module object itself (one instance, aliased
    in `sys.modules` under both names), so `verifiers.types.State is
    verifiers.legacy.types.State`. Installed at the *front* of `sys.meta_path`
    so the path finder never re-executes an aliased package's children under
    the old name.
    """

    _prefix = "verifiers."
    _legacy_prefix = "verifiers.legacy."

    def _legacy_name(self, fullname: str) -> str:
        return self._legacy_prefix + fullname.removeprefix(self._prefix)

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._prefix):
            return None
        top = fullname.removeprefix(self._prefix).split(".", 1)[0]
        # `legacy` first: probing the membership set resolves `verifiers.legacy`
        # through this very finder, so it must pass through before the lookup.
        if top in ("legacy", "v1", "cli") or top not in _legacy_modules():
            return None
        legacy_name = self._legacy_name(fullname)
        try:
            legacy_spec = find_spec(legacy_name)
        except ModuleNotFoundError as e:
            # A miss on the probed module itself just means the alias has no
            # target; anything else (a missing dependency raised while the
            # legacy package initializes) is a real failure and must surface.
            if e.name and legacy_name.startswith(e.name):
                legacy_spec = None
            else:
                raise
        if legacy_spec is None:  # unknown submodule: fail like any missing import
            return None
        spec = ModuleSpec(
            fullname,
            self,
            origin=legacy_spec.origin,
            is_package=legacy_spec.submodule_search_locations is not None,
        )
        if legacy_spec.submodule_search_locations is not None:
            # The real search path (not a copy), so package walkers —
            # pkgutil.iter_modules over the alias spec, child imports through
            # the parent's __path__ — see the legacy tree.
            spec.submodule_search_locations = legacy_spec.submodule_search_locations
        return spec

    def create_module(self, spec):
        module = import_module(self._legacy_name(spec.name))
        # module_from_spec is about to stamp the alias spec onto this shared
        # module object; stash its real identity for exec_module to restore.
        spec.loader_state = (
            module.__dict__.get("__spec__"),
            module.__dict__.get("__loader__"),
        )
        return module

    def exec_module(self, module):
        # Restore the legacy module's own __spec__/__loader__ so reload,
        # find_spec on the module, and importlib.resources keep seeing its
        # real identity — the alias exists only in sys.modules.
        alias_spec = module.__dict__.get("__spec__")
        state = getattr(alias_spec, "loader_state", None)
        if state:
            real_spec, real_loader = state
            if real_spec is not None:
                module.__spec__ = real_spec
                module.__loader__ = real_loader or real_spec.loader
                if real_spec.submodule_search_locations is not None:
                    module.__path__ = real_spec.submodule_search_locations

    def _inspect_loader(self, fullname: str) -> tuple[str, InspectLoader]:
        legacy_name = self._legacy_name(fullname)
        spec = find_spec(legacy_name)
        if spec is None or not isinstance(spec.loader, InspectLoader):
            raise ImportError(
                f"no loadable source behind alias {fullname!r}", name=fullname
            )
        return legacy_name, spec.loader

    # runpy (`python -m verifiers.<v0 module>`) executes through the loader.
    def get_code(self, fullname):
        legacy_name, loader = self._inspect_loader(fullname)
        return loader.get_code(legacy_name)

    def get_source(self, fullname):
        legacy_name, loader = self._inspect_loader(fullname)
        return loader.get_source(legacy_name)


if not any(isinstance(finder, LegacyAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, LegacyAliasFinder())


def __getattr__(name: str):
    """`vf.XMLParser`-style access forwards to the v0 surface in
    `verifiers.legacy` (loaded on first touch). `__all__` forwards too, so
    `from verifiers import *` keeps its pre-move meaning."""
    if name.startswith("_") and name != "__all__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    legacy = import_module("verifiers.legacy")
    return getattr(legacy, name)


def __dir__():
    legacy = import_module("verifiers.legacy")
    return sorted(set(globals()) | set(legacy.__all__) | _legacy_modules())


if TYPE_CHECKING:  # static view of the forwarded surface
    from verifiers.legacy import *
