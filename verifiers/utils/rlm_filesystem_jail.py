import builtins
import io
import os
from typing import Any, Callable


class FilesystemJail:
    """Best-effort filesystem jail that restricts access to a root directory."""

    def __init__(
        self,
        root_dir: str,
        *,
        allowed_paths: list[str] | None = None,
        disallowed_modules: set[str] | None = None,
        disallowed_builtins: set[str] | None = None,
    ) -> None:
        self.root_dir = os.path.realpath(root_dir)
        self.allowed_paths = [os.path.realpath(path) for path in (allowed_paths or [])]
        self.disallowed_modules = set(disallowed_modules or [])
        self.disallowed_builtins = set(disallowed_builtins or [])
        self._installed = False

        self._original_open = builtins.open
        self._original_io_open = io.open
        self._original_import = builtins.__import__
        self._os_originals: dict[str, Callable[..., Any]] = {}
        self._removed_builtins: dict[str, Any] = {}

    def _resolve_path(self, path: Any) -> Any:
        if path is None or isinstance(path, int):
            return path
        try:
            raw = os.fspath(path)
        except TypeError:
            return path
        if isinstance(raw, bytes):
            raw = raw.decode()
        if not isinstance(raw, str):
            return path
        if os.path.isabs(raw):
            abs_path = raw
        else:
            abs_path = os.path.join(os.getcwd(), raw)
        abs_path = os.path.realpath(abs_path)
        if not self._is_within_root(abs_path):
            if not self._is_within_allowed(abs_path):
                raise PermissionError(
                    f"Access outside working directory is blocked: {abs_path}"
                )
        return abs_path

    def _is_within_root(self, path: str) -> bool:
        try:
            return os.path.commonpath([self.root_dir, path]) == self.root_dir
        except ValueError:
            return False

    def _is_within_allowed(self, path: str) -> bool:
        for allowed in self.allowed_paths:
            try:
                if os.path.commonpath([allowed, path]) == allowed:
                    return True
            except ValueError:
                continue
        return False

    def _wrap_os_single_path(self, name: str) -> None:
        original = getattr(os, name, None)
        if not callable(original):
            return

        def wrapper(path: Any, *args, **kwargs):
            resolved = self._resolve_path(path)
            return original(resolved, *args, **kwargs)

        self._os_originals[name] = original
        setattr(os, name, wrapper)

    def _wrap_os_two_paths(self, name: str) -> None:
        original = getattr(os, name, None)
        if not callable(original):
            return

        def wrapper(src: Any, dst: Any, *args, **kwargs):
            src_resolved = self._resolve_path(src)
            dst_resolved = self._resolve_path(dst)
            return original(src_resolved, dst_resolved, *args, **kwargs)

        self._os_originals[name] = original
        setattr(os, name, wrapper)

    def _patch_os(self) -> None:
        for name in (
            "listdir",
            "scandir",
            "stat",
            "access",
            "mkdir",
            "makedirs",
            "rmdir",
            "unlink",
            "remove",
            "chdir",
            "open",
        ):
            self._wrap_os_single_path(name)

        for name in ("rename", "replace"):
            self._wrap_os_two_paths(name)

        # Avoid patching os.path.* helpers to prevent recursion through
        # os.path.realpath/commonpath in _resolve_path.

    def _wrap_open(self) -> None:
        def open_wrapper(file: Any, *args, **kwargs):
            resolved = self._resolve_path(file)
            return self._original_open(resolved, *args, **kwargs)

        builtins.open = open_wrapper  # type: ignore[assignment]
        io.open = open_wrapper  # type: ignore[assignment]

    def _wrap_import(self) -> None:
        if not self.disallowed_modules:
            return

        def import_wrapper(name, globals=None, locals=None, fromlist=(), level=0):
            for blocked in self.disallowed_modules:
                if name == blocked or name.startswith(blocked + "."):
                    raise ImportError(f"Import of '{name}' is blocked by RLM policy")
            return self._original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_wrapper  # type: ignore[assignment]

    def _remove_builtins(self) -> None:
        if not self.disallowed_builtins:
            return
        for name in self.disallowed_builtins:
            if hasattr(builtins, name):
                self._removed_builtins[name] = getattr(builtins, name)
                delattr(builtins, name)

    def install(self) -> None:
        if self._installed:
            return
        self._wrap_open()
        self._wrap_import()
        self._patch_os()
        self._remove_builtins()
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        builtins.open = self._original_open  # type: ignore[assignment]
        io.open = self._original_io_open  # type: ignore[assignment]
        builtins.__import__ = self._original_import  # type: ignore[assignment]
        for name, original in self._os_originals.items():
            if name.startswith("path."):
                _, attr = name.split(".", 1)
                setattr(os.path, attr, original)
            else:
                setattr(os, name, original)
        for name, value in self._removed_builtins.items():
            setattr(builtins, name, value)
        self._removed_builtins = {}
        self._os_originals = {}
        self._installed = False
