from __future__ import annotations

import importlib
from typing import Any, cast


def _load_toml_module() -> Any:
    try:
        return importlib.import_module("tomllib")
    except ModuleNotFoundError:
        return importlib.import_module("tomli")


_TOML = _load_toml_module()


def load_toml(file_obj: Any) -> dict[str, Any]:
    return cast(dict[str, Any], _TOML.load(file_obj))
