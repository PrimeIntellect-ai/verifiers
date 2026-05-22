import importlib
from typing import Any, cast

try:
    _TOML = importlib.import_module("tomllib")
except ModuleNotFoundError:
    _TOML = importlib.import_module("tomli")


def load_toml(file_obj: Any) -> dict[str, Any]:
    return cast(dict[str, Any], _TOML.load(file_obj))
