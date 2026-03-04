import importlib


try:
    _inference = importlib.import_module("verifiers_rl.rl.inference")
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "verifiers.rl.inference moved to optional package 'verifiers-rl'. Install with: uv add verifiers-rl"
    ) from e

__all__ = getattr(_inference, "__all__", [])
