import importlib


def __getattr__(name: str):
    try:
        mod = importlib.import_module("verifiers_rl.rl.trainer")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "verifiers.rl moved to optional package 'verifiers-rl'. Install with: uv add verifiers-rl"
        ) from e
    return getattr(mod, name)
