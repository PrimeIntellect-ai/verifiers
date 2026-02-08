import importlib


def main() -> None:
    try:
        mod = importlib.import_module("verifiers_rl.rl.inference.server")
    except ModuleNotFoundError as e:
        raise SystemExit(
            "verifiers.rl.inference.server moved to optional package 'verifiers-rl'. Install with: uv add verifiers-rl"
        ) from e
    mod.main()
