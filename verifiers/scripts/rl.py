import importlib


def main() -> None:
    try:
        mod = importlib.import_module("verifiers_rl.scripts.rl")
    except ModuleNotFoundError as e:
        raise SystemExit(
            "vf-rl now lives in the optional 'verifiers-rl' package. Install with: uv add verifiers-rl"
        ) from e
    mod.main()
