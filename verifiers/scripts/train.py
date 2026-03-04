import importlib


def main() -> None:
    try:
        mod = importlib.import_module("verifiers_rl.scripts.train")
    except ModuleNotFoundError as e:
        raise SystemExit(
            "vf-train now lives in the optional 'verifiers-rl' package. Install with: uv add verifiers-rl"
        ) from e
    mod.main()
