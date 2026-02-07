import importlib


def main() -> None:
    try:
        mod = importlib.import_module("verifiers_rl.rl.inference.server")
    except ModuleNotFoundError as e:
        raise SystemExit(
            "vf-vllm now lives in the optional 'verifiers-rl' package. Install with: uv add verifiers-rl"
        ) from e
    mod.main()
