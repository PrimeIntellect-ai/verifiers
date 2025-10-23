import argparse
from pathlib import Path

try:
    import tomllib  # type: ignore[unresolved-import]
except ImportError:
    import tomli as tomllib  # type: ignore[unresolved-import]

import verifiers as vf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("at", type=str)
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    if args.at != "@":
        raise SystemExit("Usage: vf-train @ path/to/file.toml")

    config_path_str = args.config_path

    config_path = Path(config_path_str)
    if not config_path.exists():
        raise SystemExit(f"TOML config not found: {config_path}")

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    model = config["model"]
    env_id = config["env"]["id"]
    env_args = config["env"].get("args", {})
    env = vf.load_environment(env_id=env_id, **env_args)
    rl_config = vf.RLConfig(**config["trainer"].get("args", {}))
    trainer = vf.RLTrainer(model=model, env=env, args=rl_config)
    trainer.train()


if __name__ == "__main__":
    main()
