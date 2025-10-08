import argparse
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

import verifiers as vf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("at", type=str)
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    if args.at != "@":
        raise SystemExit("Usage: vf-rl @ path/to/file.toml")

    config_path_str = args.config_path

    config_path = Path(config_path_str)
    if not config_path.exists():
        raise SystemExit(f"TOML config not found: {config_path}")

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    model = config["model"]
    env = vf.load_environment(env_id=config["env"]["id"], **config["env"]["args"])
    rl_config = vf.RLConfig(**config["trainer"]["args"])
    trainer = vf.RLTrainer(model=model, env=env, args=rl_config)
    trainer.train()


if __name__ == "__main__":
    main()
