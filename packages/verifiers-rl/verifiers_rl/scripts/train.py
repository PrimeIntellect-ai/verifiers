import argparse
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import verifiers as vf
from verifiers.utils.v1_config_aliases import normalize_v1_config_aliases
from verifiers_rl.rl.trainer import RLConfig, RLTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("at", type=str)
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    if args.at != "@":
        raise SystemExit("Usage: vf-train @ path/to/file.toml")

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise SystemExit(f"TOML config not found: {config_path}")

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    model = config["model"]
    env_config = normalize_v1_config_aliases(
        config["env"],
        args_key="args",
        global_harness=config.get("harness"),
    )
    env_id = env_config["id"]
    env_args = env_config.get("args", {})
    env = vf.load_environment(env_id=env_id, **env_args)
    rl_config = RLConfig(**config["trainer"].get("args", {}))
    trainer = RLTrainer(model=model, env=env, args=rl_config)
    trainer.train()


if __name__ == "__main__":
    main()
