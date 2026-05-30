import argparse
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import verifiers as vf
from verifiers.utils.env_config_utils import config_table, normalize_env_config_sections
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
    env_config = normalize_env_config_sections(config["env"])
    env_id = env_config.get("id")
    if not isinstance(env_id, str) or not env_id:
        raise SystemExit("Missing required 'env.id' in TOML.")
    env_args = config_table(env_config.get("env_args", {}), "env.env_args")
    env = vf.load_environment(env_id=env_id, **env_args)
    rl_config = RLConfig(**config["trainer"].get("args", {}))
    trainer = RLTrainer(model=model, env=env, args=rl_config)
    trainer.train()


if __name__ == "__main__":
    main()
