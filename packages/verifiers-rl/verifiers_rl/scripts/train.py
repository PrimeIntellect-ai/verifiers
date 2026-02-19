import argparse
import logging
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import verifiers as vf
from verifiers_rl.rl.trainer import RLConfig, RLTrainer

logger = logging.getLogger("verifiers_rl.scripts.train")


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

    # Handle both [[env]] array syntax (configs/rl/*.toml) and [env] dict syntax (configs/local/vf-rl/*.toml)
    env_section = config["env"]
    if isinstance(env_section, list):
        # [[env]] array - use first environment
        env_config = env_section[0]
        if len(env_section) > 1:
            logger.warning(f"Multiple environments in config, using first: {env_config['id']}")
    else:
        # [env] dict - single environment
        env_config = env_section

    env_id = env_config["id"]
    env_args = env_config.get("args", {})

    # Extract tools from config (will be resolved by env_utils.py after environment import)
    if "tools" in env_config:
        tool_names = env_config["tools"]
        if not isinstance(tool_names, list):
            raise ValueError(
                f"env.tools must be list of tool names, got {type(tool_names).__name__}"
            )
        tools = tool_names  # Pass as-is, let env_utils.py resolve after import
    else:
        tools = None

    env = vf.load_environment(env_id=env_id, tools=tools, **env_args)
    rl_config = RLConfig(**config["trainer"].get("args", {}))
    trainer = RLTrainer(model=model, env=env, args=rl_config)
    trainer.train()


if __name__ == "__main__":
    main()
