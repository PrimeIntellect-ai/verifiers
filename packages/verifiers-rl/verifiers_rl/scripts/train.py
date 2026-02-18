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
    env_id = config["env"]["id"]
    env_args = config["env"].get("args", {})

    # Resolve tools from config
    if "tools" in config["env"]:
        from verifiers.utils.tool_registry import (
            get_tools as registry_get_tools,
            validate_tools,
        )

        tool_names = config["env"]["tools"]
        if not isinstance(tool_names, list):
            raise ValueError(
                f"env.tools must be list of tool names, got {type(tool_names).__name__}"
            )

        logger.info(f"Loading tools from config: {tool_names}")

        # Validate before loading
        try:
            validate_tools(env_id, tool_names)
        except ValueError as e:
            raise ValueError(f"Tool validation failed for env '{env_id}': {e}") from e

        tools = registry_get_tools(env_id, tool_names)
    else:
        tools = None

    env = vf.load_environment(env_id=env_id, tools=tools, **env_args)
    rl_config = RLConfig(**config["trainer"].get("args", {}))
    trainer = RLTrainer(model=model, env=env, args=rl_config)
    trainer.train()


if __name__ == "__main__":
    main()
