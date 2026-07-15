# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "nemo-gym @ git+https://github.com/NVIDIA-NeMo/Gym.git@05eb7df9f42f59286abac339593473ef7ca9448d",
# ]
# ///
"""Launch only NeMo Gym's upstream ``example_mcp_weather`` resources server.

This module intentionally bypasses Gym's agent/model/head stack. Its inline dependency
pins the upstream version this adapter is verified against.
"""

import os

import uvicorn
from omegaconf import OmegaConf

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.example_mcp_weather.app import (
    ExampleMCPWeatherResourcesServer,
    ExampleMCPWeatherResourcesServerConfig,
)

HOST = os.environ.get("NEMO_GYM_HOST", "127.0.0.1")
PORT = int(os.environ.get("NEMO_GYM_PORT", "8000"))


def main() -> None:
    config = ExampleMCPWeatherResourcesServerConfig(
        name="example_mcp_weather",
        host=HOST,
        port=PORT,
        entrypoint="app.py",
        domain="agent",
    )
    client = ServerClient(
        head_server_config=BaseServerConfig(host=HOST, port=11000),
        global_config_dict=OmegaConf.create({}),
    )
    server = ExampleMCPWeatherResourcesServer(
        config=config,
        server_client=client,
    )
    app = server.setup_webserver()
    server.setup_liveness(app)
    server.setup_exception_middleware(app)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
