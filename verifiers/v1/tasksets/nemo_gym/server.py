"""Serve one resource-server class from the published NeMo Gym package."""

import os
from importlib import import_module

import uvicorn
from omegaconf import OmegaConf

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient

HOST = os.environ.get("NEMO_GYM_HOST", "127.0.0.1")
PORT = int(os.environ.get("NEMO_GYM_PORT", "8000"))


def main() -> None:
    module_name, class_name = os.environ["NEMO_GYM_RESOURCE_SERVER"].split(":", 1)
    server_class = getattr(import_module(module_name), class_name)
    config_class = server_class.model_fields["config"].annotation
    name = module_name.split(".")[-2]
    server = server_class(
        config=config_class(name=name, host=HOST, port=PORT, entrypoint="app.py"),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host=HOST, port=11000),
            global_config_dict=OmegaConf.create({}),
        ),
    )
    app = server.setup_webserver()
    server.setup_liveness(app)
    server.setup_exception_middleware(app)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
