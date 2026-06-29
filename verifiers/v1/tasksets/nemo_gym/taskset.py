"""Run a packaged NeMo Gym dataset with any MCP-capable Verifiers harness."""

import importlib
import json
from pathlib import Path
from typing import Any

import httpx

from verifiers.v1.decorators import tool
from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.mcp import Toolset, ToolsetConfig
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig

DEFAULT_RESOURCE_SERVER = "example_single_tool_call"


class NeMoGymTask(Task):
    nemo_gym_row: dict[str, Any]


class NeMoGymConfig(TasksetConfig):
    resource_server: str = DEFAULT_RESOURCE_SERVER
    config_name: str | None = None
    data_name: str = "example.jsonl"


class _NeMoGymToolsConfig(ToolsetConfig):
    resource_server: str
    config_name: str | None = None


class _NeMoGymTools(Toolset[_NeMoGymToolsConfig]):
    """Expose a NeMo Gym resource server through one generic MCP call."""

    TOOL_PREFIX = "nemo_gym"

    async def setup(self) -> None:
        nemo_gym = importlib.import_module("nemo_gym")
        SimpleResourcesServer = importlib.import_module(
            "nemo_gym.base_resources_server"
        ).SimpleResourcesServer
        BaseServerConfig = importlib.import_module(
            "nemo_gym.config_types"
        ).BaseServerConfig
        global_config = importlib.import_module("nemo_gym.global_config")
        GlobalConfigDictParser = global_config.GlobalConfigDictParser
        GlobalConfigDictParserConfig = global_config.GlobalConfigDictParserConfig
        ServerClient = importlib.import_module("nemo_gym.server_utils").ServerClient
        OmegaConf = importlib.import_module("omegaconf").OmegaConf

        root = Path(nemo_gym.PARENT_DIR)
        config_name = self.config.config_name or self.config.resource_server
        path = (
            root
            / "resources_servers"
            / self.config.resource_server
            / "configs"
            / f"{config_name}.yaml"
        )
        parser = GlobalConfigDictParser()
        config = parser.parse_no_environment(
            OmegaConf.merge(
                OmegaConf.load(path),
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
            )
        )
        resource = next(
            server
            for server in parser.filter_for_server_instance_configs(config)
            if server.SERVER_TYPE == "resources_servers"
        )
        entrypoint = Path(resource.get_inner_run_server_config().entrypoint).stem
        module = importlib.import_module(
            f"resources_servers.{self.config.resource_server}.{entrypoint}"
        )
        server_cls: Any = next(
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, SimpleResourcesServer)
            and value.__module__ == module.__name__
        )
        server_config = server_cls.model_fields["config"].annotation.model_validate(
            {
                "name": resource.name,
                **resource.get_inner_run_server_config().model_dump(),
            }
        )
        server = server_cls(
            config=server_config,
            server_client=ServerClient(
                head_server_config=BaseServerConfig.model_validate(
                    config["head_server"]
                ),
                global_config_dict=config,
            ),
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.setup_webserver()),
            base_url="http://nemo-gym",
        )

    async def setup_task(self, task: NeMoGymTask) -> None:
        self.tool_names = {
            spec["name"]
            for spec in task.nemo_gym_row["responses_create_params"].get("tools", [])
        }
        response = await self.client.post("/seed_session", json=task.nemo_gym_row)
        response.raise_for_status()

    @tool
    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool declared by the current NeMo Gym task."""
        if name not in self.tool_names:
            raise ValueError(f"unknown NeMo Gym tool: {name}")
        response = await self.client.post(f"/{name}", json=arguments)
        response.raise_for_status()
        return response.json()


class NeMoGymTaskset(Taskset[NeMoGymTask, NeMoGymConfig]):
    def load_tasks(self) -> list[NeMoGymTask]:
        try:
            nemo_gym = importlib.import_module("nemo_gym")
        except ImportError as exc:
            raise ImportError(
                "Run this taskset with `uv run --with nemo-gym==0.3.0 eval nemo_gym`."
            ) from exc

        path = (
            Path(nemo_gym.PARENT_DIR)
            / "resources_servers"
            / self.config.resource_server
            / "data"
            / self.config.data_name
        )
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        dialect = ResponsesDialect()
        return [
            NeMoGymTask(
                idx=idx,
                name=f"{self.config.resource_server}:{idx}",
                prompt=dialect.parse_request(row["responses_create_params"])[0],
                system_prompt=(
                    "Call `nemo_gym_call` with the matching tool name and arguments. "
                    f"Available NeMo Gym tools: "
                    f"{json.dumps(row['responses_create_params'].get('tools', []))}"
                ),
                nemo_gym_row=row,
            )
            for idx, row in enumerate(rows)
        ]

    def tools(self, task: NeMoGymTask) -> list[Toolset]:
        if not task.nemo_gym_row["responses_create_params"].get("tools"):
            return []
        return [
            _NeMoGymTools(
                _NeMoGymToolsConfig(
                    resource_server=self.config.resource_server,
                    config_name=self.config.config_name,
                )
            )
        ]


if __name__ == "__main__":
    _NeMoGymTools.run()
