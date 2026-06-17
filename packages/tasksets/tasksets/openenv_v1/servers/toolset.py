import asyncio
import inspect
from typing import Annotated, Any

import aiohttp
from pydantic import WithJsonSchema

import verifiers.v1 as vf
from tasksets.openenv_v1.taskset import OpenEnvState, OpenEnvTask


class OpenEnvToolset(vf.Toolset[vf.ToolsetConfig, OpenEnvState]):
    TOOL_PREFIX = "openenv"

    async def setup_task(self, task: OpenEnvTask) -> None:
        self.task = task
        session = aiohttp.ClientSession()
        socket = await session.ws_connect(
            f"http://127.0.0.1:{task.port}/ws",
            max_msg_size=100 << 20,
        )
        await socket.send_json({"type": "step", "data": {"type": "list_tools"}})
        response = await socket.receive_json()
        await socket.close()
        await session.close()
        if response.get("type") == "error":
            raise RuntimeError(response["data"]["message"])
        self.tools = response["data"]["observation"]["tools"]
        self.session = None
        self.socket = None
        self.lock = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:
            if self.socket is None:
                self.session = aiohttp.ClientSession()
                self.socket = await self.session.ws_connect(
                    f"http://127.0.0.1:{self.task.port}/ws",
                    max_msg_size=100 << 20,
                )
                await self.socket.send_json(
                    {"type": "reset", "data": {"seed": self.task.seed}}
                )
                reset = await self.socket.receive_json()
                if reset.get("type") == "error":
                    raise RuntimeError(reset["data"]["message"])
                self.state.done = reset["data"].get("done", False)

            await self.socket.send_json(
                {
                    "type": "step",
                    "data": {
                        "type": "call_tool",
                        "tool_name": name,
                        "arguments": arguments,
                    },
                }
            )
            response = await self.socket.receive_json()
            if response.get("type") == "error":
                raise RuntimeError(response["data"]["message"])
            result = response["data"]
            self.state.reward += result.get("reward") or 0.0
            self.state.done = result.get("done", False)
            observation = result.get("observation", {})
            if observation.get("error") is not None:
                return {"error": observation["error"]}
            value = observation.get("result")
            if isinstance(value, dict) and "data" in value:
                return value["data"]
            return value

    def _register(self, mcp) -> None:
        for tool in self.tools:

            async def call(_name=tool["name"], **arguments):
                return await self.call_tool(
                    _name,
                    {
                        key: value
                        for key, value in arguments.items()
                        if value is not None
                    },
                )

            schema = tool.get("input_schema") or {"properties": {}}
            required = set(schema.get("required", []))
            call.__name__ = tool["name"]
            call.__doc__ = tool.get("description", "")
            call.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
                [
                    inspect.Parameter(
                        name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=inspect.Parameter.empty
                        if name in required
                        else spec.get("default"),
                        annotation=Annotated[Any, WithJsonSchema(spec)],
                    )
                    for name, spec in schema.get("properties", {}).items()
                ],
                return_annotation=Any,
            )
            mcp.add_tool(
                self._with_state(call),
                name=tool["name"],
                description=tool.get("description") or None,
            )


if __name__ == "__main__":
    OpenEnvToolset.run()
