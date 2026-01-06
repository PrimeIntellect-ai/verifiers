import asyncio
import json
from pathlib import Path
from typing import Any, Callable, cast

import verifiers as vf
from .remote_env import RemoteEnv


class RemoteToolWrapper:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        env: "TypeScriptEnv",
    ):
        self.name = name
        self.__name__ = name
        self.__doc__ = description
        self.parameters = parameters
        self.env = env

    async def __call__(self, **kwargs) -> str:
        return await self.env._call_remote_tool(self.name, kwargs)

    def to_oai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.__name__,
                "description": self.__doc__ or "",
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


class RemoteRewardRubric(vf.Rubric):
    def __init__(self, reward_specs: list[dict], env: "TypeScriptEnv", **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.reward_specs = reward_specs

        for spec in reward_specs:
            name = spec["name"]
            weight = spec.get("weight", 1.0)
            reward_func = self._create_reward_func(name)
            self.add_reward_func(reward_func, weight=weight)

    def _create_reward_func(self, name: str) -> Callable:
        async def reward_func(
            prompt: vf.Messages,
            completion: vf.Messages,
            answer: Any,
            state: vf.State,
            **kwargs,
        ) -> float:
            return await self.env._call_remote_reward(name, prompt, completion, answer, state)

        reward_func.__name__ = name
        return reward_func


class TypeScriptEnv(RemoteEnv):
    def __init__(
        self,
        sandbox_path: Path | str,
        server_port: int = 3000,
        server_ready_timeout: int = 120,
        **kwargs,
    ):
        super().__init__(sandbox_path=sandbox_path, **kwargs)

        self.server_port = server_port
        self.server_ready_timeout = server_ready_timeout
        self.remote_tools: dict[str, RemoteToolWrapper] = {}
        self._remote_rubric: RemoteRewardRubric | None = None
        self._tools_discovered = False

    async def _wait_for_server(self, sandbox_id: str) -> None:
        for _ in range(self.server_ready_timeout):
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                f"curl -sf http://localhost:{self.server_port}/tools > /dev/null",
                timeout=5,
            )
            if result.exit_code == 0:
                return

            await asyncio.sleep(1)

        raise TimeoutError(f"Server not ready after {self.server_ready_timeout} seconds")

    async def _discover_tools(self, sandbox_id: str) -> list[dict]:
        result = await self.sandbox_client.execute_command(
            sandbox_id,
            f"curl -sf http://localhost:{self.server_port}/tools",
            timeout=10,
        )

        if result.exit_code != 0:
            raise RuntimeError(f"Failed to fetch tools: {result.stderr}")

        data = json.loads(result.stdout)
        return data["tools"]

    async def _discover_rewards(self, sandbox_id: str) -> list[dict]:
        result = await self.sandbox_client.execute_command(
            sandbox_id,
            f"curl -sf http://localhost:{self.server_port}/rewards",
            timeout=10,
        )

        if result.exit_code != 0:
            raise RuntimeError(f"Failed to fetch rewards: {result.stderr}")

        data = json.loads(result.stdout)
        return data["rewards"]

    def _register_tools(self, tool_specs: list[dict]) -> None:
        for spec in tool_specs:
            func_spec = spec.get("function", spec) if spec.get("type") == "function" else spec
            name = func_spec["name"]
            description = func_spec.get("description", "")
            parameters = func_spec.get("parameters", {"type": "object", "properties": {}})

            wrapper = RemoteToolWrapper(name, description, parameters, self)
            self.remote_tools[name] = wrapper
            self.tools.append(wrapper)
            self.oai_tools.append(wrapper.to_oai_tool())
            self.tool_map[name] = wrapper

    def _register_rewards(self, reward_specs: list[dict]) -> None:
        self._remote_rubric = RemoteRewardRubric(reward_specs, self)
        self.add_rubric(self._remote_rubric)

    async def _call_remote_tool(self, tool_name: str, args: dict) -> str:
        sandbox_id = args.pop("_sandbox_id")
        state = args.pop("_state", None)

        payload = json.dumps({"args": args, "state": state or {}})
        payload_escaped = payload.replace("'", "'\"'\"'")

        result = await self.sandbox_client.execute_command(
            sandbox_id,
            f"curl -sf -X POST http://localhost:{self.server_port}/tools/{tool_name} "
            f"-H 'Content-Type: application/json' -d '{payload_escaped}'",
            timeout=self.timeout_per_command_seconds,
        )

        if result.exit_code != 0:
            return f"Error calling tool {tool_name}: {result.stderr or 'Unknown error'}"

        data = json.loads(result.stdout)
        return data.get("result", str(data))

    async def _call_remote_reward(
        self,
        reward_name: str,
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: Any,
        state: vf.State,
    ) -> float:
        sandbox_id = state["sandbox_id"]

        payload = json.dumps({
            "prompt": prompt,
            "completion": completion,
            "answer": answer,
            "state": {k: v for k, v in state.items() if k not in ["sandbox_state"]},
        })
        payload_escaped = payload.replace("'", "'\"'\"'")

        result = await self.sandbox_client.execute_command(
            sandbox_id,
            f"curl -sf -X POST http://localhost:{self.server_port}/rewards/{reward_name} "
            f"-H 'Content-Type: application/json' -d '{payload_escaped}'",
            timeout=30,
        )

        if result.exit_code != 0:
            raise RuntimeError(f"Reward {reward_name} failed: {result.stderr}")

        data = json.loads(result.stdout)
        return float(data["score"])

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict[str, Any]:
        updated_args = super().update_tool_args(tool_name, tool_args, messages, state, **kwargs)

        if tool_name in self.remote_tools:
            updated_args["_sandbox_id"] = state["sandbox_id"]
            updated_args["_state"] = {k: v for k, v in state.items() if k not in ["sandbox_state"]}

        return updated_args

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        sandbox_id = state["sandbox_id"]

        await self._wait_for_server(sandbox_id)

        if not self._tools_discovered:
            tool_specs = await self._discover_tools(sandbox_id)
            self._register_tools(tool_specs)

            reward_specs = await self._discover_rewards(sandbox_id)
            self._register_rewards(reward_specs)

            self._tools_discovered = True

        return state

    async def call_tool(
        self,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        **kwargs,
    ) -> vf.Message:
        if tool_name in self.remote_tools:
            try:
                result = await self.remote_tools[tool_name](**tool_args)
                return cast(
                    vf.Message,
                    {"role": "tool", "content": str(result), "tool_call_id": tool_call_id},
                )
            except Exception as e:
                return cast(
                    vf.Message,
                    {"role": "tool", "content": self.error_formatter(e), "tool_call_id": tool_call_id},
                )

        return await super().call_tool(tool_name, tool_args, tool_call_id, **kwargs)
