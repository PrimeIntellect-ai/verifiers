import json
from urllib.request import urlopen

import aiohttp

import verifiers.v1 as vf
from tasksets.openenv_v1.types import OpenEnvState, OpenEnvTask


class OpenEnvUser(vf.User[vf.UserConfig, OpenEnvState]):
    """Drive an OpenEnv gym episode through the v1 user-simulator interface."""

    async def setup_task(self, task: OpenEnvTask) -> None:
        self.task = task
        with urlopen(f"http://127.0.0.1:{task.port}/schema", timeout=5) as response:
            self.action_schema = json.load(response)["action"]
        self.socket = None

    async def respond(self, message: str) -> vf.Messages:
        stepping = self.socket is not None
        if not stepping:
            self.session = aiohttp.ClientSession()
            self.socket = await self.session.ws_connect(
                f"http://127.0.0.1:{self.task.port}/ws",
                max_msg_size=100 << 20,
            )
            payload = {"type": "reset", "data": {"seed": self.task.seed}}
        else:
            cleaned = message.strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
            properties = self.action_schema.get("properties", {})
            required = self.action_schema.get("required", [])
            fields = required if len(required) == 1 else list(properties)
            field = (
                fields[0]
                if len(fields) == 1
                and properties.get(fields[0], {}).get("type") == "string"
                else None
            )
            try:
                action = json.loads(cleaned)
            except json.JSONDecodeError:
                action = cleaned
            if not isinstance(action, dict) or (
                field is not None and field not in action
            ):
                if field is None:
                    raise ValueError(
                        "Return a JSON object matching the OpenEnv action schema."
                    )
                value = (
                    next(iter(action.values()))
                    if isinstance(action, dict) and len(action) == 1
                    else action
                )
                action = {field: value if isinstance(value, str) else cleaned}
            payload = {"type": "step", "data": action}

        await self.socket.send_json(payload)
        response = await self.socket.receive_json()
        if response.get("type") == "error":
            raise RuntimeError(response["data"]["message"])
        result = response["data"]
        if stepping:
            self.state.reward += result.get("reward") or 0.0
        self.state.done = result.get("done", False)

        observation = result.get("observation")
        content = observation
        if isinstance(observation, dict):
            messages = observation.get("messages") or []
            latest = messages[-1] if messages else None
            if isinstance(latest, dict):
                latest = latest.get("content")
            prompt = next(
                (
                    observation[key]
                    for key in ("prompt", "question", "instruction", "content", "text")
                    if isinstance(observation.get(key), str)
                    and observation[key].strip()
                ),
                None,
            )
            content = (
                latest if stepping and latest else prompt or latest
            ) or json.dumps(observation, ensure_ascii=False)

        if self.state.done:
            await self.socket.close()
            await self.session.close()
        return [{"role": "user", "content": str(content)}]


if __name__ == "__main__":
    OpenEnvUser.run()
