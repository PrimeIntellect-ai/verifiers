"""Run OpenEnv environments with UV by default, or Docker when requested.

The engine plays the user: the env's `run()` opens the model's interaction and
steps the OpenEnv client host-side — each assistant action advances the
environment, the next observation comes back as the user turn, and a `done` result
ends the exchange. OpenEnv's per-step rewards are summed onto the seat's trace
(`openenv_reward`)."""

import asyncio
import base64
import json
import re
import subprocess
import threading
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any, cast

import httpx
from pydantic import Field

import verifiers.v1 as vf

# Magic bytes of the media an observation may carry, keyed to their MIME type.
MEDIA_TYPES = {
    rb"\x89PNG": "image/png",
    rb"\xff\xd8\xff": "image/jpeg",
    rb"RIFF....WEBP": "image/webp",
    rb"RIFF....WAVE": "audio/wav",
}
# Observation fields meant for the harness, not the model: OpenEnv's `metadata`
# (which may identify the task) and an env-supplied system prompt, which is sent
# as the system message instead.
HIDDEN_FIELDS = {"metadata", "system_prompt"}
DOCKER_START = threading.Lock()


class OpenEnvData(vf.TaskData):
    reset: dict[str, Any]


class OpenEnvConfig(vf.TasksetConfig):
    env: str | None = None
    """Environment id passed to OpenEnv. Required unless `base_url` is set."""
    base_url: str | None = None
    """Connect to an existing OpenEnv server instead of starting `env`."""
    use_docker: bool = False
    """Use OpenEnv's Docker provider instead of the default UV provider. The engine
    runs host-side (in the eval process), so this needs Docker on the host."""
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra arguments for OpenEnv's provider: `UVProvider(...)`, or
    `LocalDockerProvider.start_container(...)` plus the image `tag`."""
    resets: list[dict[str, Any]] = Field(default_factory=lambda: [{}])
    """One finite task per set of arguments passed to OpenEnv's `reset`. Each task
    resets with `seed=<task index>` unless its arguments set a seed."""
    split: str | None = None
    """Load one task per row of this split from the server's OpenEnv Task API,
    reset with `split` and `index`, instead of `resets`. Needs `base_url`."""
    timeout: float = 600.0
    """Seconds to wait for a started server to become ready, and for each OpenEnv
    reset or step reply."""


class OpenEnvTask(vf.Task[OpenEnvData]):
    pass


def parse_action(message: str, action_schema: dict[str, Any]) -> dict[str, Any]:
    """The model's reply as an OpenEnv action dict (JSON, fenced JSON, or — for a
    single-field schema such as Wordle's — the reply as that field's value, kept as
    raw text when the field takes a string)."""
    message = message.strip()
    if message.startswith("```") and message.endswith("```"):
        message = "\n".join(message.splitlines()[1:-1]).strip()
    try:
        action = json.loads(message)
    except json.JSONDecodeError:
        action = None
    if isinstance(action, dict):
        return action
    properties = action_schema["properties"]
    fields = action_schema.get("required") or [
        name for name in properties if name != "metadata"
    ]
    if len(fields) != 1:
        raise ValueError("non-object actions require a single-field action schema")
    schema = properties[fields[0]]
    options = [schema, *schema.get("anyOf", []), *schema.get("oneOf", [])]
    if action is None or any("string" in option.get("type", "") for option in options):
        action = message
    return {fields[0]: action}


def media_part(data: bytes) -> vf.ContentPart | None:
    """`data` as an image or audio content part, or None when it is neither."""
    mime = next(
        (
            mime
            for magic, mime in MEDIA_TYPES.items()
            if re.match(magic, data, re.DOTALL)
        ),
        None,
    )
    if mime is None:
        return None
    encoded = base64.b64encode(data).decode()
    kind, _, subtype = mime.partition("/")
    if kind == "audio":
        return vf.InputAudioContentPart(
            input_audio=vf.InputAudioSource(data=encoded, format=subtype)
        )
    return vf.ImageUrlContentPart(
        image_url=vf.ImageUrlSource(url=f"data:{mime};base64,{encoded}")
    )


def numeric_fields(
    value: Any, path: tuple[str, ...] = ()
) -> Iterator[tuple[str, float]]:
    """Every number (bools included) nested in `value`, named by its `/`-joined keys."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_fields(item, (*path, key))
    elif isinstance(value, (bool, int, float)):
        yield "/".join(path), float(value)


def split_media(value: Any, media: list[bytes | str]) -> Any:
    """`value` with every media payload moved to `media` and replaced by a
    placeholder: decoded `*base64` fields and MCP image blocks, and the server path
    of each `asset_path` file. Empty fields are dropped."""
    if isinstance(value, list):
        return [split_media(item, media) for item in value]
    if not isinstance(value, dict):
        return value
    fields: dict[str, Any] = {}
    for key, item in value.items():
        if item in (None, "", [], {}):
            continue
        if key == "asset_path":
            media.append(item)
        elif key.endswith("base64") or (key == "data" and value.get("type") == "image"):
            media.append(base64.b64decode(item))
        else:
            fields[key] = split_media(item, media)
            continue
        fields[key] = "<attached>"
    return fields


async def observation_content(
    observation: dict[str, Any], http: httpx.AsyncClient
) -> list[vf.ContentPart]:
    """The observation as user-turn content: its media as content parts, followed by
    the rest as JSON text."""
    media: list[bytes | str] = []
    fields = split_media(
        {key: value for key, value in observation.items() if key not in HIDDEN_FIELDS},
        media,
    )
    parts: list[vf.ContentPart] = []
    for item in media:
        if isinstance(item, str):
            # httpx fetches an absolute URL as-is, bypassing the env server.
            if httpx.URL(item).is_absolute_url:
                raise ValueError(f"`asset_path` must be a server path, got {item!r}")
            response = await http.get(item)
            response.raise_for_status()
            item = response.content
        part = media_part(item)
        if part is None:
            raise ValueError("observation carries media of an unknown type")
        parts.append(part)
    return [*parts, vf.TextContentPart(text=json.dumps(fields, ensure_ascii=False))]


def start_server(config: OpenEnvConfig) -> tuple[Callable[[], None], str]:
    """Start `config.env` as OpenEnv's `from_env` does, without opening a session:
    returns the server's stop function and URL. Blocks, so run it in a thread."""
    from openenv.core.containers.runtime import LocalDockerProvider, UVProvider

    assert config.env is not None
    kwargs = dict(config.provider_kwargs)
    if config.use_docker:
        docker = LocalDockerProvider()
        tag = kwargs.pop("tag", "latest")
        image = f"registry.hf.space/{config.env.replace('/', '-')}:{tag}"
        # OpenEnv names containers by millisecond and probes for a free port, so
        # concurrent starts collide; one at a time, they never do.
        with DOCKER_START:
            base_url = docker.start_container(image, **kwargs)
        assert docker._container_id is not None
        # Not `stop_container`: OpenEnv images ignore SIGTERM, so its `docker stop`
        # outlives its own 10s timeout and never reaches `docker rm`.
        stop = partial(
            subprocess.run,
            ["docker", "rm", "-f", docker._container_id],
            capture_output=True,
            check=True,
        )
        ready = partial(docker.wait_for_ready, base_url, timeout_s=config.timeout)
    else:
        kwargs.setdefault(
            "project_path", f"git+https://huggingface.co/spaces/{config.env}"
        )
        uv = UVProvider(**kwargs)
        base_url = uv.start()
        stop = uv.stop
        ready = partial(uv.wait_for_ready, timeout_s=config.timeout)
    try:
        ready()
    except BaseException:
        stop()
        raise
    return stop, base_url


class OpenEnvEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig()


class OpenEnvEnv(vf.Env[OpenEnvEnvConfig]):
    async def start(self) -> None:
        # Servers started from `env`, idle between episodes. OpenEnv servers default
        # to a single session, so each concurrent episode gets a server of its own.
        self._idle: list[tuple[Callable[[], None], str]] = []
        self._starting: set[asyncio.Future] = set()

    async def stop(self) -> None:
        # Starts abandoned by cancelled episodes still finish; stop those servers too.
        started = await asyncio.gather(*self._starting, return_exceptions=True)
        servers = [*self._idle, *(s for s in started if isinstance(s, tuple))]
        # gather runs every stop to completion even when one of them raises.
        await asyncio.gather(*(asyncio.to_thread(stop) for stop, _ in servers))

    async def run(self, task, agents):
        config = cast(OpenEnvConfig, self.taskset.config)
        if config.base_url:
            await self.play(task, agents, config.base_url)
            return
        if self._idle:
            server = self._idle.pop()
        else:
            # A worker thread can't be cancelled, so a cancelled episode leaves its
            # start to finish under the shield and to `stop()`.
            starting = asyncio.ensure_future(asyncio.to_thread(start_server, config))
            self._starting.add(starting)
            server = await asyncio.shield(starting)
            self._starting.discard(starting)
        stop, base_url = server
        try:
            await self.play(task, agents, base_url)
        except BaseException:
            await asyncio.to_thread(stop)
            raise
        self._idle.append(server)

    async def play(self, task, agents, base_url: str) -> None:
        """One episode against the OpenEnv server at `base_url`."""
        from openenv import GenericEnvClient
        from openenv.core import CallToolAction

        config = cast(OpenEnvConfig, self.taskset.config)
        client = GenericEnvClient(base_url=base_url, message_timeout_s=config.timeout)
        http = httpx.AsyncClient(base_url=base_url, timeout=config.timeout)
        async with client, http:
            response = await http.get("/schema")
            response.raise_for_status()
            action_schema = response.json()["action"]
            if action_schema.get("title") in {
                "Action",
                "CallToolAction",
                "ListToolsAction",
            }:
                # Generic MCP Action omits how to call the tools it advertises.
                result = await client.step({"type": "list_tools"})
                action_schema = CallToolAction.model_json_schema() | {
                    "available_tools": result.observation["tools"]
                }
            result = await client.reset(**{"seed": task.data.idx} | task.data.reset)
            # OpenEnv reports a reward per reset and step, None when it gave none.
            rewards = [result.reward]
            schema = "Reply with an action matching this JSON schema:\n" + json.dumps(
                action_schema, ensure_ascii=False
            )
            content = await observation_content(result.observation, http)
            messages: vf.Messages = [
                vf.UserMessage(content=[*content, vf.TextContentPart(text=schema)])
            ]
            if system_prompt := result.observation.get("system_prompt"):
                messages.insert(0, vf.SystemMessage(content=system_prompt))

            async with agents.player.interaction(task) as interaction:
                while not result.done:
                    segment = await interaction.turn(messages)
                    if segment.terminated:
                        break
                    action = segment.last_reply.strip()
                    if not action:
                        # No action can advance OpenEnv. End this run explicitly
                        # instead of replaying the same observation forever when
                        # the agent has no turn or episode cap.
                        interaction.trace.stop("empty_action")
                        break
                    try:
                        result = await client.step(parse_action(action, action_schema))
                    except (ValueError, RuntimeError) as e:
                        if not isinstance(
                            e, ValueError
                        ) and "VALIDATION_ERROR" not in str(e):
                            raise
                        # A malformed action is the policy's failure, not the env's:
                        # score the episode as it stands instead of erroring it.
                        interaction.trace.stop("invalid_action")
                        break
                    rewards.append(result.reward)
                    if result.done:
                        break
                    content = await observation_content(result.observation, http)
                    messages = [vf.UserMessage(content=content)]
        trace = interaction.trace
        scores = [reward for reward in rewards if reward is not None]
        if result.done and not scores:
            # The env ended the episode without scoring it (e.g. its grader failed).
            trace.rewards["openenv_reward"] = None
        else:
            trace.record_reward("openenv_reward", sum(scores))
        # The last observation carries the env's grading details (often revealed
        # only once the episode ends).
        final = split_media(result.observation, [])
        trace.info["openenv"] = final
        for name, value in numeric_fields(final):
            trace.record_metric(name, value)


class OpenEnvTaskset(vf.Taskset[OpenEnvTask, OpenEnvConfig]):
    def load(self) -> Iterator[OpenEnvTask]:
        config = self.config
        resets = config.resets
        source = config.base_url or config.env
        if not source:
            raise ValueError("pass `env` or `base_url`")
        if config.split:
            if not config.base_url:
                raise ValueError("`split` reads the Task API of a running `base_url`")
            base_url = config.base_url.rstrip("/")
            response = httpx.get(
                f"{base_url}/list_environments", timeout=config.timeout
            )
            response.raise_for_status()
            (name,) = response.json()
            response = httpx.post(
                f"{base_url}/{name}/num_tasks",
                json={"split": config.split},
                timeout=config.timeout,
            )
            response.raise_for_status()
            resets = (
                {"split": config.split, "index": index}
                for index in range(response.json()["num_tasks"])
            )
        for idx, reset in enumerate(resets):
            yield OpenEnvTask(
                OpenEnvData(idx=idx, name=f"{source}#{idx}", prompt=None, reset=reset),
                config.task,
            )
