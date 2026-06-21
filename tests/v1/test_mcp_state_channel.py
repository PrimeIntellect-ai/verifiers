import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    field_serializer,
    field_validator,
)

from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.mcp import server as mcp_server
from verifiers.v1.mcp.toolset import Toolset, ToolsetConfig
from verifiers.v1.state import State


_SECRET = "state-secret"


class NestedState(BaseModel):
    count: int = 1


class SpecialState(State):
    nested: NestedState = Field(default_factory=NestedState)
    text: str = "Grüße 東京"
    negative_zero: float = -0.0
    not_a_number: float = math.nan
    positive_infinity: float = math.inf
    negative_infinity: float = -math.inf
    custom: int = 7

    @field_validator("custom", mode="before")
    @classmethod
    def parse_custom(cls, value: int | str) -> int:
        return int(value.removeprefix("value=")) if isinstance(value, str) else value

    @field_serializer("custom")
    def serialize_custom(self, value: int) -> str:
        return f"value={value}"


class DeepState(State):
    payload: list[object] = Field(default_factory=list)


class ProbeToolset(Toolset[ToolsetConfig, SpecialState]):
    pass


class DeepToolset(Toolset[ToolsetConfig, DeepState]):
    pass


def _json_bytes(state: State) -> bytes:
    return TypeAdapter(type(state)).dump_json(state)


def _request(raw: bytes = b"") -> MagicMock:
    request = MagicMock()
    request.headers = {"Authorization": f"Bearer {_SECRET}"}
    request.read = AsyncMock(return_value=raw)
    return request


def _interception_server(state: State) -> InterceptionServer:
    server = InterceptionServer()
    trace = SimpleNamespace(id="trace-id", state=state)
    server.sessions[_SECRET] = SimpleNamespace(trace=trace)
    return server


async def test_interception_state_bytes_round_trip_special_values():
    server = _interception_server(SpecialState())

    response = await server.handle_state_get(_request())

    assert response.headers["Content-Type"] == "application/json; charset=utf-8"
    assert response.body == _json_bytes(server.sessions[_SECRET].trace.state)
    assert "Grüße 東京".encode() in response.body
    assert b'"negative_zero":-0.0' in response.body
    assert b'"not_a_number":NaN' in response.body
    assert b'"positive_infinity":Infinity' in response.body
    assert b'"negative_infinity":-Infinity' in response.body
    assert b'"custom":"value=7"' in response.body

    put_response = await server.handle_state_put(_request(response.body))
    restored = server.sessions[_SECRET].trace.state

    assert put_response.status == 200
    assert isinstance(restored.nested, NestedState)
    assert restored.text == "Grüße 東京"
    assert math.copysign(1, restored.negative_zero) == -1
    assert math.isnan(restored.not_a_number)
    assert restored.positive_infinity == math.inf
    assert restored.negative_infinity == -math.inf
    assert restored.custom == 7


async def test_mcp_state_sync_sends_one_typed_json_put_only_when_changed(
    monkeypatch,
):
    remote_state = [SpecialState()]
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, content=_json_bytes(remote_state[0]))
        remote_state[0] = SpecialState.model_validate_json(request.content)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setenv("VF_STATE_URL", "http://state.test/state")
    monkeypatch.setenv("VF_STATE_SECRET", _SECRET)
    toolset = ProbeToolset(ToolsetConfig())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        toolset._state_client = client
        result = await toolset._with_state(lambda: toolset.state.text)()

        assert result == "Grüße 東京"
        assert [request.method for request in requests] == ["GET"]

        def mutate_state() -> None:
            toolset.state.nested.count += 1
            toolset.state.text = "naïve café"
            toolset.state.negative_zero = 0.0

        await toolset._with_state(mutate_state)()

    puts = [request for request in requests if request.method == "PUT"]
    assert [request.method for request in requests] == ["GET", "GET", "PUT"]
    assert len(puts) == 1
    assert puts[0].headers["Content-Type"] == "application/json"
    assert puts[0].content == _json_bytes(remote_state[0])
    assert remote_state[0].nested.count == 2
    assert remote_state[0].text == "naïve café"
    assert math.copysign(1, remote_state[0].negative_zero) == 1
    assert math.isnan(remote_state[0].not_a_number)
    assert remote_state[0].positive_infinity == math.inf
    assert remote_state[0].custom == 7


async def test_state_put_falls_back_to_stdlib_for_valid_deep_json():
    depth = 250
    raw = b'{"payload":' + b"[" * depth + b"0" + b"]" * depth + b"}"
    server = _interception_server(DeepState())
    with pytest.raises(ValidationError, match="recursion limit exceeded"):
        DeepState.model_validate_json(raw)

    response = await server.handle_state_put(_request(raw))

    assert response.status == 200
    value = server.sessions[_SECRET].trace.state.payload
    for _ in range(depth):
        value = value[0]
    assert value == 0


async def test_mcp_state_pull_warns_and_rejects_deep_json(monkeypatch):
    depth = 250
    raw = b'{"payload":' + b"[" * depth + b"0" + b"]" * depth + b"}"

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw)

    monkeypatch.setenv("VF_STATE_URL", "http://state.test/state")
    monkeypatch.setenv("VF_STATE_SECRET", _SECRET)
    warning = MagicMock()
    monkeypatch.setattr(mcp_server.logger, "warning", warning)
    toolset = DeepToolset(ToolsetConfig())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        toolset._state_client = client
        with pytest.raises(ValidationError, match="recursion limit exceeded"):
            await toolset._pull_state()

    warning.assert_called_once()
    assert warning.call_args.args[:2] == (
        "state pull rejected for %s: %s",
        "DeepState",
    )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"payload":',
        b'{"payload":"not a list"}',
        b'{"payload":' + b"[" * 100_000 + b"0" + b"]" * 100_000 + b"}",
    ],
    ids=["malformed", "invalid-state", "too-deep"],
)
async def test_state_put_rejects_invalid_json_cleanly(raw: bytes):
    server = _interception_server(DeepState())

    response = await server.handle_state_put(_request(raw))

    assert response.status == 400
    assert json.loads(response.body)["error"].startswith(
        "invalid state PUT for DeepState:"
    )
    assert server.sessions[_SECRET].trace.state == DeepState()
