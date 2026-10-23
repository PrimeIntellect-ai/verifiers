"""A pool's `order` decides which deployment a session takes: "priority" fills the
first declared deployment and overflows only when its gate does not admit; "shuffle"
spreads sessions over every deployment."""

import pytest

from verifiers.v1 import model_routing
from verifiers.v1.configs.model_pool import ModelPoolConfig


class FakeGate:
    def __init__(self, room):
        self.room = room

    def try_admit(self, config):
        if self.room <= 0:
            return None
        self.room -= 1
        return 0


class Dialect:
    def auth_headers(self, key):
        return {}


def _pool(order, *models):
    return ModelPoolConfig.model_validate(
        {
            "order": order,
            "deployments": [
                {
                    "id": model,
                    "model": model,
                    "client": {
                        "type": "eval",
                        "base_url": f"http://{model}.test/v1",
                        "request_gate": {"max_concurrent": 1},
                    },
                }
                for model in models
            ],
        }
    )


@pytest.fixture
def gates(monkeypatch, tmp_path):
    rooms = {}
    made = {}

    def model_gate(directory, key, model):
        return made.setdefault(model, FakeGate(rooms[model]))

    monkeypatch.setattr(model_routing, "model_gate", model_gate)
    monkeypatch.setattr(model_routing, "gate_dir", lambda config: tmp_path)
    monkeypatch.setattr(model_routing.RequestPermit, "__init__", lambda *a: None)
    return rooms


async def _models(pool, n):
    return [(await model_routing._choose(pool, Dialect()))[0].model for _ in range(n)]


async def test_priority_fills_the_first_deployment_then_overflows(gates):
    gates.update({"glm-fast": 3, "deepseek": 10})
    pool = _pool("priority", "glm-fast", "deepseek")
    assert await _models(pool, 5) == ["glm-fast"] * 3 + ["deepseek"] * 2


async def test_shuffle_is_the_default_and_spreads_sessions(gates):
    gates.update({"a": 100, "b": 100})
    pool = _pool("shuffle", "a", "b")
    assert (
        ModelPoolConfig.model_validate(
            {"deployments": pool.model_dump()["deployments"]}
        ).order
        == "shuffle"
    )
    assert set(await _models(pool, 40)) == {"a", "b"}


def test_an_unknown_order_is_refused():
    with pytest.raises(ValueError):
        _pool("round_robin", "a")
