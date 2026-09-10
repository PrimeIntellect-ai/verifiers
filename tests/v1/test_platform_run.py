"""`open_run`: the eval's run on the platform, and `run.attach` for hosted evaluations."""

import prime_runs as pr
import pytest

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.utils import platform
from verifiers.v1.utils.platform import PushState, open_run


class _FakeRun:
    """Stands in for `pr.Run`: an id, a URL, nothing else the tests reach for."""

    def __init__(self, run_id: str, mode: str) -> None:
        self.id = run_id
        self.mode = mode
        self.url = (
            None if mode == "disabled" else f"https://app.example/evaluations/{run_id}"
        )
        self.finished = False


@pytest.fixture
def init_calls(monkeypatch):
    """Record every `pr.init` call; the fake returns a run keyed by the attach id."""
    calls: list[dict] = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return _FakeRun(kwargs.get("id") or "eval-created", kwargs["mode"])

    monkeypatch.setattr(platform.pr, "init", fake_init)
    monkeypatch.delenv(pr.MODE_ENV, raising=False)
    return calls


def test_attach_needs_push():
    with pytest.raises(ValueError, match="needs push"):
        EvalConfig(run={"attach": "eval-hosted"}, push=False)


def test_a_run_without_attach_is_created_on_the_platform(init_calls):
    config = EvalConfig(push=True)
    state = PushState()

    run = open_run(config, state, num_examples=3)

    assert init_calls[0]["mode"] == "online"
    assert init_calls[0]["id"] is None
    assert run.id == config.run.id == "eval-created"


def test_attach_hands_the_launchers_id_to_the_sdk(init_calls):
    config = EvalConfig(run={"attach": "eval-hosted"})
    state = PushState()

    run = open_run(config, state, num_examples=3)

    assert init_calls[0]["mode"] == "online"
    assert init_calls[0]["id"] == "eval-hosted"
    # The launcher's id is the run's one id: traces and the run dir carry it too.
    assert run.id == config.run.id == "eval-hosted"
    assert state.error is None


def test_a_failed_attach_fails_the_eval_instead_of_falling_back(monkeypatch):
    def refuse(**kwargs):
        raise pr.UnauthorizedError("bad key")

    monkeypatch.setattr(platform.pr, "init", refuse)
    monkeypatch.delenv(pr.MODE_ENV, raising=False)
    config = EvalConfig(run={"attach": "eval-hosted"})

    with pytest.raises(RuntimeError, match="could not attach to run 'eval-hosted'"):
        open_run(config, PushState(), num_examples=3)


def test_a_failed_open_without_attach_still_falls_back_to_a_local_run(monkeypatch):
    calls: list[dict] = []

    def flaky(**kwargs):
        calls.append(kwargs)
        if kwargs["mode"] == "online":
            raise pr.UnauthorizedError("bad key")
        return _FakeRun("local-run", kwargs["mode"])

    monkeypatch.setattr(platform.pr, "init", flaky)
    monkeypatch.delenv(pr.MODE_ENV, raising=False)
    state = PushState()

    run = open_run(EvalConfig(push=True), state, num_examples=3)

    assert [call["mode"] for call in calls] == ["online", "disabled"]
    assert run.mode == "disabled"
    assert state.error is not None and "UnauthorizedError" in state.error


def test_the_sdk_kill_switch_refuses_an_attach(init_calls, monkeypatch):
    monkeypatch.setenv(pr.MODE_ENV, "disabled")
    config = EvalConfig(run={"attach": "eval-hosted"})

    with pytest.raises(RuntimeError, match="PRIME_RUNS_MODE=disabled"):
        open_run(config, PushState(), num_examples=3)
    assert init_calls == []
