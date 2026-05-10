"""Tests for verifiers.integrations.braintrust."""

from __future__ import annotations

import sys
import types

import pytest


# -- Mock braintrust --------------------------------------------------------


def _mock_bt():
    mod = types.ModuleType("braintrust")
    mod._root: list = []
    mod._stack: list = []

    class Span:
        def __init__(self, name="", type=""):
            self.name, self.type = name, type
            self.logs: list[dict] = []
            self.children: list[Span] = []
            self._ended = False

        def log(self, **kw):
            self.logs.append(kw)

        def start_span(self, name="", type=""):
            c = Span(name, type)
            self.children.append(c)
            return c

        def end(self):
            self._ended = True

        def __enter__(self):
            mod._stack.append(self)
            return self

        def __exit__(self, *a):
            if mod._stack and mod._stack[-1] is self:
                mod._stack.pop()
            self._ended = True

    mod.Span = Span
    mod.init_logger = lambda project="", api_key=None: setattr(mod, "_project", project)
    mod.start_span = lambda name="", type="": (
        lambda s: (
            mod._stack[-1].children.append(s) if mod._stack else mod._root.append(s),
            s,
        )[-1]
    )(Span(name, type))
    mod.current_span = lambda: mod._stack[-1] if mod._stack else Span("_none")
    mod.flush = lambda: None
    return mod


@pytest.fixture(autouse=True)
def bt(monkeypatch):
    m = _mock_bt()
    monkeypatch.setitem(sys.modules, "braintrust", m)
    import verifiers.integrations.braintrust as mod

    monkeypatch.setattr(mod, "_bt", None)
    mod._ctx.clear()
    yield m
    mod._ctx.clear()


# -- Fakes ------------------------------------------------------------------


class S(dict):
    """Fake State."""


class T(dict):
    """Fake Task."""


class Runtime:
    def __init__(self):
        self._n = 0

    async def submit_model_request(
        self, prompt, task, state, tool_defs=None, extras=None
    ):
        self._n += 1
        state.setdefault("trajectory", []).append(
            {
                "tokens": {
                    "prompt_tokens": 500 + self._n * 100,
                    "completion_tokens": 50,
                    "cache_read_input_tokens": 0,
                },
            }
        )
        return object()

    async def call_tool(self, name, task, state, **kw):
        return f"{name} done"


class Harness:
    def __init__(self):
        self.runtime = Runtime()

    async def setup_state(self, task, state):
        state.setdefault("trajectory", [])
        state.setdefault("runtime", {})
        return state

    async def run(self, task, state=None):
        state = state if state is not None else S()
        state = await self.setup_state(task, state)
        # 4-turn agent loop matching mini-browse-env screenshot.
        await self.runtime.submit_model_request([], task, state)
        await self.runtime.call_tool("find", task, state)
        await self.runtime.submit_model_request([], task, state)
        await self.runtime.call_tool("navigate", task, state)
        await self.runtime.submit_model_request([], task, state)
        await self.runtime.call_tool("submit_result", task, state)
        state["reward"] = 1.0
        state["metrics"] = {"accuracy": 1.0}
        state["is_completed"] = True
        state["stop_condition"] = "program_completed"
        return state


class Env:
    def __init__(self, harness=None):
        self.harness = harness or Harness()


# -- Trace tree structure ---------------------------------------------------


class TestTraceTree:
    @pytest.mark.asyncio
    async def test_full_hierarchy(self, bt):
        from verifiers.integrations.braintrust import instrument, traced_group

        env = Env()
        instrument(env, project="p")
        with traced_group("p"):
            await env.harness.run(T())

        group = bt._root[0]
        rollout = group.children[0]
        names = [c.name for c in rollout.children]
        assert names == ["setup_state", "turn_0", "turn_1", "turn_2", "turn_3"]

    @pytest.mark.asyncio
    async def test_turn_0(self, bt):
        from verifiers.integrations.braintrust import instrument

        env = Env()
        instrument(env, project="p")
        await env.harness.run(T())
        turn = bt._root[0].children[1]
        assert [c.name for c in turn.children] == ["model_request"]

    @pytest.mark.asyncio
    async def test_turn_1(self, bt):
        from verifiers.integrations.braintrust import instrument

        env = Env()
        instrument(env, project="p")
        await env.harness.run(T())
        turn = bt._root[0].children[2]
        assert [c.name for c in turn.children] == ["tool_call:find", "model_request"]

    @pytest.mark.asyncio
    async def test_turn_2(self, bt):
        from verifiers.integrations.braintrust import instrument

        env = Env()
        instrument(env, project="p")
        await env.harness.run(T())
        turn = bt._root[0].children[3]
        assert [c.name for c in turn.children] == [
            "tool_call:navigate",
            "model_request",
        ]

    @pytest.mark.asyncio
    async def test_turn_3_final_tool(self, bt):
        from verifiers.integrations.braintrust import instrument

        env = Env()
        instrument(env, project="p")
        await env.harness.run(T())
        turn = bt._root[0].children[4]
        assert [c.name for c in turn.children] == ["tool_call:submit_result"]
        assert turn._ended


# -- Scores & metadata on rollout span --------------------------------------


class TestRolloutData:
    @pytest.mark.asyncio
    async def test_scores(self, bt):
        from verifiers.integrations.braintrust import instrument

        env = Env()
        instrument(env, project="p")
        await env.harness.run(T())
        scores = [e for e in bt._root[0].logs if "scores" in e][0]["scores"]
        assert scores["reward"] == 1.0
        assert scores["accuracy"] == 1.0

    @pytest.mark.asyncio
    async def test_metadata(self, bt):
        from verifiers.integrations.braintrust import instrument

        env = Env()
        instrument(env, project="p")
        await env.harness.run(T())
        meta = [e for e in bt._root[0].logs if "metadata" in e][0]["metadata"]
        assert meta["stop_condition"] == "program_completed"
        assert meta["is_completed"] is True
        assert meta["num_turns"] == 3
        assert meta["total_tokens"] > 0


# -- Per-span metrics -------------------------------------------------------


class TestSpanMetrics:
    @pytest.mark.asyncio
    async def test_model_request_tokens(self, bt):
        from verifiers.integrations.braintrust import instrument

        env = Env()
        instrument(env, project="p")
        await env.harness.run(T())
        mr = bt._root[0].children[1].children[0]
        m = mr.logs[0]["metrics"]
        assert m["tokens"] == m["prompt_tokens"] + m["completion_tokens"]

    @pytest.mark.asyncio
    async def test_model_request_timing(self, bt):
        from verifiers.integrations.braintrust import instrument

        env = Env()
        instrument(env, project="p")
        await env.harness.run(T())
        mr = bt._root[0].children[1].children[0]
        assert isinstance(mr.logs[0]["metadata"]["elapsed_s"], float)

    @pytest.mark.asyncio
    async def test_tool_call_timing(self, bt):
        from verifiers.integrations.braintrust import instrument

        env = Env()
        instrument(env, project="p")
        await env.harness.run(T())
        tc = bt._root[0].children[2].children[0]
        assert tc.name == "tool_call:find"
        assert isinstance(tc.logs[0]["metadata"]["elapsed_s"], float)


# -- traced_group -----------------------------------------------------------


class TestGroup:
    @pytest.mark.asyncio
    async def test_nests_rollouts(self, bt):
        from verifiers.integrations.braintrust import instrument, traced_group

        env = Env()
        instrument(env, project="p")
        with traced_group("p"):
            await env.harness.run(T())
            await env.harness.run(T())
        assert len(bt._root) == 1
        assert [c.name for c in bt._root[0].children] == ["rollout", "rollout"]

    @pytest.mark.asyncio
    async def test_without_group(self, bt):
        from verifiers.integrations.braintrust import instrument

        env = Env()
        instrument(env, project="p")
        await env.harness.run(T())
        assert bt._root[0].name == "rollout"

    @pytest.mark.asyncio
    async def test_flushes(self, bt):
        from verifiers.integrations.braintrust import traced_group

        calls = []
        bt.flush = lambda: calls.append(1)
        with traced_group("p"):
            pass
        assert calls == [1]


# -- Edge cases -------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_no_tools(self, bt):
        from verifiers.integrations.braintrust import instrument

        class H(Harness):
            async def run(self, task, state=None):
                state = state if state is not None else S()
                state = await self.setup_state(task, state)
                await self.runtime.submit_model_request([], task, state)
                await self.runtime.submit_model_request([], task, state)
                state["reward"] = 0.5
                state["is_completed"] = True
                state["stop_condition"] = "done"
                return state

        env = Env(H())
        instrument(env, project="p")
        await env.harness.run(T())
        names = [c.name for c in bt._root[0].children]
        assert names == ["setup_state", "turn_0", "turn_1"]

    @pytest.mark.asyncio
    async def test_submit_error_closes_spans(self, bt):
        from verifiers.integrations.braintrust import instrument

        class R(Runtime):
            async def submit_model_request(self, *a, **k):
                raise RuntimeError("boom")

        class H(Harness):
            def __init__(self):
                super().__init__()
                self.runtime = R()

            async def run(self, task, state=None):
                state = state if state is not None else S()
                state = await self.setup_state(task, state)
                try:
                    await self.runtime.submit_model_request([], task, state)
                except RuntimeError:
                    pass
                state["reward"] = 0.0
                state["is_completed"] = True
                state["stop_condition"] = "error"
                return state

        env = Env(H())
        instrument(env, project="p")
        await env.harness.run(T())
        turn = bt._root[0].children[1]
        assert turn.children[0]._ended and turn._ended

    @pytest.mark.asyncio
    async def test_tool_error_closes_span(self, bt):
        from verifiers.integrations.braintrust import instrument

        class R(Runtime):
            async def call_tool(self, *a, **k):
                raise RuntimeError("boom")

        class H(Harness):
            def __init__(self):
                super().__init__()
                self.runtime = R()

            async def run(self, task, state=None):
                state = state if state is not None else S()
                state = await self.setup_state(task, state)
                await self.runtime.submit_model_request([], task, state)
                try:
                    await self.runtime.call_tool("x", task, state)
                except RuntimeError:
                    pass
                state["reward"] = 0.0
                state["is_completed"] = True
                state["stop_condition"] = "error"
                return state

        env = Env(H())
        instrument(env, project="p")
        await env.harness.run(T())
        tc = bt._root[0].children[2].children[0]
        assert tc.name == "tool_call:x" and tc._ended

    @pytest.mark.asyncio
    async def test_uninstrumented(self, bt):
        state = await Env().harness.run(T())
        assert state["is_completed"] is True
        assert bt._root == []

    @pytest.mark.asyncio
    async def test_sequential_isolation(self, bt):
        from verifiers.integrations.braintrust import instrument

        env = Env()
        instrument(env, project="p")
        await env.harness.run(T())
        await env.harness.run(T())
        assert len(bt._root) == 2
        for r in bt._root:
            assert [c.name for c in r.children] == [
                "setup_state",
                "turn_0",
                "turn_1",
                "turn_2",
                "turn_3",
            ]

    @pytest.mark.asyncio
    async def test_no_trajectory_step(self, bt):
        from verifiers.integrations.braintrust import instrument

        class R(Runtime):
            async def submit_model_request(self, *a, **k):
                return object()

        class H(Harness):
            def __init__(self):
                super().__init__()
                self.runtime = R()

            async def run(self, task, state=None):
                state = state if state is not None else S()
                state = await self.setup_state(task, state)
                await self.runtime.submit_model_request([], task, state)
                state["reward"] = 0.0
                state["is_completed"] = True
                state["stop_condition"] = "done"
                return state

        env = Env(H())
        instrument(env, project="p")
        await env.harness.run(T())
        mr = bt._root[0].children[1].children[0]
        assert mr.logs[0]["metrics"] == {}

    @pytest.mark.asyncio
    async def test_scores_no_metrics(self, bt):
        from verifiers.integrations.braintrust import instrument

        class H(Harness):
            async def run(self, task, state=None):
                state = state if state is not None else S()
                state = await self.setup_state(task, state)
                await self.runtime.submit_model_request([], task, state)
                state["reward"] = 0.75
                state["is_completed"] = True
                state["stop_condition"] = "done"
                return state

        env = Env(H())
        instrument(env, project="p")
        await env.harness.run(T())
        scores = [e for e in bt._root[0].logs if "scores" in e][0]["scores"]
        assert scores == {"reward": 0.75}


# -- Import safety ----------------------------------------------------------


class TestImport:
    def test_no_braintrust(self, monkeypatch):
        import verifiers.integrations.braintrust as mod

        monkeypatch.setattr(mod, "_bt", None)
        monkeypatch.delitem(sys.modules, "braintrust", raising=False); import builtins; real_import = builtins.__import__; monkeypatch.setattr(builtins, "__import__", lambda name, *a, **k: (_ for _ in ()).throw(ModuleNotFoundError("blocked")) if name == "braintrust" else real_import(name, *a, **k))
        with pytest.raises(ModuleNotFoundError, match="pip install"):
            mod._bt_or_raise()

    def test_module_importable(self, monkeypatch):
        import verifiers.integrations.braintrust as mod

        monkeypatch.setattr(mod, "_bt", None)
        monkeypatch.delitem(sys.modules, "braintrust", raising=False); import builtins; real_import = builtins.__import__; monkeypatch.setattr(builtins, "__import__", lambda name, *a, **k: (_ for _ in ()).throw(ModuleNotFoundError("blocked")) if name == "braintrust" else real_import(name, *a, **k))
        import importlib

        importlib.reload(mod)
