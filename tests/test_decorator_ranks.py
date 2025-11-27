"""Tests for decorator priority ordering functionality."""

import pytest
from datasets import Dataset

import verifiers as vf
from verifiers.types import RolloutInput, State


class RankedStopEnv(vf.MultiTurnEnv):
    """Test environment with prioritized stop conditions."""

    def __init__(self, **kwargs):
        super().__init__(max_turns=2, **kwargs)
        self.stop_order = []
        self.turn_count = 0

    @vf.stop(priority=1)
    async def early_stop(self, state: State) -> bool:
        """Early stop condition."""
        self.stop_order.append("early_stop")
        return False

    @vf.stop(priority=0)
    async def default_stop(self, state: State) -> bool:
        """Default stop condition."""
        self.stop_order.append("default_stop")
        return False

    @vf.stop(priority=-1)
    async def late_stop(self, state: State) -> bool:
        """Late stop condition."""
        self.stop_order.append("late_stop")
        self.turn_count += 1
        return self.turn_count >= 2

    @vf.stop(priority=0)
    async def another_default(self, state: State) -> bool:
        """Another default priority stop condition."""
        self.stop_order.append("another_default")
        return False

    async def env_response(self, messages, state, **kwargs):
        return []


class RankedCleanupEnv(vf.MultiTurnEnv):
    """Test environment with prioritized cleanup handlers."""

    def __init__(self, **kwargs):
        super().__init__(max_turns=1, **kwargs)
        self.cleanup_order = []

    @vf.cleanup(priority=1)
    async def early_cleanup(self, state: State):
        """Early cleanup handler."""
        self.cleanup_order.append("early_cleanup")

    @vf.cleanup(priority=0)
    async def default_cleanup(self, state: State):
        """Default cleanup handler."""
        self.cleanup_order.append("default_cleanup")

    @vf.cleanup(priority=-1)
    async def late_cleanup(self, state: State):
        """Late cleanup handler."""
        self.cleanup_order.append("late_cleanup")

    @vf.cleanup(priority=0)
    async def another_default_cleanup(self, state: State):
        """Another default priority cleanup handler."""
        self.cleanup_order.append("another_default_cleanup")

    async def env_response(self, messages, state, **kwargs):
        return []


class RankedTeardownEnv(vf.MultiTurnEnv):
    """Test environment with prioritized teardown handlers."""

    def __init__(self, **kwargs):
        super().__init__(max_turns=1, **kwargs)
        self.teardown_order = []

    @vf.teardown(priority=1)
    async def early_teardown(self):
        """Early teardown handler."""
        self.teardown_order.append("early_teardown")

    @vf.teardown(priority=0)
    async def default_teardown(self):
        """Default teardown handler."""
        self.teardown_order.append("default_teardown")

    @vf.teardown(priority=-1)
    async def late_teardown(self):
        """Late teardown handler."""
        self.teardown_order.append("late_teardown")

    @vf.teardown(priority=0)
    async def another_default_teardown(self):
        """Another default priority teardown handler."""
        self.teardown_order.append("another_default_teardown")

    async def env_response(self, messages, state, **kwargs):
        return []


class TestStopPriorityOrdering:
    """Test stop condition priority ordering."""

    def _get_class_methods(self, env, methods, class_name):
        """Filter methods to only those defined in the specified class."""
        import inspect

        return [
            method
            for method in methods
            if getattr(method, "__self__", None)
            and inspect.getmodule(getattr(method, "__func__", None))
            == inspect.getmodule(env)
            and (
                getattr(
                    getattr(method, "__func__", None), "__qualname__", ""
                ).startswith(f"{class_name}.")
                or f".{class_name}."
                in getattr(getattr(method, "__func__", None), "__qualname__", "")
            )
        ]

    def test_stop_conditions_sorted_by_priority(self, mock_openai_client):
        """Test that stop conditions are sorted by priority."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RankedStopEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        class_stop_conditions = self._get_class_methods(
            env, env._stop_conditions, "RankedStopEnv"
        )
        assert len(class_stop_conditions) == 4
        names = [method.__name__ for method in class_stop_conditions]
        assert names == ["early_stop", "another_default", "default_stop", "late_stop"]

    def test_stop_conditions_tie_breaking_alphabetical(self, mock_openai_client):
        """Test that stop conditions with same priority are sorted alphabetically."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RankedStopEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        class_stop_conditions = self._get_class_methods(
            env, env._stop_conditions, "RankedStopEnv"
        )
        default_priority_conditions = [
            method
            for method in class_stop_conditions
            if getattr(method, "stop_priority", 0) == 0
        ]
        assert len(default_priority_conditions) == 2
        names = [method.__name__ for method in default_priority_conditions]
        assert names == ["another_default", "default_stop"]

    @pytest.mark.asyncio
    async def test_stop_conditions_execution_order(self, mock_openai_client):
        """Test that stop conditions execute in priority order."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RankedStopEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        mock_openai_client.set_default_responses(chat_response="test")

        await env.rollout(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "test"}],
                answer="test",
                example_id=0,
                task="test",
            ),
            client=mock_openai_client,
            model="test-model",
        )

        assert len(env.stop_order) > 0
        class_stop_order = [
            name
            for name in env.stop_order
            if name in ["early_stop", "another_default", "default_stop", "late_stop"]
        ]
        assert len(class_stop_order) >= 4, (
            f"Expected at least 4 class stop condition calls, got {len(class_stop_order)}: {class_stop_order}"
        )
        assert class_stop_order[0] == "early_stop"
        assert "late_stop" in class_stop_order
        assert "another_default" in class_stop_order
        assert "default_stop" in class_stop_order
        first_occurrence_order = []
        seen = set()
        for name in class_stop_order:
            if name not in seen:
                first_occurrence_order.append(name)
                seen.add(name)
        assert len(first_occurrence_order) == 4
        assert first_occurrence_order[0] == "early_stop"
        assert first_occurrence_order[-1] == "late_stop"
        assert first_occurrence_order.index(
            "another_default"
        ) < first_occurrence_order.index("default_stop")


class TestCleanupPriorityOrdering:
    """Test cleanup handler priority ordering."""

    def test_cleanup_handlers_sorted_by_priority(self, mock_openai_client):
        """Test that cleanup handlers are sorted by priority."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RankedCleanupEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        assert len(env._cleanup_handlers) == 4
        names = [method.__name__ for method in env._cleanup_handlers]
        assert names == [
            "early_cleanup",
            "another_default_cleanup",
            "default_cleanup",
            "late_cleanup",
        ]

    def test_cleanup_handlers_tie_breaking_alphabetical(self, mock_openai_client):
        """Test that cleanup handlers with same priority are sorted alphabetically."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RankedCleanupEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        default_priority_handlers = [
            method
            for method in env._cleanup_handlers
            if getattr(method, "cleanup_priority", 0) == 0
        ]
        assert len(default_priority_handlers) == 2
        names = [method.__name__ for method in default_priority_handlers]
        assert names == ["another_default_cleanup", "default_cleanup"]

    @pytest.mark.asyncio
    async def test_cleanup_handlers_execution_order(self, mock_openai_client):
        """Test that cleanup handlers execute in priority order."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RankedCleanupEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        state = await env.init_state(
            RolloutInput(
                prompt=[{"role": "user", "content": "test"}],
                answer="test",
                example_id=0,
                task="test",
            ),
            client=mock_openai_client,
            model="test-model",
        )
        await env._cleanup(state)

        assert len(env.cleanup_order) == 4
        assert env.cleanup_order[0] == "early_cleanup"
        assert env.cleanup_order[-1] == "late_cleanup"
        assert "another_default_cleanup" in env.cleanup_order
        assert "default_cleanup" in env.cleanup_order
        assert env.cleanup_order.index(
            "another_default_cleanup"
        ) < env.cleanup_order.index("default_cleanup")


class TestTeardownPriorityOrdering:
    """Test teardown handler priority ordering."""

    def test_teardown_handlers_sorted_by_priority(self, mock_openai_client):
        """Test that teardown handlers are sorted by priority."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RankedTeardownEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        assert len(env._teardown_handlers) == 4
        names = [method.__name__ for method in env._teardown_handlers]
        assert names == [
            "early_teardown",
            "another_default_teardown",
            "default_teardown",
            "late_teardown",
        ]

    def test_teardown_handlers_tie_breaking_alphabetical(self, mock_openai_client):
        """Test that teardown handlers with same priority are sorted alphabetically."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RankedTeardownEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        default_priority_handlers = [
            method
            for method in env._teardown_handlers
            if getattr(method, "teardown_priority", 0) == 0
        ]
        assert len(default_priority_handlers) == 2
        names = [method.__name__ for method in default_priority_handlers]
        assert names == ["another_default_teardown", "default_teardown"]

    @pytest.mark.asyncio
    async def test_teardown_handlers_execution_order(self, mock_openai_client):
        """Test that teardown handlers execute in priority order."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RankedTeardownEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        await env._teardown()

        assert len(env.teardown_order) == 4
        assert env.teardown_order[0] == "early_teardown"
        assert env.teardown_order[-1] == "late_teardown"
        assert "another_default_teardown" in env.teardown_order
        assert "default_teardown" in env.teardown_order
        assert env.teardown_order.index(
            "another_default_teardown"
        ) < env.teardown_order.index("default_teardown")


class TestDecoratorPriorityEdgeCases:
    """Test edge cases for decorator priority functionality."""

    def test_decorator_without_priority_defaults_to_zero(self, mock_openai_client):
        """Test that decorators without priority parameter default to priority 0."""

        class DefaultPriorityEnv(vf.MultiTurnEnv):
            @vf.stop
            async def no_priority_stop(self, state: State) -> bool:
                return False

            @vf.cleanup
            async def no_priority_cleanup(self, state: State):
                pass

            @vf.teardown
            async def no_priority_teardown(self):
                pass

            async def env_response(self, messages, state, **kwargs):
                return []

        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = DefaultPriorityEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        import inspect

        def qualname_check(m, name):
            qualname = getattr(getattr(m, "__func__", None), "__qualname__", "")
            return qualname.startswith(f"{name}.") or f".{name}." in qualname

        class_stop = [
            m
            for m in env._stop_conditions
            if getattr(m, "__self__", None)
            and inspect.getmodule(getattr(m, "__func__", None))
            == inspect.getmodule(env)
            and qualname_check(m, "DefaultPriorityEnv")
        ][0]
        class_cleanup = [
            m
            for m in env._cleanup_handlers
            if getattr(m, "__self__", None)
            and inspect.getmodule(getattr(m, "__func__", None))
            == inspect.getmodule(env)
            and qualname_check(m, "DefaultPriorityEnv")
        ][0]
        class_teardown = [
            m
            for m in env._teardown_handlers
            if getattr(m, "__self__", None)
            and inspect.getmodule(getattr(m, "__func__", None))
            == inspect.getmodule(env)
            and qualname_check(m, "DefaultPriorityEnv")
        ][0]

        assert getattr(class_stop, "stop_priority", None) == 0
        assert getattr(class_cleanup, "cleanup_priority", None) == 0
        assert getattr(class_teardown, "teardown_priority", None) == 0

    def test_high_priorities_run_before_default(self, mock_openai_client):
        """Test that high priorities run before default priority 0."""

        class HighPriorityEnv(vf.MultiTurnEnv):
            def __init__(self, **kwargs):
                super().__init__(max_turns=1, **kwargs)
                self.order = []

            @vf.stop(priority=5)
            async def very_early(self, state: State) -> bool:
                self.order.append("very_early")
                return False

            @vf.stop(priority=1)
            async def early(self, state: State) -> bool:
                self.order.append("early")
                return False

            @vf.stop
            async def default(self, state: State) -> bool:
                self.order.append("default")
                return False

            async def env_response(self, messages, state, **kwargs):
                return []

        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = HighPriorityEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        import inspect

        def qualname_check(m, name):
            qualname = getattr(getattr(m, "__func__", None), "__qualname__", "")
            return qualname.startswith(f"{name}.") or f".{name}." in qualname

        class_stop_conditions = [
            method
            for method in env._stop_conditions
            if getattr(method, "__self__", None)
            and inspect.getmodule(getattr(method, "__func__", None))
            == inspect.getmodule(env)
            and qualname_check(method, "HighPriorityEnv")
        ]
        names = [method.__name__ for method in class_stop_conditions]
        assert names == ["very_early", "early", "default"]

    def test_low_priorities_run_after_default(self, mock_openai_client):
        """Test that low priorities run after default priority 0."""

        class LowPriorityEnv(vf.MultiTurnEnv):
            @vf.stop
            async def default(self, state: State) -> bool:
                return False

            @vf.stop(priority=-1)
            async def late(self, state: State) -> bool:
                return False

            @vf.stop(priority=-10)
            async def very_late(self, state: State) -> bool:
                return False

            async def env_response(self, messages, state, **kwargs):
                return []

        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = LowPriorityEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        import inspect

        def qualname_check(m, name):
            qualname = getattr(getattr(m, "__func__", None), "__qualname__", "")
            return qualname.startswith(f"{name}.") or f".{name}." in qualname

        class_stop_conditions = [
            method
            for method in env._stop_conditions
            if getattr(method, "__self__", None)
            and inspect.getmodule(getattr(method, "__func__", None))
            == inspect.getmodule(env)
            and qualname_check(method, "LowPriorityEnv")
        ]
        names = [method.__name__ for method in class_stop_conditions]
        assert names == ["default", "late", "very_late"]

    def test_mixed_priorities_complex_ordering(self, mock_openai_client):
        """Test complex ordering with mixed priorities."""

        class MixedPriorityEnv(vf.MultiTurnEnv):
            @vf.stop(priority=-5)
            async def z_late(self, state: State) -> bool:
                return False

            @vf.stop(priority=2)
            async def a_early(self, state: State) -> bool:
                return False

            @vf.stop(priority=0)
            async def m_default(self, state: State) -> bool:
                return False

            @vf.stop(priority=0)
            async def z_default(self, state: State) -> bool:
                return False

            @vf.stop(priority=1)
            async def b_early(self, state: State) -> bool:
                return False

            async def env_response(self, messages, state, **kwargs):
                return []

        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = MixedPriorityEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        import inspect

        def qualname_check(m, name):
            qualname = getattr(getattr(m, "__func__", None), "__qualname__", "")
            return qualname.startswith(f"{name}.") or f".{name}." in qualname

        class_stop_conditions = [
            method
            for method in env._stop_conditions
            if getattr(method, "__self__", None)
            and inspect.getmodule(getattr(method, "__func__", None))
            == inspect.getmodule(env)
            and qualname_check(method, "MixedPriorityEnv")
        ]
        names = [method.__name__ for method in class_stop_conditions]
        assert names == ["a_early", "b_early", "m_default", "z_default", "z_late"]
