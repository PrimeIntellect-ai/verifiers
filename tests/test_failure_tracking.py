"""
Unit tests that verify error tracking behavior in NewSandboxEnv.

These tests simulate sandbox failures and verify that:
1. Errors are properly recorded with rollout_id
2. Resource states are correctly tracked
3. Errors can be queried by rollout
4. Infrastructure errors are distinguishable from model errors
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verifiers.envs.experimental.resource_managers.base import (
    ManagedResource,
    ResourceError,
    ResourceManager,
    ResourceState,
)
from verifiers.envs.experimental.resource_managers.sandbox_manager import (
    ManagedSandbox,
    SandboxManager,
)
from verifiers.envs.experimental.resource_managers.retry import RetryConfig
from verifiers.envs.experimental.resource_managers.errors import (
    SandboxCreationError,
    SandboxNotReadyError,
)


class TestResourceManagerErrorTracking:
    """Tests for ResourceManager error tracking."""

    @pytest.fixture
    def manager(self):
        """Create a concrete ResourceManager subclass for testing."""

        class TestManager(ResourceManager[ManagedResource]):
            def __init__(self):
                super().__init__(enable_health_monitoring=False)
                self.create_should_fail = False
                self.destroy_should_fail = False
                self.health_check_result = True
                self._counter = 0

            def create_resource_object(self, rollout_id):
                self._counter += 1
                return ManagedResource(id=f"resource-{self._counter}", rollout_id=rollout_id)

            async def create_resource(self, resource):
                if self.create_should_fail:
                    raise Exception("Simulated creation failure")
                # Normally would create actual resource

            async def destroy_resource(self, resource_id):
                if self.destroy_should_fail:
                    raise Exception("Simulated destroy failure")

            async def _check_health_impl(self, resource_id):
                return self.health_check_result

        return TestManager()

    @pytest.mark.asyncio
    async def test_successful_acquire_records_no_errors(self, manager):
        """Successful resource acquisition should not record errors."""
        resource = await manager.acquire(rollout_id="rollout-1")

        assert resource.state == ResourceState.READY
        assert resource.rollout_id == "rollout-1"
        assert resource.error is None
        assert len(manager.get_all_errors()) == 0

    @pytest.mark.asyncio
    async def test_failed_acquire_records_error(self, manager):
        """Failed resource acquisition should record error with rollout_id."""
        manager.create_should_fail = True

        with pytest.raises(Exception, match="Simulated creation failure"):
            await manager.acquire(rollout_id="rollout-2")

        errors = manager.get_all_errors()
        assert len(errors) == 1

        error = errors[0]
        assert error.rollout_id == "rollout-2"
        assert error.phase == "create"
        assert "Simulated creation failure" in str(error.error)

    @pytest.mark.asyncio
    async def test_errors_queryable_by_rollout(self, manager):
        """Errors should be queryable by rollout_id."""
        # Create two resources, second one fails
        await manager.acquire(rollout_id="rollout-1")

        manager.create_should_fail = True
        with pytest.raises(Exception):
            await manager.acquire(rollout_id="rollout-2")

        # Query errors by rollout
        rollout_1_errors = manager.get_errors_for_rollout("rollout-1")
        rollout_2_errors = manager.get_errors_for_rollout("rollout-2")

        assert len(rollout_1_errors) == 0
        assert len(rollout_2_errors) == 1

    @pytest.mark.asyncio
    async def test_resource_state_tracked(self, manager):
        """Resource state should be tracked correctly."""
        resource = await manager.acquire(rollout_id="rollout-1")
        assert resource.state == ResourceState.READY

        # Simulate error
        resource.mark_error(Exception("Test error"))
        assert resource.state == ResourceState.ERROR
        assert resource.error is not None

    @pytest.mark.asyncio
    async def test_multiple_errors_tracked(self, manager):
        """Multiple errors should all be tracked."""
        manager.create_should_fail = True

        for i in range(3):
            with pytest.raises(Exception):
                await manager.acquire(rollout_id=f"rollout-{i}")

        errors = manager.get_all_errors()
        assert len(errors) == 3

        # Each error should have correct rollout_id
        rollout_ids = {e.rollout_id for e in errors}
        assert rollout_ids == {"rollout-0", "rollout-1", "rollout-2"}


class TestSandboxManagerErrorTracking:
    """Tests for SandboxManager-specific error tracking."""

    @pytest.fixture
    def sandbox_manager(self):
        """Create SandboxManager with mocked client."""
        with patch(
            "verifiers.envs.experimental.resource_managers.sandbox_manager.ThreadedAsyncSandboxClient"
        ) as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            # Create a mock default request
            mock_request = MagicMock()
            mock_request.name = "test-sandbox"

            manager = SandboxManager(
                default_request=mock_request,
                timeout_per_command=30,
                enable_health_monitoring=False,
                retry_config=RetryConfig(max_attempts=1, initial_delay=0.01),
            )

            return manager, mock_client

    @pytest.mark.asyncio
    async def test_creation_failure_records_error(self, sandbox_manager):
        """Sandbox creation failure should record error."""
        manager, mock_client = sandbox_manager

        # Make create fail
        mock_client.create = AsyncMock(side_effect=Exception("API error"))

        with pytest.raises(SandboxCreationError):
            await manager.acquire(rollout_id="rollout-1")

        errors = manager.get_all_errors()
        assert len(errors) == 1
        assert errors[0].rollout_id == "rollout-1"
        assert errors[0].phase == "create"

    @pytest.mark.asyncio
    async def test_ready_failure_records_error(self, sandbox_manager):
        """Sandbox ready failure should record error in failure_info."""
        manager, mock_client = sandbox_manager

        # Make create succeed but ready fail
        mock_sandbox = MagicMock()
        mock_sandbox.id = "sandbox-123"
        mock_client.create = AsyncMock(return_value=mock_sandbox)
        mock_client.wait_for_creation = AsyncMock(
            side_effect=Exception("Timeout waiting for ready")
        )

        # First acquire succeeds
        sandbox = await manager.acquire(rollout_id="rollout-1")

        # Then wait_for_ready fails
        with pytest.raises(SandboxNotReadyError):
            await manager.wait_for_ready(sandbox.id)

        # Failure tracked in sandbox.failure_info
        assert sandbox.failure_info.not_ready is True
        assert "Timeout waiting for ready" in sandbox.failure_info.error_message

    @pytest.mark.asyncio
    async def test_execute_failure_records_error(self, sandbox_manager):
        """Command execution OOM failure should record error in failure_info."""
        from prime_sandboxes import SandboxOOMError as PrimeOOMError
        from verifiers.envs.experimental.resource_managers.errors import SandboxOOMError

        manager, mock_client = sandbox_manager

        # Make create and ready succeed
        mock_sandbox = MagicMock()
        mock_sandbox.id = "sandbox-123"
        mock_client.create = AsyncMock(return_value=mock_sandbox)
        mock_client.wait_for_creation = AsyncMock()

        sandbox = await manager.acquire(rollout_id="rollout-1")

        # Set up OOM error after acquire
        mock_client.execute_command = AsyncMock(
            side_effect=PrimeOOMError(sandbox_id=sandbox.id)
        )

        with pytest.raises(SandboxOOMError):
            await manager.execute_command(sandbox.id, "echo test")

        # Failure tracked in sandbox.failure_info
        assert sandbox.failure_info.oom is True


class TestFailureScenarios:
    """Tests for realistic failure scenarios."""

    @pytest.fixture
    def test_manager(self):
        """Create a ResourceManager that can simulate various failures."""

        class ScenarioManager(ResourceManager[ManagedResource]):
            def __init__(self):
                super().__init__(
                    enable_health_monitoring=False,
                    retry_config=RetryConfig(max_attempts=1, initial_delay=0.01),
                )
                self.failure_scenario = None
                self.created_count = 0
                self._counter = 0

            def create_resource_object(self, rollout_id):
                self._counter += 1
                return ManagedResource(id=f"resource-{self._counter}", rollout_id=rollout_id)

            async def create_resource(self, resource):
                self.created_count += 1
                scenario = self.failure_scenario

                if scenario == "fail_every_other":
                    if self.created_count % 2 == 0:
                        raise Exception("Simulated intermittent failure")
                elif scenario == "fail_all":
                    raise Exception("All creations fail")

            async def destroy_resource(self, resource_id):
                pass

            async def _check_health_impl(self, resource_id):
                return True

        return ScenarioManager()

    @pytest.mark.asyncio
    async def test_intermittent_failures_tracked(self, test_manager):
        """Intermittent failures should be properly tracked."""
        test_manager.failure_scenario = "fail_every_other"

        results = {"success": 0, "failed": 0}
        for i in range(6):
            try:
                await test_manager.acquire(rollout_id=f"rollout-{i}")
                results["success"] += 1
            except Exception:
                results["failed"] += 1

        # 3 succeed (odd numbers: 1, 3, 5), 3 fail (even numbers: 2, 4, 6)
        assert results["success"] == 3
        assert results["failed"] == 3

        errors = test_manager.get_all_errors()
        assert len(errors) == 3

        # Verify we can identify which rollouts failed
        failed_rollouts = {e.rollout_id for e in errors}
        assert len(failed_rollouts) == 3

    @pytest.mark.asyncio
    async def test_all_failures_scenario(self, test_manager):
        """When all operations fail, all errors should be tracked."""
        test_manager.failure_scenario = "fail_all"

        for i in range(5):
            with pytest.raises(Exception):
                await test_manager.acquire(rollout_id=f"rollout-{i}")

        errors = test_manager.get_all_errors()
        assert len(errors) == 5

        # All rollouts should have recorded errors
        for i in range(5):
            rollout_errors = test_manager.get_errors_for_rollout(f"rollout-{i}")
            assert len(rollout_errors) == 1

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self, test_manager):
        """Mix of success and failure should be properly distinguishable."""
        # First two succeed
        r1 = await test_manager.acquire(rollout_id="rollout-1")
        r2 = await test_manager.acquire(rollout_id="rollout-2")

        # Then enable failures
        test_manager.failure_scenario = "fail_all"

        with pytest.raises(Exception):
            await test_manager.acquire(rollout_id="rollout-3")
        with pytest.raises(Exception):
            await test_manager.acquire(rollout_id="rollout-4")

        # Check states
        assert r1.state == ResourceState.READY
        assert r2.state == ResourceState.READY

        # Check errors
        assert len(test_manager.get_errors_for_rollout("rollout-1")) == 0
        assert len(test_manager.get_errors_for_rollout("rollout-2")) == 0
        assert len(test_manager.get_errors_for_rollout("rollout-3")) == 1
        assert len(test_manager.get_errors_for_rollout("rollout-4")) == 1


class TestRewardFunctionIntegration:
    """Tests showing how reward functions can use error tracking."""

    def test_reward_can_check_for_infra_errors(self):
        """Reward function can filter out infrastructure errors."""
        # Simulate the scenario from a reward function's perspective
        errors = [
            ResourceError(
                resource_id="sb-1",
                rollout_id="rollout-1",
                error=Exception("Connection timeout"),
                phase="execute",
                timestamp=1234567890.0,
            )
        ]

        def should_compute_reward(rollout_id, errors_list):
            """Helper to check if we should compute reward for this rollout."""
            rollout_errors = [e for e in errors_list if e.rollout_id == rollout_id]
            return len(rollout_errors) == 0

        # Rollout 1 had errors - don't compute reward
        assert not should_compute_reward("rollout-1", errors)

        # Rollout 2 had no errors - compute reward
        assert should_compute_reward("rollout-2", errors)

    def test_can_distinguish_error_phases(self):
        """Different error phases can be distinguished."""
        errors = [
            ResourceError(
                resource_id="sb-1",
                rollout_id="rollout-1",
                error=Exception("API error"),
                phase="create",
                timestamp=1.0,
            ),
            ResourceError(
                resource_id="sb-2",
                rollout_id="rollout-2",
                error=Exception("Timeout"),
                phase="ready",
                timestamp=2.0,
            ),
            ResourceError(
                resource_id="sb-3",
                rollout_id="rollout-3",
                error=Exception("Connection reset"),
                phase="execute",
                timestamp=3.0,
            ),
        ]

        create_errors = [e for e in errors if e.phase == "create"]
        ready_errors = [e for e in errors if e.phase == "ready"]
        execute_errors = [e for e in errors if e.phase == "execute"]

        assert len(create_errors) == 1
        assert len(ready_errors) == 1
        assert len(execute_errors) == 1

        # All are infrastructure errors, none are model errors
        infra_phases = {"create", "ready", "execute", "health_check", "destroy"}
        for e in errors:
            assert e.phase in infra_phases


class TestResultsFiltering:
    """Tests for filtering evaluation results to exclude infrastructure errors."""

    def test_filter_results_excludes_error_rollouts(self):
        """filter_results should exclude rollouts with infrastructure errors."""
        # Create mock results
        results = {
            "outputs": [
                {"state": {"trajectory_id": "rollout-1", "reward": 1.0}},
                {"state": {"trajectory_id": "rollout-2", "reward": 0.5}},
                {"state": {"trajectory_id": "rollout-3", "reward": 0.8}},
            ],
            "mean_reward": 0.77,
        }

        # Simulate manager with error for rollout-2
        class MockManager:
            def get_all_errors(self):
                return [
                    ResourceError(
                        resource_id="sb-2",
                        rollout_id="rollout-2",
                        error=Exception("Connection died"),
                        phase="execute",
                        timestamp=1.0,
                    )
                ]

            def get_errors_for_rollout(self, rollout_id):
                if rollout_id == "rollout-2":
                    return self.get_all_errors()
                return []

        # Create a minimal mock env with filter_results
        class MockEnv:
            def __init__(self):
                self._sandbox_manager = MockManager()
                self.logger = MagicMock()

            def get_rollouts_with_errors(self):
                errors = self._sandbox_manager.get_all_errors()
                return {e.rollout_id for e in errors if e.rollout_id is not None}

            def filter_results(self, results):
                if "outputs" not in results:
                    return results

                error_rollouts = self.get_rollouts_with_errors()
                original_outputs = results["outputs"]
                filtered_outputs = []
                excluded_ids = []

                for output in original_outputs:
                    state = output.get("state", {})
                    rollout_id = state.get("trajectory_id")

                    if rollout_id in error_rollouts:
                        excluded_ids.append(rollout_id)
                    else:
                        filtered_outputs.append(output)

                filtered_results = dict(results)
                filtered_results["outputs"] = filtered_outputs

                if filtered_outputs:
                    rewards = [
                        o.get("state", {}).get("reward", 0.0)
                        for o in filtered_outputs
                        if o.get("state", {}).get("reward") is not None
                    ]
                    if rewards:
                        filtered_results["mean_reward"] = sum(rewards) / len(rewards)

                filtered_results["filtered"] = {
                    "total_rollouts": len(original_outputs),
                    "excluded_rollouts": len(excluded_ids),
                    "excluded_rollout_ids": excluded_ids,
                    "clean_rollouts": len(filtered_outputs),
                }

                return filtered_results

        env = MockEnv()
        filtered = env.filter_results(results)

        # Check that rollout-2 was excluded
        assert filtered["filtered"]["excluded_rollouts"] == 1
        assert filtered["filtered"]["excluded_rollout_ids"] == ["rollout-2"]
        assert filtered["filtered"]["clean_rollouts"] == 2

        # Check outputs only contain clean rollouts
        assert len(filtered["outputs"]) == 2
        rollout_ids = [o["state"]["trajectory_id"] for o in filtered["outputs"]]
        assert "rollout-1" in rollout_ids
        assert "rollout-3" in rollout_ids
        assert "rollout-2" not in rollout_ids

        # Check mean_reward recalculated (1.0 + 0.8) / 2 = 0.9
        assert filtered["mean_reward"] == 0.9

    def test_filter_results_no_errors(self):
        """filter_results with no errors should return all rollouts."""
        results = {
            "outputs": [
                {"state": {"trajectory_id": "rollout-1", "reward": 1.0}},
                {"state": {"trajectory_id": "rollout-2", "reward": 0.5}},
            ],
        }

        class MockManager:
            def get_all_errors(self):
                return []

        class MockEnv:
            def __init__(self):
                self._sandbox_manager = MockManager()
                self.logger = MagicMock()

            def get_rollouts_with_errors(self):
                return set()

            def filter_results(self, results):
                if "outputs" not in results:
                    return results
                results["filtered"] = {
                    "total_rollouts": len(results["outputs"]),
                    "excluded_rollouts": 0,
                    "excluded_rollout_ids": [],
                    "clean_rollouts": len(results["outputs"]),
                }
                return results

        env = MockEnv()
        filtered = env.filter_results(results)

        assert filtered["filtered"]["excluded_rollouts"] == 0
        assert filtered["filtered"]["clean_rollouts"] == 2
        assert len(filtered["outputs"]) == 2
