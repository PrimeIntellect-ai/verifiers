from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from verifiers.envs.experimental.resource_managers.retry import RetryConfig


logger = logging.getLogger(__name__)


class ResourceState(Enum):
    CREATING = "creating"
    READY = "ready"
    ERROR = "error"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"


@dataclass(slots=True)
class ResourceError:
    """Captures error information for a resource operation."""
    resource_id: str
    rollout_id: str | None
    error: Exception
    timestamp: float
    phase: str  # "create", "ready", "execute", "health_check", "destroy"


@dataclass
class ManagedResource:
    id: str
    state: ResourceState = ResourceState.CREATING
    created_at: float = field(default_factory=time.time)
    ready_at: float | None = None
    destroyed_at: float | None = None
    error: Exception | None = None
    rollout_id: str | None = None

    def mark_ready(self) -> None:
        self.state = ResourceState.READY
        self.ready_at = time.time()

    def mark_error(self, error: Exception) -> None:
        self.state = ResourceState.ERROR
        self.error = error

    def mark_destroying(self) -> None:
        self.state = ResourceState.DESTROYING

    def mark_destroyed(self) -> None:
        self.state = ResourceState.DESTROYED
        self.destroyed_at = time.time()

    @property
    def is_active(self) -> bool:
        return self.state in (ResourceState.CREATING, ResourceState.READY)

    @property
    def ready_wait_time(self) -> float:
        """Time spent waiting for resource to become ready."""
        if self.ready_at is None:
            return time.time() - self.created_at
        return self.ready_at - self.created_at


R = TypeVar("R", bound=ManagedResource)


class ResourceManager(ABC, Generic[R]):
    """Base class for resource lifecycle management.

    Generic over the resource type R (must be a ManagedResource subclass).
    Subclasses should specify the concrete type, e.g.:
        class SandboxManager(ResourceManager[ManagedSandbox]): ...
    """

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        health_check_interval: float = 30.0,
        enable_health_monitoring: bool = False,
    ):
        # Avoid circular import
        from verifiers.envs.experimental.resource_managers.retry import DEFAULT_RETRY_CONFIG

        self.resources: dict[str, R] = {}
        self.errors: list[ResourceError] = []
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self.health_check_interval = health_check_interval
        self.enable_health_monitoring = enable_health_monitoring
        self.health_monitor_task: asyncio.Task[None] | None = None
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Build retry wrapper
        self._with_retry = self.retry_config.build_retryer(self.logger).wraps

    async def acquire(self, rollout_id: str | None = None) -> R:
        resource = self.create_resource_object(rollout_id)
        return await self._do_acquire(resource, rollout_id)

    async def _do_acquire(self, resource: R, rollout_id: str | None) -> R:
        original_id = resource.id

        async with self.lock:
            self.resources[resource.id] = resource

        try:
            await self._with_retry(self.create_resource)(resource)

            if resource.id != original_id:
                async with self.lock:
                    del self.resources[original_id]
                    self.resources[resource.id] = resource

            resource.mark_ready()

            if self.enable_health_monitoring and self.health_monitor_task is None:
                self.health_monitor_task = asyncio.create_task(self.health_monitor_loop())

            return resource

        except Exception as e:
            resource.mark_error(e)
            self.record_error(resource.id, rollout_id, e, "create")
            raise

    async def release(self, resource_id: str) -> None:
        """Release a resource."""
        async with self.lock:
            resource = self.resources.get(resource_id)
            if resource is None:
                return

        if resource.state == ResourceState.DESTROYED:
            return

        resource.mark_destroying()

        try:
            await self._with_retry(self.destroy_resource)(resource_id)
            resource.mark_destroyed()
        except Exception as e:
            self.record_error(resource_id, resource.rollout_id, e, "destroy")
            resource.mark_destroyed()

    async def release_all(self) -> None:
        """Release all active resources using TaskGroup for better error handling."""
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.health_monitor_task
            self.health_monitor_task = None

        async with self.lock:
            active = [r for r in self.resources.values() if r.is_active]

        if active:
            self.logger.info(f"Releasing {len(active)} resources")
            # Use TaskGroup for structured concurrency - all tasks complete or all are cancelled
            async with asyncio.TaskGroup() as tg:
                for r in active:
                    tg.create_task(self._release_with_error_handling(r.id))

    async def _release_with_error_handling(self, resource_id: str) -> None:
        """Release a resource, catching exceptions to not break TaskGroup."""
        try:
            await self.release(resource_id)
        except Exception as e:
            self.logger.warning(f"Error releasing {resource_id}: {e}")

    def get_resource(self, resource_id: str) -> R | None:
        """Get a resource by ID."""
        return self.resources.get(resource_id)

    def get_active_resources(self) -> list[R]:
        """Get all active resources."""
        return [r for r in self.resources.values() if r.is_active]

    def record_error(self, resource_id: str, rollout_id: str | None, error: Exception, phase: str) -> None:
        self.errors.append(ResourceError(resource_id, rollout_id, error, time.time(), phase))

    def get_errors_for_rollout(self, rollout_id: str) -> list[ResourceError]:
        return [e for e in self.errors if e.rollout_id == rollout_id]

    def get_all_errors(self) -> list[ResourceError]:
        return list(self.errors)

    def clear_errors(self) -> None:
        """Clear all recorded errors."""
        self.errors.clear()

    async def health_check(self, resource_id: str) -> bool:
        """Run health check and mark resource as ERROR if unhealthy.

        This is the public API for health checking. It calls the abstract
        check_health() method and handles state transitions.
        """
        resource = self.get_resource(resource_id)
        if resource is None:
            return False

        try:
            is_healthy = await self._check_health_impl(resource_id)
            if not is_healthy:
                error = RuntimeError(f"Health check failed for {resource_id}")
                resource.mark_error(error)
                self.record_error(resource_id, resource.rollout_id, error, "health_check")
            return is_healthy
        except Exception as e:
            resource.mark_error(e)
            self.record_error(resource_id, resource.rollout_id, e, "health_check")
            return False

    async def stop_health_monitoring(self) -> None:
        """Stop the health monitoring task."""
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.health_monitor_task
            self.health_monitor_task = None

    async def health_monitor_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                for resource in list(self.resources.values()):
                    if resource.state == ResourceState.READY:
                        # health_check() handles error marking internally
                        await self.health_check(resource.id)
            except asyncio.CancelledError:
                break

    @abstractmethod
    def create_resource_object(self, rollout_id: str | None) -> R:
        """Create the resource object (not the actual resource yet)."""
        pass

    @abstractmethod
    async def create_resource(self, resource: R) -> None:
        """Create the actual resource. May update resource.id."""
        pass

    @abstractmethod
    async def destroy_resource(self, resource_id: str) -> None:
        """Destroy the resource."""
        pass

    @abstractmethod
    async def _check_health_impl(self, resource_id: str) -> bool:
        """Check if resource is healthy. Called by health_check()."""
        pass
