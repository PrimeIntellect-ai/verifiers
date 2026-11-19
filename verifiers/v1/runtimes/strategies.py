"""Colocated execution, local Compose, and native Prime service VMs."""

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from urllib.parse import urlsplit

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeBackend, PrimeConfig, make_backend
from verifiers.v1.runtimes.compose import compose_services
from verifiers.v1.runtimes.deployment import ServicesSpec, _DeploymentStrategy
from verifiers.v1.runtimes.harness_pool import (
    HarnessLease,
    HarnessPool,
    current_harness_pool,
)
from verifiers.v1.runtimes.private_network import (
    PrivateNetworkGroup,
    lan_host,
    private_network,
)
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)


class _ColocatedDeployment(_DeploymentStrategy):
    async def start(self, d):
        if d.borrowed_runtime is not None:
            d.execution_backend = d.harness_backend = d.borrowed_runtime.backend
        else:
            d.backend.env = d.env
            await d.own(d.backend)
            d.execution_backend = d.harness_backend = d.backend
        d.target_backends = {"main": d.execution_backend}

    async def prepare_verifier(self, deployment):
        # Borrowed solvers belong to their caller; owned solvers can release their
        # resources now that metrics and submission artifacts have been collected.
        if deployment.borrowed_runtime is None:
            try:
                await deployment.execution_backend.stop()
            except Exception:
                logger.warning(
                    "Failed to tear down the solver before grading", exc_info=True
                )


class _SplitDeployment(_DeploymentStrategy):
    """Docker split execution, where workers use the evaluator's host bridge."""

    async def start(self, d):
        if d.borrowed_runtime is not None or d.spec is not None or d.tool_vms:
            raise ValueError("split deployment requires a dedicated single workspace")
        if d.backend.type != "docker" or d.harness_config is None:
            raise ValueError("host-mediated deployment requires Docker")
        if d.backend.type == "docker" and d.backend.config.mounts:
            raise ValueError(
                "split deployment cannot use Docker bind mounts in the task workspace"
            )
        if not d.backend.network_restricted:
            d.backend.config = d.backend.config.model_copy(
                update={"allow": [], "block": ["*"]}
            )
        d.backend.env = d.env
        await d.own(d.backend)
        config = d.harness_config.model_copy(
            update={"mounts": {}, "workdir": d.backend.config.workdir}
        )
        harness = make_backend(config, name=f"{d.backend.name}-harness")
        harness.env = {}
        await d.own(harness)
        d.execution_backend = d.backend
        d.harness_backend = harness
        d.target_backends = {"main": d.backend}

    async def prepare_execution(self, d, routes, state_route=None):
        await d.harness_backend.prepare_execution(routes)
        await _prepare_split_workspace(d)

    async def stop_execution(self, d):
        await d.execution_backend.stop()

    async def prepare_verifier(self, d):
        try:
            await d.execution_backend.stop()
        except Exception:
            logger.warning(
                "Failed to tear down the solver before grading",
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )


async def _prepare_split_workspace(d):
    """Apply task egress without exposing the evaluator's control endpoints."""
    if not isinstance(d.execution_network, NetworkPolicyConfig):
        raise SandboxError("split workspace has no enforceable network policy")
    if not d.control_routes or any(
        urlsplit(route).scheme not in ("http", "https") or not urlsplit(route).hostname
        for route in d.control_routes
    ):
        raise SandboxError("split workspace has no valid control-plane routes")
    policy = d.execution_network
    hosts = {
        host
        for route in d.control_routes
        if (host := urlsplit(route).hostname) and not host.endswith(".sandbox.internal")
    }
    for route in d.control_routes:
        parsed = urlsplit(route)
        if policy.allow != ["*"] and policy.permits(
            parsed.scheme,
            parsed.hostname or "",
            parsed.port or (443 if parsed.scheme == "https" else 80),
        ):
            raise SandboxError("task network allowlist includes a framework endpoint")
    had_task_blocks = bool(policy.block)
    if policy.allow == ["*"] and hosts:
        policy = type(policy).model_validate(
            {
                **policy.model_dump(),
                "block": list(dict.fromkeys([*policy.block, *sorted(hosts)])),
            }
        )
    d.execution_backend._control_blocked_hosts = (
        frozenset(hosts)
        if policy.allow == ["*"] and not had_task_blocks
        else frozenset()
    )
    if (
        policy.network_restricted
        or not isinstance(d.execution_backend, PrimeBackend)
        or d.execution_backend._egress_enforced
    ):
        await d.execution_backend.prepare_execution([], policy)
    d.execution_backend.config = d.execution_backend.config.model_copy(
        update={"allow": policy.allow, "block": policy.block}
    )
    d.execution_backend.info.allow = policy.allow
    d.execution_backend.info.block = policy.block


class _ContainerDeployment(_DeploymentStrategy):
    async def start(self, d):
        if d.borrowed_runtime is not None:
            raise ValueError("a composed deployment cannot borrow a single target")
        self._services = await d.enter(AsyncExitStack())
        services, self._stop_main = await self._services.enter_async_context(
            compose_services(
                d.backend.config, d.spec, d.env, trust_compose=d.config.trust_compose
            )
        )
        d.target_backends = dict(services)
        d.execution_backend = d.harness_backend = services[d.spec.workspace]

    async def stop_execution(self, d):
        await self._stop_main()

    async def prepare_verifier(self, deployment):
        # Evidence is collected; release the solver before the verifier clock starts.
        try:
            await self._services.aclose()
        except Exception:
            logger.warning(
                "Failed to tear down the solver services before grading",
                exc_info=True,
            )


class _PrivateNetworkDeployment(_DeploymentStrategy):
    _networks: dict[str, list[str]] | None = None
    _internal: frozenset[str] = frozenset()
    _lease: HarnessLease | None = None
    _owned_pool: HarnessPool | None = None
    _harness_released: bool = False
    _net: PrivateNetworkGroup | None = None

    async def _release_harness(self, *, healthy: bool | None = None) -> None:
        if self._harness_released or self._lease is None:
            return
        self._harness_released = True
        lease = self._lease
        if healthy is None:
            healthy = not lease.backend.stopped and (
                not lease.entry.boot_task.done() or lease.backend._client is not None
            )
        network_id = self._net.network_id if self._net is not None else None
        await lease.release(network_id, healthy=healthy)
        if self._owned_pool is not None:
            pool, self._owned_pool = self._owned_pool, None
            await pool.aclose()

    async def start(self, d):
        if d.borrowed_runtime is not None:
            raise ValueError(
                "a deployment with services on VMs of their own cannot borrow a "
                "single target"
            )
        d.backend.env = d.env
        native = isinstance(d.spec, ServicesSpec)
        members = d.spec.services if native else d.tool_vms
        self._lease = None
        self._owned_pool = None
        self._harness_released = False
        self._net = PrivateNetworkGroup()
        if d.config.mode == "split":
            d._harness_workdir = d.backend.config.workdir
            h_cfg = _harness_config(
                d.config.harness, d.execution_network or d.backend.config
            ).model_copy(update={"workdir": d.backend.config.workdir})
            pool = current_harness_pool()
            if pool is None:
                pool = self._owned_pool = HarnessPool()
            lease = self._lease = await pool.acquire(
                h_cfg,
                f"{d.backend.name}-harness",
                multiplex=d.config.multiplex,
            )
            d._harness_lease = lease
            d._stack.push_async_callback(self._release_harness)
        self._services = await d.enter(AsyncExitStack())
        vms = await self._services.enter_async_context(
            private_network(
                d.backend,
                members,
                partial(d.own, d.backend),
                interconnect=native,
                networks=self._networks,
                harness_lease=self._lease,
                harness_peers=d.tool_services,
                out=self._net,
            )
        )
        d.target_backends = {"main": d.backend, **vms}
        d.hosts = {name: lan_host(backend) for name, backend in vms.items()}
        if native:
            d.hosts["main"] = lan_host(d.backend)
        d.execution_backend = d.backend
        d.harness_backend = self._lease.backend if self._lease else d.backend

    async def prepare_execution(self, d, routes, state_route=None):
        def own(name, backend):
            if name == "main":
                return [] if self._lease else routes
            if name in self._internal:
                return []
            return [backend.host_url(state_route)] if state_route else []

        lan_waiters = [
            wait_lan()
            for backend in d.target_backends.values()
            if (wait_lan := getattr(backend, "wait_lan_ready", None)) is not None
        ]
        if self._lease is not None:
            lan_waiters.append(self._lease.wait_lan_ready())
        await asyncio.gather(
            *lan_waiters,
            *(
                _prepare_split_workspace(d)
                if name == "main" and self._lease
                else backend.prepare_execution(own(name, backend))
                for name, backend in d.target_backends.items()
            ),
            *([self._lease.prepare_execution(routes)] if self._lease else []),
        )

    async def stop_execution(self, d):
        # Neither agent VM may reach services while their evidence is collected.
        await asyncio.gather(
            d.execution_backend.stop(),
            *([self._release_harness()] if self._lease else []),
        )

    async def close(self, deployment):
        vms = [
            *(
                [deployment.execution_backend]
                if deployment.execution_backend is not None
                else []
            ),
            *getattr(deployment, "target_backends", {}).values(),
        ]
        seen: set[int] = set()
        unique = [vm for vm in vms if not (id(vm) in seen or seen.add(id(vm)))]
        tasks = [vm.stop() for vm in unique]
        if self._lease is not None:

            async def detach_and_delete() -> None:
                await self._release_harness()
                if self._net is not None:
                    await self._net.delete_network()

            tasks.append(detach_and_delete())
        elif self._net is not None:
            tasks.append(self._net.delete_network())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def prepare_verifier(self, deployment):
        # Evidence is collected; release the solver's VMs before the verifier clock
        # starts, the workspace's even if the service VMs' or the network's release
        # fails.
        if self._net is not None:
            await self.close(deployment)
            try:
                await self._services.aclose()
            except Exception:
                logger.warning(
                    "Failed to tear down service VMs before grading", exc_info=True
                )
            return
        for release, what in (
            *([(self._release_harness, "harness VM")] if self._lease else []),
            (deployment.execution_backend.stop, "workspace VM"),
            (self._services.aclose, "service VMs"),
        ):
            try:
                await release()
            except Exception:
                logger.warning(
                    "Failed to tear down %s before grading", what, exc_info=True
                )


class _NativeComposeDeployment(_PrivateNetworkDeployment):
    _processes = None

    @asynccontextmanager
    async def verifier(self, deployment, config, env):
        main = self._processes.services["main"]
        if isinstance(config, PrimeConfig) and config.image == main.config.image:
            # Dockerfile ENV cannot preserve multiline values; pass image defaults
            # directly, excluding the solver's Compose and rollout overrides.
            env = main.image_env | env
        async with super().verifier(deployment, config, env) as backend:
            yield backend

    async def start(self, d):
        from verifiers.v1.runtimes.native_compose import NativeProcesses, prepare_native

        if d.borrowed_runtime is not None:
            raise ValueError("a composed deployment cannot borrow a single target")
        services = await prepare_native(
            d.backend.config.model_copy(
                update={"workdir": d.backend._workdir_override}
            ),
            d.spec,
            d.env,
        )
        main = services["main"]
        d.backend.config = main.config
        for field, value in main.config.model_dump().items():
            setattr(d.backend.info, field, value)
        d.env = main.env
        self._networks = {}
        for name, service in services.items():
            # Default-bridge peers share IP connectivity without Compose aliases.
            networks = [
                f"project:{network}" for network in service.definition["networks"]
            ]
            for network in networks or ["bridge"]:
                self._networks.setdefault(network, []).append(name)
        self._internal = frozenset(
            name for name, service in services.items() if service.internal
        )
        d.spec = ServicesSpec(
            {
                name: service.config
                for name, service in services.items()
                if name != "main"
            }
        )
        await super().start(d)
        lan_waiters = [
            wait_lan()
            for backend in d.target_backends.values()
            if (wait_lan := getattr(backend, "wait_lan_ready", None)) is not None
        ]
        if lan_waiters:
            await asyncio.gather(*lan_waiters)
        self._processes = NativeProcesses(services)
        await self._processes.start(d)

    async def stop_execution(self, d):
        await self._processes.stop(d, ["main"])
        await super().stop_execution(d)

    async def prepare_verifier(self, deployment):
        if self._processes is not None:
            await self._processes.stop(
                deployment, reversed(list(self._processes.processes))
            )
        await super().prepare_verifier(deployment)

    async def close(self, deployment):
        if self._processes is not None:
            await self._processes.stop(
                deployment, reversed(list(self._processes.processes))
            )
        await super().close(deployment)


def _harness_config(harness: PrimeConfig, workspace: PrimeConfig) -> PrimeConfig:
    """A split harness's VM: placed and labelled like the workspace unless set, and
    under the task's network policy, which the harness would share colocated."""
    return harness.model_copy(
        update={
            "labels": list(dict.fromkeys([*workspace.labels, *harness.labels])),
            "region": harness.region or workspace.region,
            "creates_per_min": (
                harness.creates_per_min
                if harness.creates_per_min is not None
                else workspace.creates_per_min
            ),
            "allow": workspace.allow,
            "block": workspace.block,
        }
    )


def _deployment_strategy(config, backend, spec, tool_vms=None):
    split = config is not None and config.mode == "split"
    if split and backend.type == "docker":
        return _SplitDeployment()
    if split and not isinstance(backend, PrimeBackend):
        raise ValueError("split deployment requires Docker or Prime")
    if isinstance(spec, ServicesSpec):
        if not isinstance(backend, PrimeBackend):
            raise TypeError("Native service VMs require a Prime workspace")
        if "main" in spec.services:
            raise ValueError(
                "'main' is the task workspace; name its services separately"
            )
        if any(
            not isinstance(runtime, PrimeConfig) for runtime in spec.services.values()
        ):
            raise ValueError("Native services require PrimeConfig runtimes")
        return _PrivateNetworkDeployment()
    if spec is not None:
        if isinstance(backend, PrimeBackend):
            return _NativeComposeDeployment()
        if not backend.SUPPORTS_COMPOSE:
            raise ValueError(
                f"the task deploys Compose services, but the {backend.type} runtime "
                "cannot run Compose services; use ServicesSpec on Prime or local Docker"
            )
        return _ContainerDeployment()
    if tool_vms or split:
        return _PrivateNetworkDeployment()
    return _ColocatedDeployment()
