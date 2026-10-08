"""Concrete placement strategies; provider and topology assumptions stay here."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    DockerConfig,
    PrimeConfig,
    RuntimeConfig,
    SubprocessConfig,
    make_runtime,
)
from verifiers.v1.runtimes.compose import compose_services
from verifiers.v1.runtimes.deployment import Deployment, DeploymentStrategy


class SharedDeployment(DeploymentStrategy):
    async def start(self, d):
        if d.execution_config is not None:
            raise ValueError(
                "shared deployment cannot use a separate execution runtime"
            )
        if d.spec is not None:
            raise ValueError("shared Compose requires the nested strategy")
        if d.borrowed is not None:
            d.execution = d.harness = d.borrowed.with_env(d.env)
        else:
            d.owner.env = d.env
            await d.own(d.owner)
            d.execution = d.harness = d.owner
        d.targets = {"main": d.execution}


class IndependentDeployment(DeploymentStrategy):
    """Targets are separate allocations. No parent-child resource arithmetic."""

    async def start(self, d):
        if d.borrowed is not None:
            raise ValueError("independent deployment cannot borrow a single runtime")
        if d.execution_config is None:
            raise ValueError("independent deployment requires agent.execution")
        if isinstance(d.execution_config, SubprocessConfig):
            raise TypeError("independent execution requires an isolated workspace")
        d.harness = await d.own(d.owner)
        if d.spec is not None:
            services, self._stop_main, _host = await d.enter(
                compose_services(
                    d.execution_config,
                    d.spec,
                    d.env,
                    trust_compose=d.config.trust_compose,
                )
            )
            d.targets = dict(services)
            d.execution = services[d.spec.workspace]
        else:
            d.execution = await d.allocate("main", d.execution_config, env=d.env)

    async def stop_execution(self, d):
        if d.spec is not None:
            await self._stop_main()


class NestedContainerDeployment(DeploymentStrategy):
    """Containers share a provider VM or use the local Docker daemon."""

    def verifier_config(self, deployment):
        return deployment.owner.config

    def resolve_execution(self, harness, execution):
        if not isinstance(execution, DockerConfig) or not isinstance(
            harness, (DockerConfig, PrimeConfig)
        ):
            raise TypeError(
                "nested execution requires Docker targets on Prime or local Docker"
            )

        if execution.cpu is None:
            execution = execution.model_copy(update={"cpu": (harness.cpu or 2) * 0.75})
        if execution.memory is None:
            execution = execution.model_copy(
                update={"memory": (harness.memory or 8) * 0.75}
            )
        assert execution.memory is not None
        if harness.memory is not None and execution.memory >= harness.memory:
            raise ValueError(
                "task memory must leave space for the harness in the outer VM"
            )
        if (
            execution.cpu is not None
            and harness.cpu is not None
            and execution.cpu > harness.cpu
        ):
            raise ValueError("task CPUs exceed the outer VM allocation")
        if execution.gpu:
            raise ValueError("split task runtimes do not yet support GPU limits")
        if execution.disk is not None:
            if not isinstance(harness, PrimeConfig):
                raise TypeError("execution disk quotas currently require a Prime VM")
            if execution.disk >= harness.disk:
                raise ValueError("execution disk must leave storage for the harness")
        if execution.network_restricted and execution.allow:
            raise ValueError(
                "split execution supports unrestricted task networking or an empty allowlist"
            )
        return execution

    async def start(self, deployment: Deployment) -> None:
        d = deployment
        self._stop_main = None
        self._grading_networks = []
        self._isolated_networks = []
        if d.spec is not None:
            if d.borrowed is not None:
                raise ValueError("a composed deployment cannot borrow a single target")
            services, self._stop_main, host = await d.enter(
                compose_services(
                    d.owner.config.model_copy(
                        update=d.execution_config.model_dump(
                            include={"image", "workdir"}
                        )
                        if d.execution_config
                        else {}
                    ),
                    d.spec,
                    d.env,
                    trust_compose=d.config.trust_compose,
                    execution=d.execution_config,
                )
            )
            d.targets = dict(services)
            d.execution = services[d.spec.workspace]
            d.harness = d.execution
            if d.execution_config is not None:
                if host is None:
                    d.harness = make_runtime(SubprocessConfig())
                    await d.own(d.harness)
                else:
                    d.harness = host
            return
        if d.borrowed is not None:
            d.execution = d.harness = d.borrowed.with_env(d.env)
        elif d.execution_config is None:
            d.owner.env = d.env
            await d.own(d.owner)
            d.execution = d.harness = d.owner
        else:
            from verifiers.v1.runtimes.container_target import ContainerTarget

            if isinstance(d.owner.config, PrimeConfig):
                d.harness = d.owner
            elif isinstance(d.owner.config, DockerConfig):
                # Local Docker needs no outer allocation: the trusted harness runs
                # on the evaluator, and all task code runs inside its owned container.
                d.harness = make_runtime(SubprocessConfig())
            else:
                raise ValueError("split execution requires Prime or local Docker")
            await d.own(d.harness)
            if isinstance(d.owner.config, PrimeConfig):
                from verifiers.v1.runtimes.docker_host import prepare_docker_host

                await prepare_docker_host(
                    d.harness, d.execution_config.disk or d.owner.config.disk * 0.6
                )
            d.execution = ContainerTarget(
                d.execution_config, host=d.harness, name=f"{d.owner.name}-task"
            )
            d.execution.env = d.env
            await d.own(d.execution)
        d.targets = {"main": d.execution}

    async def prepare_execution(self, d: Deployment, routes: list[str]) -> None:
        assert d.execution is not None and d.harness is not None
        await d.harness.prepare_execution(routes)
        if not d.split:
            return
        if d.spec is None or not d.execution_config.network_restricted:
            await d.execution.prepare_execution([])
            return
        # Replace each task network with an internal network, retaining service
        # membership and DNS aliases. The harness is outside the task networks.
        import uuid

        networks = {}
        for name, target in d.targets.items():
            result = await target._run_host(
                "docker",
                "inspect",
                "--format",
                "{{json .NetworkSettings.Networks}}",
                target._container,
            )
            if result.exit_code:
                raise SandboxError("cannot inspect task network membership")
            for network, settings in json.loads(result.stdout).items():
                networks.setdefault(network, []).append(
                    (name, target, settings.get("Aliases") or [])
                )
        for original, members in networks.items():
            isolated = f"vf-internal-{uuid.uuid4().hex}"
            result = await d.execution._run_host(
                "docker", "network", "create", "--internal", isolated
            )
            if result.exit_code:
                raise SandboxError(f"cannot isolate task network: {result.stderr}")
            containers = []
            self._isolated_networks.append((isolated, containers))
            for name, target, aliases in members:
                args = ["docker", "network", "connect", "--alias", name]
                for alias in aliases:
                    args += ["--alias", alias]
                result = await target._run_host(*args, isolated, target._container)
                if result.exit_code:
                    raise SandboxError(
                        f"cannot connect isolated service: {result.stderr}"
                    )
                containers.append(target._container)
                self._grading_networks.append(
                    (target, original, isolated, name, aliases)
                )
                result = await target._run_host(
                    "docker", "network", "disconnect", original, target._container
                )
                if result.exit_code:
                    raise SandboxError(f"cannot remove task egress: {result.stderr}")

    async def prepare_grading(self, d: Deployment) -> None:
        """Restore task networking for a trusted shared-workspace verifier."""
        for target, original, isolated, name, aliases in self._grading_networks:
            args = ["docker", "network", "connect", "--alias", name]
            for alias in aliases:
                args += ["--alias", alias]
            result = await target._run_host(*args, original, target._container)
            if result.exit_code:
                raise SandboxError(f"cannot restore verifier network: {result.stderr}")
            result = await target._run_host(
                "docker", "network", "disconnect", isolated, target._container
            )
            if result.exit_code:
                raise SandboxError(f"cannot release isolated network: {result.stderr}")
        self._grading_networks.clear()
        await d.execution.prepare_execution(None)

    async def stop_execution(self, d):
        if self._stop_main is not None:
            await self._stop_main()

    async def close(self, d):
        if d.execution is None:
            return
        for network, containers in self._isolated_networks:
            for container in containers:
                with contextlib.suppress(Exception):
                    await d.execution._run_host(
                        "docker", "network", "disconnect", "--force", network, container
                    )
            with contextlib.suppress(Exception):
                await d.execution._run_host("docker", "network", "rm", network)
        self._isolated_networks.clear()


# Factories return a fresh strategy per attempt; mutable networking state is never shared.
_STRATEGIES: dict[str, Callable[[], DeploymentStrategy]] = {
    "shared": SharedDeployment,
    "nested": NestedContainerDeployment,
    "independent": IndependentDeployment,
}


def register_deployment_strategy(
    name: str, factory: Callable[[], DeploymentStrategy]
) -> None:
    if name == "auto" or name in _STRATEGIES:
        raise ValueError(f"deployment strategy {name!r} is already registered")
    _STRATEGIES[name] = factory


def deployment_strategy(
    name: str,
    harness: RuntimeConfig,
    execution: RuntimeConfig | None,
    *,
    composed: bool = False,
) -> DeploymentStrategy:
    if name == "auto":
        if (composed and execution is None) or (
            isinstance(execution, DockerConfig)
            and isinstance(harness, (PrimeConfig, DockerConfig))
        ):
            name = "nested"
        else:
            name = "independent" if execution is not None else "shared"
    try:
        return _STRATEGIES[name]()
    except KeyError:
        raise ValueError(
            f"unknown deployment strategy {name!r}; available: {sorted(_STRATEGIES)}"
        ) from None
