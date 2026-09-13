"""Version-gated access to Inspect's task and registry loaders.

Inspect accepts ``package/task`` and ``file.py@task`` references through its
runner, but does not expose the underlying resolver as public API. Keep that
private dependency in this module so an Inspect upgrade has one repair site.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any
from urllib.parse import urlsplit, urlunsplit

MIN_INSPECT_VERSION = "0.3.263"
MAX_INSPECT_VERSION = "0.4"
INSPECT_INSTALL_HINT = (
    "uv run --with 'inspect-ai>=0.3.263,<0.4' eval @ inspect-task.toml"
)


def _sanitize_origin(origin: str) -> str:
    """Drop credentials, query parameters, and fragments from a provenance URL."""
    parsed = urlsplit(origin)
    if parsed.scheme and parsed.hostname:
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    return origin.split("?", 1)[0].split("#", 1)[0]


def require_inspect() -> str:
    """Return the installed Inspect version or raise an actionable error."""
    try:
        version = importlib.metadata.version("inspect-ai")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError(
            "the Inspect taskset requires inspect-ai on both the task loader and "
            f"scoring workers; install it for this run, e.g. {INSPECT_INSTALL_HINT}"
        ) from e

    try:
        from packaging.version import Version

        parsed = Version(version)
        if not (Version(MIN_INSPECT_VERSION) <= parsed < Version(MAX_INSPECT_VERSION)):
            raise RuntimeError(
                f"unsupported inspect-ai version {version}; this adapter is tested with "
                f">={MIN_INSPECT_VERSION},<{MAX_INSPECT_VERSION}"
            )
    except ImportError as e:  # packaging is an inspect-ai dependency
        raise RuntimeError(
            "inspect-ai is installed without its packaging dependency"
        ) from e
    return version


def load_inspect_task(source: str, source_args: dict[str, Any]) -> Any:
    """Resolve exactly one Inspect task from a normal Inspect reference."""
    require_inspect()
    try:
        from inspect_ai._eval.loader import load_tasks
    except ImportError as e:
        raise RuntimeError(
            "inspect-ai changed its private task loader; update "
            "verifiers.v1.tasksets.inspect.compat for the installed version"
        ) from e

    tasks = load_tasks([source], dict(source_args))
    if len(tasks) != 1:
        names = [getattr(task, "name", type(task).__name__) for task in tasks]
        raise ValueError(
            f"Inspect source {source!r} resolved to {len(tasks)} tasks ({names}); "
            "select exactly one package/task or file.py@task"
        )
    return tasks[0]


def registry_spec(value: Any) -> tuple[str, dict[str, Any]]:
    """Registry name and explicitly supplied construction arguments."""
    require_inspect()
    from inspect_ai._util.registry import registry_info, registry_params

    try:
        return registry_info(value).name, dict(registry_params(value))
    except ValueError as e:
        raise ValueError(
            f"Inspect object {getattr(value, '__name__', type(value).__name__)!r} "
            "has no registry metadata and cannot be reconstructed safely"
        ) from e


def task_provenance(task: Any) -> dict[str, str | None]:
    """Best-effort installed-distribution and PEP 610 VCS provenance."""
    require_inspect()
    result: dict[str, str | None] = {
        "distribution": None,
        "distribution_version": None,
        "revision": None,
        "origin": None,
    }
    try:
        from inspect_ai._eval.task.log import (
            resolve_package_revision,
            resolve_task_distribution,
        )

        distribution = resolve_task_distribution(getattr(task, "registry_name", None))
        if distribution is None:
            return result
        result["distribution"] = distribution.name
        result["distribution_version"] = distribution.version
        revision = resolve_package_revision(distribution)
        if revision is not None:
            result["revision"] = revision.commit
            result["origin"] = (
                _sanitize_origin(revision.origin) if revision.origin else None
            )
    except (
        AttributeError,
        ImportError,
        ValueError,
        importlib.metadata.PackageNotFoundError,
    ):
        # Provenance is useful but not part of task execution. The version gate and
        # resolver remain strict; an Inspect patch moving only its log helpers should
        # not make an otherwise compatible task unrunnable.
        pass
    return result
