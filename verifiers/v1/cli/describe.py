"""The `vf-describe` entrypoint: describe an installed package by importing it.

Prints, as JSON on stdout, what a package exports and how it is built: its taskset,
harness, env and judge classes, their config schemas, capability flags, rewards and
metrics with their source locations, toolsets, and the env's agent roles with their
default harnesses. Nothing is loaded: `Taskset.load()` never runs, no runtime starts
and no model is called. So a registry or hub indexer can describe a package
without running it.

Describing still runs the package's own code: its import, the default factories
of its taskset and task configs (their validators don't run), any JSON-schema
hooks on its models, and the `toolsets()` classmethods, as an env calls them
before a run. Describe an untrusted package where its code may run, such as a
sandbox. A reward or metric plugged in the default config is named by its `fn`
path, never imported.

A package that can't be described exits non-zero with a JSON error on stderr.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import inspect
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, NoReturn, get_args, get_origin

from pydantic import BaseModel

import verifiers
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.configs.judge import judge_key
from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.env import Env
from verifiers.v1.envs.single_agent import SingleAgentEnv
from verifiers.v1.harness import Harness
from verifiers.v1.judge import Judge, judge_config_cls
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset
from verifiers.v1.utils.generic import concrete_type
from verifiers.v1.utils.loaders import (
    _import_plugin,
    _plugin_class,
    default_harness_id,
)

SCHEMA_VERSION = 1
"""The descriptor's format. Raised when a field changes meaning or is removed; adding
a field keeps it."""

USAGE = "usage: uv run vf-describe <id>"

KINDS: dict[str, tuple[type, str]] = {
    "taskset": (Taskset, "verifiers.v1.tasksets"),
    "harness": (Harness, "verifiers.v1.harnesses"),
    "env": (Env, "verifiers.v1.envs"),
    "judge": (Judge, "verifiers.v1.judges"),
}
"""Each plugin kind a package may export, with its base class and built-in group."""


class DescribeError(Exception):
    """A package that can't be described."""


def _name(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _relative(path: str) -> str | None:
    """`path` relative to the import root it was imported from, such as
    `my_taskset/__init__.py`, so it names the file inside its package. None for a
    file outside every import root."""
    resolved = Path(path).resolve()
    roots = sorted(
        (Path(entry or ".").resolve() for entry in sys.path),
        key=lambda p: -len(p.parts),
    )
    for root in roots:
        if resolved.is_relative_to(root):
            return resolved.relative_to(root).as_posix()
    return None


def _source(fn: Callable[..., Any]) -> dict[str, Any] | None:
    target = inspect.unwrap(fn)
    try:
        file = inspect.getsourcefile(target)
        line = inspect.getsourcelines(target)[1]
    except (OSError, TypeError):
        return None
    if file is None:
        return None
    return {"file": _relative(file), "line": line}


def _doc(obj: Any) -> str | None:
    """`obj`'s own docstring: a class's is never its base's (`inspect.getdoc` would
    hand back `ABC`'s)."""
    if isinstance(obj, type):
        own = vars(obj).get("__doc__")
        return inspect.cleandoc(own) if isinstance(own, str) and own.strip() else None
    return inspect.getdoc(obj) or None


def _decorated(cls: type, attr: str) -> dict[str, Callable[..., Any]]:
    """`cls`'s functions tagged `attr` (by `@vf.<attr>`), by name: the class-level twin
    of `discover_decorated`, so no instance is built. The most-derived definition wins,
    and an undecorated override drops a decorated base method."""
    names = {
        name
        for klass in cls.__mro__
        for name, fn in vars(klass).items()
        if callable(fn) and hasattr(fn, attr)
    }
    return {name: fn for name in names if hasattr(fn := getattr(cls, name), attr)}


def _signal(name: str, fn: Callable[..., Any], attr: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "priority": getattr(fn, f"{attr}_priority", 0),
    }
    if attr == "reward":
        entry["weight"] = getattr(fn, "_vf_weight", 1.0)
    entry["doc"] = _doc(fn)
    entry["source"] = _source(fn)
    entry["fn"] = None
    return entry


def _plugged(
    name: str, spec: Any, attr: str, decorated: dict[str, Any] | None
) -> dict[str, Any]:
    """A config-plugged entry, merged as `load_plugged_fn` merges it but without
    importing its `fn`: a value the config leaves unset that the function itself
    would set is unknown (None)."""
    if not spec.fn and decorated is None:
        raise ValueError(
            f"{attr} {name!r} plugs no `fn` and the task has no @vf.{attr} "
            f"method named {name!r} to override"
        )
    base = decorated if not spec.fn else None
    entry: dict[str, Any] = {
        "name": name,
        "priority": spec.priority
        if spec.priority is not None
        else (base["priority"] if base else None),
    }
    if attr == "reward":
        entry["weight"] = (
            spec.weight
            if spec.weight is not None
            else (base["weight"] if base else None)
        )
    entry["doc"] = base["doc"] if base else None
    entry["source"] = base["source"] if base else None
    entry["fn"] = spec.fn
    return entry


def _signals(
    cls: type, attr: str, plugged: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """The `@vf.<attr>` methods of `cls`, merged with `plugged` config entries as
    `Task.hooks` merges them, sorted by descending priority then name. A plugged
    entry's `fn` is named, never imported."""
    merged = {
        name: _signal(name, fn, attr) for name, fn in _decorated(cls, attr).items()
    }
    for name, spec in (plugged or {}).items():
        merged[name] = _plugged(name, spec, attr, merged.get(name))
    entries = list(merged.values())
    entries.sort(key=lambda entry: (-(entry["priority"] or 0), entry["name"]))
    return entries


def _flags(cls: type, base: type) -> dict[str, bool]:
    """`base`'s boolean class flags (`NEEDS_CONTAINER`, `SUPPORTS_MCP`, ...) as `cls`
    sets them."""
    return {
        name: bool(getattr(cls, name))
        for name in sorted(vars(base))
        if name.isupper() and isinstance(getattr(base, name), bool)
    }


def _literals(annotation: Any) -> list[Any]:
    """The values of every `Literal` inside `annotation`, in order."""
    if get_origin(annotation) is Literal:
        return list(get_args(annotation))
    return [value for arg in get_args(annotation) for value in _literals(arg)]


class _Describer:
    def __init__(self, plugin_id: str) -> None:
        self.plugin_id = plugin_id
        self.errors: list[dict[str, str]] = []

    def attempt(self, where: str, fn: Callable[[], Any]) -> Any:
        """`fn()`, or None with the failure recorded under `where`: one part that
        can't be described leaves the rest of the descriptor."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - package code; recorded, not raised
            self.errors.append(
                {"where": where, "type": type(e).__name__, "message": str(e)}
            )
            return None

    def schema(self, where: str, model: type[BaseModel]) -> dict[str, Any] | None:
        return self.attempt(where, model.model_json_schema)

    def describe(self) -> dict[str, Any]:
        exports, modules = self._exports()
        taskset = exports.get("taskset")
        harness = exports.get("harness")
        env = exports.get("env")
        judge = exports.get("judge")
        default_harness = default_harness_id(self.plugin_id) if taskset else "bash"
        return {
            "schema_version": SCHEMA_VERSION,
            "verifiers_version": verifiers.__version__,
            "id": self.plugin_id,
            "modules": modules,
            "distribution": self._distribution(modules),
            "exports": {
                kind: _name(cls) if cls else None
                for kind, cls in (
                    ("taskset", taskset),
                    ("harness", harness),
                    ("env", env),
                    ("judge", judge),
                )
            },
            "taskset": self._taskset(taskset) if taskset else None,
            "harness": self._harness(harness) if harness else None,
            "env": self._env(env, default_harness) if env or taskset else None,
            "judge": self._judge(judge) if judge else None,
            "default_harness": default_harness,
            "errors": self.errors,
        }

    def _exports(self) -> tuple[dict[str, type], list[str]]:
        """The one class of each kind the package exports via `__all__`, found where a
        run finds it, and the modules they came from. A run reads a taskset's bundled
        harness and env from the taskset's own module and nowhere else, so for a
        taskset package those two come from that module. Everything else comes from
        the module its own built-in group resolves the id to, as the run's loaders
        resolve a standalone harness or env, and every judge."""
        exports: dict[str, type] = {}
        modules: list[str] = []
        self._not_found: ModuleNotFoundError | None = None
        self._no_exports: Exception | None = None
        imported: dict[str, ModuleType | None] = {}
        for kind, (base, _) in KINDS.items():
            # A run reads a taskset package's bundled harness and env from its own
            # module, never from a built-in that shares its id. Everything else
            # resolves through its own group, as `harness_class` and `judge_class` do:
            # a judge is always loaded by its id, even one a taskset package exports.
            bundled = "taskset" in exports and kind in ("harness", "env")
            group_kind = "taskset" if bundled else kind
            if kind == "judge" and "taskset" in exports:
                # A built-in taskset's id names no judge, so the judge group falls
                # back to a third-party library of that name (say `textarena`): one
                # that fails to import is no judge of the package's, not a failure.
                try:
                    module = self._import(group_kind)
                    found = self._exported(module, base, kind) if module else None
                except Exception:  # noqa: BLE001 - an unrelated library's import
                    found = None
            else:
                if group_kind not in imported:
                    imported[group_kind] = self._import(group_kind)
                module = imported[group_kind]
                found = self._exported(module, base, kind) if module else None
            if module is None or found is None:
                continue
            exports[kind] = found
            if module.__name__ not in modules:
                modules.append(module.__name__)
        if not exports:
            if self._no_exports is not None:
                raise DescribeError(
                    f"{self.plugin_id!r} exports no Taskset, Harness, Env or Judge "
                    f"subclass via `__all__` ({self._no_exports})"
                )
            module = _module_name(self.plugin_id)
            raise DescribeError(
                f"{self.plugin_id!r} not found: no installed package imports as "
                f"{module!r}, and no built-in taskset, harness, env or judge has that "
                "name"
            ) from self._not_found
        return exports, modules

    def _import(self, kind: str) -> ModuleType | None:
        """The module the id resolves to in `kind`'s built-in group (else installed),
        or None when there is none."""
        group = KINDS[kind][1]
        try:
            return _import_plugin(self.plugin_id, kind, group)
        except ModuleNotFoundError as e:
            cause = e.__cause__
            # A dependency missing inside the package is its own failure, not a
            # package that isn't there.
            if isinstance(cause, ModuleNotFoundError) and not _is_target(
                cause.name, self.plugin_id, group
            ):
                raise DescribeError(str(cause)) from cause
            self._not_found = self._not_found or e
            return None

    def _exported(self, module: ModuleType, base: type, kind: str) -> type | None:
        try:
            return _plugin_class(module, base, kind)
        except (AttributeError, TypeError) as e:
            self._no_exports = self._no_exports or e
            return None
        except ValueError as e:
            raise DescribeError(str(e)) from e

    def _distribution(self, modules: list[str]) -> dict[str, str] | None:
        """The installed distribution that provides the package's module, when one
        does (a built-in plugin is verifiers'). An editable install records no
        top-level module, so the distribution named like the module is the fallback."""
        top = modules[0].split(".", 1)[0]
        names = importlib.metadata.packages_distributions().get(top) or [top]
        for name in names:
            try:
                dist = importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                continue
            return {"name": dist.metadata["Name"], "version": dist.version}
        return None

    def _taskset(self, cls: type[Taskset]) -> dict[str, Any]:
        config_type = concrete_type(cls, TasksetConfig, origin=Taskset) or TasksetConfig
        task_cls = cls.task_type()
        # The declared defaults, built without validation: a config's validators are
        # package code that runs only when a run configures it. Default factories
        # still run, as they must to build a default.
        config = self.attempt(
            "taskset.config",
            lambda: config_type.model_construct(id=self.plugin_id),
        )
        task_config = self.attempt(
            "task.config",
            lambda: (
                getattr(config, "task", None)
                or task_cls.config_type().model_construct()
            ),
        )
        return {
            "class": _name(cls),
            "doc": _doc(cls),
            "infinite": bool(cls.INFINITE),
            "config_schema": self.schema("taskset.config_schema", config_type),
            "toolsets": (
                self._toolsets("taskset.toolsets", lambda: cls.toolsets(config))
                if config is not None
                else None
            ),
            "task": self._task(task_cls, task_config),
        }

    def _task(self, cls: type[Task], config: TaskConfig | None) -> dict[str, Any]:
        data_type = cls.data_type()
        rewards = self.attempt(
            "task.rewards",
            lambda: _signals(cls, "reward", config.rewards if config else None),
        )
        metrics = self.attempt(
            "task.metrics",
            lambda: _signals(cls, "metric", config.metrics if config else None),
        )
        judges = (
            [
                {"name": judge_key(judge), "id": judge.id, "weight": judge.weight}
                for judge in config.judges
            ]
            if config is not None
            else None
        )
        return {
            "class": _name(cls),
            "doc": _doc(cls),
            "needs_container": bool(cls.NEEDS_CONTAINER),
            "data_class": _name(data_type),
            "data_schema": self.schema("task.data_schema", data_type),
            "config_schema": self.schema("task.config_schema", cls.config_type()),
            "rewards": rewards,
            "metrics": metrics,
            "judges": judges,
            "toolsets": (
                self._toolsets("task.toolsets", lambda: cls.toolsets(config))
                if config is not None
                else None
            ),
        }

    def _toolsets(
        self, where: str, build: Callable[[], list[Any]]
    ) -> list[dict[str, Any]] | None:
        """The tool servers a `toolsets()` classmethod returns for the default config,
        each with its tools."""

        def describe() -> list[dict[str, Any]]:
            entries = []
            for toolset in build():
                cls = type(toolset)
                tools = [
                    {
                        "name": getattr(fn, "tool_name", None) or name,
                        "doc": _doc(fn),
                        "source": _source(fn),
                    }
                    for name, fn in sorted(_decorated(cls, "tool").items())
                ]
                entries.append(
                    {
                        "class": _name(cls),
                        "tool_prefix": getattr(cls, "TOOL_PREFIX", None),
                        # A remote server's tools are listed by the server, not here.
                        "url": getattr(getattr(toolset, "config", None), "url", None),
                        "tools": tools,
                    }
                )
            return entries

        return self.attempt(where, describe)

    def _harness(self, cls: type[Harness]) -> dict[str, Any]:
        config_type = concrete_type(cls, HarnessConfig) or HarnessConfig
        dialect = config_type.model_fields.get("dialect")
        return {
            "class": _name(cls),
            "doc": _doc(cls),
            "capabilities": _flags(cls, Harness),
            "dialects": _literals(dialect.annotation) if dialect else [],
            "default_dialect": dialect.default if dialect else None,
            "config_schema": self.schema("harness.config_schema", config_type),
            "metrics": self.attempt("harness.metrics", lambda: _signals(cls, "metric")),
        }

    def _env(self, cls: type[Env] | None, default_harness: str) -> dict[str, Any]:
        """The env that runs the package: its own, else `SingleAgentEnv`, with each
        agent role and the harness it plays with unless the run pins another."""
        env_cls = cls or SingleAgentEnv
        config_type = concrete_type(env_cls, EnvConfig, origin=Env) or EnvConfig
        roles = []
        for name, field in config_type.model_fields.items():
            if not isinstance(field.default, AgentConfig):
                continue
            pinned = field.default.harness
            roles.append(
                {
                    "name": name,
                    "doc": field.description,
                    "harness": pinned.id if pinned is not None else default_harness,
                    "harness_pinned": pinned is not None,
                }
            )
        return {
            "class": _name(env_cls),
            "exported": cls is not None,
            "doc": _doc(env_cls),
            "config_schema": self.schema("env.config_schema", config_type),
            "roles": roles,
        }

    def _judge(self, cls: type[Judge]) -> dict[str, Any]:
        return {
            "class": _name(cls),
            "doc": _doc(cls),
            "config_schema": self.schema("judge.config_schema", judge_config_cls(cls)),
        }


def _is_target(missing: str | None, plugin_id: str, group: str) -> bool:
    """Whether the missing module is the plugin itself (or the built-in group's copy
    of it), rather than something the plugin imports."""
    if not missing:
        return True
    name = _module_name(plugin_id)
    return missing in (name, f"{group}.{name}")


def _module_name(plugin_id: str) -> str:
    """The module an id imports as: a hub id's owner and version dropped, as the
    loaders drop them."""
    return plugin_id.rsplit("/", 1)[-1].split("@", 1)[0].replace("-", "_").lower()


def describe(plugin_id: str) -> dict[str, Any]:
    """The descriptor of the installed package `plugin_id` (an id as the other `vf-*`
    commands take it). Raises `DescribeError` for one that can't be described."""
    return _Describer(plugin_id).describe()


def _fail(error: BaseException) -> NoReturn:
    print(
        json.dumps({"error": {"type": type(error).__name__, "message": str(error)}}),
        file=sys.stderr,
    )
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:]) if argv is None else list(argv)
    if any(arg in ("-h", "--help") for arg in args):
        print(f"{USAGE}\n\n{__doc__}")
        return
    if len(args) != 1 or args[0].startswith("-"):
        raise SystemExit(USAGE)
    # Package code may print while it imports: keep stdout for the descriptor alone.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            descriptor = describe(args[0])
    except DescribeError as e:
        _fail(e)
    except Exception as e:  # noqa: BLE001 - importing package code can raise anything
        _fail(e)
    print(json.dumps(descriptor, indent=2, default=str))


if __name__ == "__main__":
    main()
