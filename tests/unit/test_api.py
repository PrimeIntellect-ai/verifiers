"""The public API: the `verifiers.v1` namespace and the plugin packages. Their lazy names
stay in sync with what type checkers see, their signatures use only public types,
docs/api.md lists exactly their names, and the config tree imports without the runtime."""

import ast
import contextlib
import importlib
import inspect
import json
import pkgutil
import re
import subprocess
import sys
import typing
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import verifiers.v1 as vf

API_REFERENCE = Path(__file__).parents[2] / "docs" / "api.md"

# What the config tree may load: a trainer validates its configs against it in an
# environment that installs nothing else.
CONFIG_TREE_PACKAGES = {
    "annotated_types",
    "pydantic",
    "pydantic_config",
    "pydantic_core",
    "renderers",
    "tomli",
    "tomli_w",
    "typing_extensions",
    "typing_inspection",
    "verifiers",
}

IMPORT_CONFIG_TREE = """
import importlib, json, pkgutil, sys
import verifiers.v1._configs as configs
for module in pkgutil.walk_packages(configs.__path__, "verifiers.v1._configs."):
    importlib.import_module(module.name)
packages = {name.partition(".")[0] for name in sys.modules if not name.startswith("_")}
print(json.dumps(sorted(packages - sys.stdlib_module_names)))
"""


def type_checking_name_to_module() -> dict[str, str]:
    tree = ast.parse(Path(vf.__file__).read_text())
    block = next(node for node in tree.body if isinstance(node, ast.If))
    return {alias.name: node.module for node in block.body for alias in node.names}


def test_lazy_names_match_type_checking_imports():
    assert vf._NAME_TO_MODULE == type_checking_name_to_module()


def test_all_exports_every_lazy_name():
    assert sorted(vf.__all__) == sorted(vf._NAME_TO_MODULE)


def test_every_lazy_name_resolves():
    resolved = {name: getattr(vf, name) for name in vf.__all__}

    assert sorted(resolved) == sorted(vf.__all__)


def test_config_tree_imports_without_the_runtime():
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_CONFIG_TREE],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )

    assert set(json.loads(result.stdout)) - CONFIG_TREE_PACKAGES == set()


PLUGIN_GROUPS = ("harnesses", "tasksets", "envs", "judges")

# Framework-built objects take internal wiring in their constructors; users receive
# them and never construct them. `Client.get_response` takes internal request types:
# a custom `Client` subclass is not supported yet.
INTERNAL_WIRING = {
    "verifiers.v1._dialects.base.Dialect",
    "verifiers.v1._graph.PendingTurn",
    "verifiers.v1._interception.base.Interception",
    "verifiers.v1._mcp.launch.SharedToolServer",
    "verifiers.v1._rollout.Rollout",
}


def public_namespaces() -> dict[str, ModuleType]:
    """`verifiers.v1` and its plugin packages: every package path without an underscore."""
    namespaces = {"verifiers.v1": vf}
    for group in PLUGIN_GROUPS:
        package = importlib.import_module(f"verifiers.v1.{group}")
        namespaces[package.__name__] = package
        for info in pkgutil.iter_modules(package.__path__, f"{package.__name__}."):
            if not info.name.rpartition(".")[2].startswith("_"):
                namespaces[info.name] = importlib.import_module(info.name)
    return namespaces


def public_objects() -> dict[str, object]:
    return {
        f"{name}.{export}": getattr(module, export)
        for name, module in public_namespaces().items()
        for export in module.__all__
    }


def referenced_types(annotation: object) -> Iterator[type]:
    origin = typing.get_origin(annotation)
    for candidate in (annotation, origin):
        if isinstance(candidate, type):
            # A parametrized pydantic generic is a subclass of its public origin.
            generic = getattr(candidate, "__pydantic_generic_metadata__", None)
            yield (generic or {}).get("origin") or candidate
    for arg in typing.get_args(annotation):
        yield from referenced_types(arg)


def type_hints(obj: object) -> dict[str, object]:
    """Resolve annotations, including names a module imports only under TYPE_CHECKING:
    look each up in the loaded verifiers modules, else treat it as a third-party `Any`."""
    names: dict[str, object] = {name: getattr(vf, name) for name in vf.__all__}
    while True:
        try:
            return typing.get_type_hints(obj, localns=names)
        except NameError as e:
            found = (
                getattr(module, e.name)
                for module_name, module in list(sys.modules.items())
                if module_name.startswith("verifiers.") and hasattr(module, e.name)
            )
            names[e.name] = next(found, typing.Any)


def signature_types(obj: object) -> Iterator[type]:
    members = [obj]
    if isinstance(obj, type):
        members = [
            getattr(member, "fget", None) or getattr(member, "__func__", member)
            for name, member in vars(obj).items()
            if not name.startswith("_") or name == "__init__"
        ]
        for name, annotation in type_hints(obj).items():
            if not name.startswith("_"):
                yield from referenced_types(annotation)
    for member in members:
        if inspect.isfunction(member):
            for annotation in type_hints(member).values():
                yield from referenced_types(annotation)


def test_public_namespaces_declare_all():
    for name, module in public_namespaces().items():
        assert hasattr(module, "__all__"), name


def import_every_module() -> None:
    """Load every verifiers module, so annotations that name a type only under
    TYPE_CHECKING resolve the same way in every test order."""
    for info in pkgutil.walk_packages(vf.__path__, "verifiers.v1."):
        with contextlib.suppress(ImportError):  # modules of optional extras
            importlib.import_module(info.name)


def test_public_signatures_use_public_types():
    import_every_module()
    public = {id(obj) for obj in public_objects().values()}
    leaks = {
        f"{tp.__module__}.{tp.__qualname__}": path
        for path, obj in public_objects().items()
        for tp in signature_types(obj)
        if tp.__module__.startswith("verifiers.") and id(tp) not in public
    }

    assert set(leaks) - INTERNAL_WIRING == set()


def test_api_reference_documents_exactly_the_public_names():
    """Each `## <namespace>` section of docs/api.md lists its `__all__` in table rows."""
    documented: dict[str, set[str]] = {}
    namespace = None
    for line in API_REFERENCE.read_text().splitlines():
        if heading := re.fullmatch(r"## `(verifiers\.[\w.]+)`", line):
            namespace = documented.setdefault(heading.group(1), set())
        elif namespace is not None and line.startswith("| `"):
            names = line.split("|")[1]
            namespace.update(re.findall(r"`(\w+)`", names))

    assert {
        name: set(module.__all__) for name, module in public_namespaces().items()
    } == documented
