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
import textwrap
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


def is_docstring(node: ast.stmt | None) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def documents_itself(target: ast.AnnAssign) -> bool:
    """A discriminator field, `type: Literal["docker"] = "docker"`, names itself."""
    return (
        isinstance(target.target, ast.Name)
        and target.target.id == "type"
        and ast.unparse(target.annotation).startswith("Literal[")
    )


def definition(
    obj: object,
) -> ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        source = textwrap.dedent(inspect.getsource(obj))  # type: ignore[arg-type]
    except (OSError, TypeError):
        return None
    node = ast.parse(source).body[0]
    if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return node
    return None


def own_members(cls: type) -> dict[str, bool]:
    """The public members a class defines itself, and whether each has a docstring:
    methods and properties, class attributes and fields, and attributes set in
    `__init__`."""
    node = definition(cls)
    members: dict[str, bool] = {}
    if not isinstance(node, ast.ClassDef):
        return members

    def add(name: str, documented: bool) -> None:
        if not name.startswith("_"):
            members[name] = members.get(name, False) or documented

    def scan(body: list[ast.stmt], *, in_init: bool) -> None:
        for i, stmt in enumerate(body):
            following = body[i + 1] if i + 1 < len(body) else None
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                if in_init:
                    continue
                add(stmt.name, ast.get_docstring(stmt) is not None)
                if stmt.name == "__init__":
                    scan(stmt.body, in_init=True)
                continue
            if isinstance(stmt, ast.AnnAssign):
                targets = [stmt.target]
            elif isinstance(stmt, ast.Assign):
                targets = stmt.targets
            else:
                continue
            for target in targets:
                if in_init and not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    continue
                if not in_init and not isinstance(target, ast.Name):
                    continue
                name = target.attr if in_init else target.id
                value = getattr(stmt, "value", None)
                documented = (
                    is_docstring(following)
                    or (value is not None and "description=" in ast.unparse(value))
                    or (isinstance(stmt, ast.AnnAssign) and documents_itself(stmt))
                )
                add(name, documented)

    scan(node.body, in_init=False)
    return members


def documented_member(cls: type, name: str) -> bool:
    """A member is documented by its own docstring or by the one it overrides. Members
    of non-verifiers bases (pydantic, the standard library) are documented there."""
    if own_members(cls).get(name):
        return True
    for base in cls.__mro__[1:]:
        if not hasattr(base, name) and name not in getattr(base, "model_fields", {}):
            continue
        if not base.__module__.startswith("verifiers."):
            return True
        if own_members(base).get(name):
            return True
    return False


def public_classes() -> dict[int, tuple[str, type]]:
    classes: dict[int, tuple[str, type]] = {}
    for path, obj in public_objects().items():
        if isinstance(obj, type) and obj.__module__.startswith("verifiers."):
            classes.setdefault(id(obj), (path, obj))
    return classes


def test_public_names_have_docstrings():
    undocumented = [
        path
        for path, obj in public_objects().items()
        if getattr(obj, "__module__", "").startswith("verifiers.")
        and callable(obj)
        and (node := definition(obj)) is not None
        and ast.get_docstring(node) is None
    ]

    assert undocumented == []


def test_public_members_have_docstrings():
    """A member without an underscore is public, and public means documented (PEP 8)."""
    import_every_module()
    classes = [
        (path, cls)
        for path, cls in public_classes().values()
        # A parametrized generic (`WireTrace`) has the members of its origin.
        if not (getattr(cls, "__pydantic_generic_metadata__", None) or {}).get("origin")
    ]
    unreadable = [path for path, cls in classes if definition(cls) is None]
    undocumented = [
        f"{path}.{name}"
        for path, cls in classes
        for name in own_members(cls)
        if not documented_member(cls, name)
    ]

    assert unreadable == []
    assert undocumented == []


def test_pydantic_validators_are_private():
    """Pydantic calls validators and serializers; user code never does."""
    public = []
    for path, cls in public_classes().values():
        decorators = getattr(cls, "__pydantic_decorators__", None)
        if decorators is None:
            continue
        for kind in (
            "validators",
            "field_validators",
            "root_validators",
            "field_serializers",
            "model_serializers",
            "model_validators",
        ):
            public += [
                f"{path}.{name}"
                for name in getattr(decorators, kind)
                if not name.startswith("_")
            ]

    assert sorted(set(public)) == []


def test_public_modules_export_every_name_they_define():
    """A name that a public module defines is in its `__all__` or starts with an underscore."""
    unexported = []
    for name, module in public_namespaces().items():
        tree = ast.parse(Path(inspect.getsourcefile(module)).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                defined = [node.name]
            elif isinstance(node, ast.Assign):
                defined = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined = [node.target.id]
            else:
                continue
            unexported += [
                f"{name}.{n}"
                for n in defined
                if not n.startswith("_") and n not in module.__all__
            ]

    assert unexported == []
