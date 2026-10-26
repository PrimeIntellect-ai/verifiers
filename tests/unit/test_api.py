"""The `verifiers.v1` namespace: its lazy names stay in sync with what type checkers
see, and the config tree imports without the runtime."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import verifiers.v1 as vf

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
import verifiers.v1.configs as configs
for module in pkgutil.walk_packages(configs.__path__, "verifiers.v1.configs."):
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
