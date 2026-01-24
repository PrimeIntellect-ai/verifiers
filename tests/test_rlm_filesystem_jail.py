import os
import sysconfig
from pathlib import Path

import pytest

from verifiers.utils.rlm_filesystem_jail import FilesystemJail


@pytest.fixture
def restore_cwd():
    cwd = os.getcwd()
    try:
        yield
    finally:
        os.chdir(cwd)


def test_jail_blocks_outside_access(tmp_path: Path, restore_cwd):
    root = tmp_path / "root"
    root.mkdir()
    (root / "inside.txt").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    os.chdir(root)
    jail = FilesystemJail(str(root))
    try:
        jail.install()
        assert (root / "inside.txt").read_text(encoding="utf-8") == "inside"
        with pytest.raises(PermissionError):
            (Path(outside)).read_text(encoding="utf-8")
        with pytest.raises(PermissionError):
            os.listdir(tmp_path)
    finally:
        jail.uninstall()


def test_jail_allows_whitelisted_paths(tmp_path: Path, restore_cwd):
    root = tmp_path / "root"
    root.mkdir()
    allowed = tmp_path / "allowed.txt"
    allowed.write_text("ok", encoding="utf-8")
    blocked = tmp_path / "blocked.txt"
    blocked.write_text("no", encoding="utf-8")

    os.chdir(root)
    jail = FilesystemJail(str(root), allowed_paths=[str(allowed)])
    try:
        jail.install()
        assert allowed.read_text(encoding="utf-8") == "ok"
        with pytest.raises(PermissionError):
            blocked.read_text(encoding="utf-8")
    finally:
        jail.uninstall()


def test_jail_allows_stdlib_when_whitelisted(tmp_path: Path, restore_cwd):
    stdlib_path = sysconfig.get_paths().get("stdlib")
    if not stdlib_path:
        pytest.skip("stdlib path not available")
    stdlib_file = Path(stdlib_path) / "os.py"
    if not stdlib_file.exists():
        pytest.skip("stdlib file not found")

    root = tmp_path / "root"
    root.mkdir()

    os.chdir(root)
    jail = FilesystemJail(str(root), allowed_paths=[str(stdlib_path)])
    try:
        jail.install()
        assert stdlib_file.read_text(encoding="utf-8")
    finally:
        jail.uninstall()


def test_jail_blocks_disallowed_imports_and_builtins(tmp_path: Path, restore_cwd):
    root = tmp_path / "root"
    root.mkdir()

    os.chdir(root)
    jail = FilesystemJail(
        str(root),
        disallowed_modules={"os"},
        disallowed_builtins={"eval"},
    )
    try:
        jail.install()
        with pytest.raises(ImportError):
            __import__("os")
        with pytest.raises(NameError):
            eval("1+1")
    finally:
        jail.uninstall()

    assert eval("1+1") == 2
