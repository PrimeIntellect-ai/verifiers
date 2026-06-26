import sys
import tarfile
from unittest.mock import patch

import pytest

from verifiers.utils.install_utils import (
    download_environment_source,
    install_from_hub,
    is_hub_env,
    normalize_package_name,
    parse_env_id,
    safe_extract,
)


class TestNormalizePackageName:
    def test_converts_hyphens_to_underscores(self):
        assert normalize_package_name("my-package") == "my_package"

    def test_lowercases(self):
        assert normalize_package_name("MyPackage") == "mypackage"

    def test_combined(self):
        assert normalize_package_name("My-Package-Name") == "my_package_name"

    def test_already_normalized(self):
        assert normalize_package_name("my_package") == "my_package"


class TestParseEnvId:
    def test_owner_and_name(self):
        owner, name, version = parse_env_id("primeintellect/gsm8k")
        assert owner == "primeintellect"
        assert name == "gsm8k"
        assert version is None

    def test_owner_name_and_version(self):
        owner, name, version = parse_env_id("primeintellect/gsm8k@1.0.0")
        assert owner == "primeintellect"
        assert name == "gsm8k"
        assert version == "1.0.0"

    def test_version_with_at_sign(self):
        owner, name, version = parse_env_id("owner/name@1.0.0")
        assert owner == "owner"
        assert name == "name"
        assert version == "1.0.0"

    def test_invalid_no_slash(self):
        with pytest.raises(ValueError, match="Invalid environment ID"):
            parse_env_id("gsm8k")

    def test_invalid_empty_owner(self):
        with pytest.raises(ValueError, match="Invalid environment ID"):
            parse_env_id("/gsm8k")

    def test_invalid_empty_name(self):
        with pytest.raises(ValueError, match="Invalid environment ID"):
            parse_env_id("owner/")

    def test_invalid_too_many_slashes(self):
        with pytest.raises(ValueError, match="Invalid environment ID"):
            parse_env_id("a/b/c")


class TestIsHubEnv:
    def test_hub_env_with_owner(self):
        assert is_hub_env("primeintellect/gsm8k") is True

    def test_hub_env_with_version(self):
        assert is_hub_env("primeintellect/gsm8k@1.0.0") is True

    def test_local_env_no_slash(self):
        assert is_hub_env("gsm8k") is False

    def test_local_env_relative_path(self):
        assert is_hub_env("./environments/gsm8k") is False

    def test_local_env_absolute_path(self):
        assert is_hub_env("/path/to/gsm8k") is False


class TestInstallFromHub:
    @patch("verifiers.utils.install_utils.subprocess.run")
    @patch("verifiers.utils.install_utils.fetch_hub_environment")
    def test_public_index_uses_targeted_uv_install(self, mock_fetch, mock_run):
        mock_fetch.return_value = {
            "simple_index_url": "https://hub.example/simple",
            "url_dependencies": ["dep @ https://example.test/dep.whl"],
        }

        module = install_from_hub("owner/my-env@1.2.3", prerelease=True)

        assert module == "my_env"
        mock_run.assert_called_once_with(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "-P",
                "my_env",
                "my_env==1.2.3",
                "dep @ https://example.test/dep.whl",
                "--extra-index-url",
                "https://hub.example/simple",
                "--exclude-newer-package",
                "my_env=false",
                "--prerelease=allow",
            ],
            check=True,
        )

    @patch("verifiers.utils.install_utils.subprocess.run")
    @patch("verifiers.utils.install_utils.build_environment_wheel")
    @patch("verifiers.utils.install_utils.download_environment_source")
    @patch("verifiers.utils.install_utils.fetch_hub_environment")
    def test_private_source_is_downloaded_and_built(
        self, mock_fetch, mock_download, mock_build, mock_run, tmp_path, monkeypatch
    ):
        content_hash = "a" * 64
        mock_fetch.return_value = {
            "visibility": "PRIVATE",
            "package_url": "https://hub.example/private.tar.gz",
            "sha256": content_hash,
        }
        wheel = tmp_path / "my_env-1.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        mock_download.side_effect = (
            lambda details, destination, api_key=None, base_url=None: destination
        )
        mock_build.return_value = wheel
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        module = install_from_hub("owner/my-env")

        assert module == "my_env"
        mock_download.assert_called_once()
        assert mock_build.call_args.args[1] == (
            tmp_path
            / ".cache"
            / "verifiers"
            / "hub"
            / "owner"
            / "my-env"
            / content_hash
        )
        assert mock_run.call_args.args[0][-1] == str(wheel)


def test_hub_install_cache_tracks_the_active_owner(monkeypatch):
    from verifiers.v1.utils import install as install_module

    calls = []
    monkeypatch.setattr(install_module, "install_from_hub", calls.append)
    monkeypatch.setattr(install_module, "_installed_hub_refs", {})

    assert install_module.ensure_installed("alice/shared-env") == "shared_env"
    assert install_module.ensure_installed("alice/shared-env") == "shared_env"
    assert install_module.ensure_installed("bob/shared-env") == "shared_env"
    assert install_module.ensure_installed("alice/shared-env") == "shared_env"

    assert calls == ["alice/shared-env", "bob/shared-env", "alice/shared-env"]


def test_safe_extract_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../evil.txt")
        member.size = 0
        tar.addfile(member)

    destination = tmp_path / "destination"
    destination.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(ValueError, match="unsafe path"):
            safe_extract(tar, destination)


def test_download_environment_source_flattens_archive_wrapper(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    (source / "demo.py").write_text("VALUE = 1\n")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="demo-1.0")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self, chunk_size):
            yield archive.read_bytes()

    monkeypatch.setattr("httpx.stream", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        "verifiers.utils.client_utils.load_prime_config",
        lambda: {"api_key": None, "base_url": "https://api.example.test"},
    )

    destination = download_environment_source(
        {"package_url": "https://storage.example.test/demo.tar.gz"},
        tmp_path / "destination",
    )

    assert (destination / "pyproject.toml").is_file()
    assert (destination / "demo.py").is_file()
    assert not (destination / "demo-1.0").exists()
