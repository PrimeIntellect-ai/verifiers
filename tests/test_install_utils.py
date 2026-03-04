import sys
from unittest.mock import MagicMock, patch

import pytest

from verifiers.utils.install_utils import (
    check_hub_env_installed,
    is_hub_env,
    is_installed,
    normalize_package_name,
    parse_env_id,
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


class TestIsInstalled:
    @patch("verifiers.utils.install_utils.subprocess.run")
    def test_installed_no_version_check(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Name: gsm8k\nVersion: 1.0.0\n"
        )
        assert is_installed("gsm8k") is True
        mock_run.assert_called_once_with(
            ["uv", "pip", "show", "--python", sys.executable, "gsm8k"],
            capture_output=True,
            text=True,
        )

    @patch("verifiers.utils.install_utils.subprocess.run")
    def test_not_installed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert is_installed("gsm8k") is False

    @patch("verifiers.utils.install_utils.subprocess.run")
    def test_installed_version_matches(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Name: gsm8k\nVersion: 1.0.0\n"
        )
        assert is_installed("gsm8k", version="1.0.0") is True

    @patch("verifiers.utils.install_utils.subprocess.run")
    def test_installed_version_mismatch(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Name: gsm8k\nVersion: 1.0.0\n"
        )
        assert is_installed("gsm8k", version="2.0.0") is False

    @patch("verifiers.utils.install_utils.subprocess.run")
    def test_latest_version_skips_check(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Name: gsm8k\nVersion: 1.0.0\n"
        )
        assert is_installed("gsm8k", version="latest") is True

    @patch("verifiers.utils.install_utils.subprocess.run")
    def test_normalizes_package_name(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        is_installed("my-package")
        mock_run.assert_called_once_with(
            ["uv", "pip", "show", "--python", sys.executable, "my_package"],
            capture_output=True,
            text=True,
        )

    @patch("verifiers.utils.install_utils.subprocess.run")
    def test_exception_returns_false(self, mock_run):
        mock_run.side_effect = Exception("subprocess error")
        assert is_installed("gsm8k") is False


class TestCheckHubEnvInstalled:
    @patch("verifiers.utils.install_utils.is_installed")
    def test_local_env_returns_true(self, mock_is_installed):
        result = check_hub_env_installed("gsm8k")
        assert result is True
        mock_is_installed.assert_not_called()

    @patch("verifiers.utils.install_utils.is_installed")
    def test_hub_env_installed(self, mock_is_installed):
        mock_is_installed.return_value = True
        result = check_hub_env_installed("primeintellect/gsm8k")
        assert result is True
        mock_is_installed.assert_called_once_with("gsm8k", None)

    @patch("verifiers.utils.install_utils.is_installed")
    def test_hub_env_not_installed(self, mock_is_installed):
        mock_is_installed.return_value = False
        result = check_hub_env_installed("primeintellect/gsm8k")
        assert result is False

    @patch("verifiers.utils.install_utils.is_installed")
    def test_hub_env_with_version(self, mock_is_installed):
        mock_is_installed.return_value = True
        result = check_hub_env_installed("primeintellect/gsm8k@1.0.0")
        assert result is True
        mock_is_installed.assert_called_once_with("gsm8k", "1.0.0")
