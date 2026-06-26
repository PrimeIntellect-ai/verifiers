import subprocess
from pathlib import Path

import pytest

from verifiers.v1.tasksets.harbor import taskset as harbor_taskset


def test_harbor_cli_resolves_from_active_python_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, str] = {}

    def fake_which(name: str, path: str | None = None) -> str:
        seen["name"] = name
        seen["path"] = path or ""
        return "/active-env/bin/harbor"

    monkeypatch.setattr(harbor_taskset.shutil, "which", fake_which)

    assert harbor_taskset.harbor_cli() == "/active-env/bin/harbor"
    assert seen == {
        "name": "harbor",
        "path": str(Path(harbor_taskset.sys.executable).parent),
    }


def test_harbor_cli_missing_mentions_harbor_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(harbor_taskset.shutil, "which", lambda name, path=None: None)

    with pytest.raises(RuntimeError, match="uv sync --python 3.12 --extra harbor"):
        harbor_taskset.harbor_cli()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "dataset": "general-agent@2026-06-25",
                "repo": "PrimeIntellect-ai/research-environments@main",
                "registry_url": "https://example.com/registry.json",
            },
            "repo and registry_url are mutually exclusive",
        ),
        (
            {
                "dataset": "general-agent@2026-06-25",
                "registry_path": Path("registry.json"),
                "registry_url": "https://example.com/registry.json",
            },
            "do not combine local registry_path with registry_url",
        ),
        (
            {
                "dataset": "harbor/hello-world",
                "repo": "PrimeIntellect-ai/research-environments@main",
            },
            "dataset must be a bare name",
        ),
    ],
)
def test_registry_selector_validation(kwargs: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        harbor_taskset.HarborConfig(**kwargs)


def test_dataset_dir_invokes_harbor_download_with_registry_selectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_harbor = tmp_path / "bin" / "harbor"
    calls: list[list[str]] = []
    cache = tmp_path / "cache"
    monkeypatch.setattr(harbor_taskset, "CACHE", cache)
    monkeypatch.setattr(harbor_taskset, "harbor_cli", lambda: str(fake_harbor))

    def fake_run(command: list[str], check: bool):
        assert check is True
        calls.append(command)
        output_dir = Path(command[command.index("-o") + 1])
        task_dir = output_dir / "general-agent" / "3d_print_shop_t0"
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text('[task]\nname = "3d_print_shop_t0"\n')
        (task_dir / "instruction.md").write_text("Do the task.\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(harbor_taskset.subprocess, "run", fake_run)

    config = harbor_taskset.HarborConfig(
        dataset="general-agent@2026-06-25",
        repo="PrimeIntellect-ai/research-environments@a95c3e8",
        registry_path=Path("registry.json"),
    )
    root = harbor_taskset.dataset_dir(config)

    assert root == harbor_taskset.cache_dir(config)
    assert (root / "general-agent" / "3d_print_shop_t0" / "task.toml").is_file()
    assert calls == [
        [
            str(fake_harbor),
            "download",
            "general-agent@2026-06-25",
            "--export",
            "-o",
            calls[0][5],
            "--repo",
            "PrimeIntellect-ai/research-environments@a95c3e8",
            "--registry-path",
            "registry.json",
        ]
    ]

    calls.clear()
    assert harbor_taskset.dataset_dir(config) == root
    assert calls == []


def test_cache_dir_includes_registry_selector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(harbor_taskset, "CACHE", tmp_path)

    hub = harbor_taskset.cache_dir(
        harbor_taskset.HarborConfig(dataset="general-agent@2026-06-25")
    )
    repo_a = harbor_taskset.cache_dir(
        harbor_taskset.HarborConfig(
            dataset="general-agent@2026-06-25",
            repo="PrimeIntellect-ai/research-environments@a",
        )
    )
    repo_b = harbor_taskset.cache_dir(
        harbor_taskset.HarborConfig(
            dataset="general-agent@2026-06-25",
            repo="PrimeIntellect-ai/research-environments@b",
        )
    )

    assert len({hub, repo_a, repo_b}) == 3
