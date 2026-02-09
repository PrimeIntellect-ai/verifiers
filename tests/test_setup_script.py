from __future__ import annotations

from pathlib import Path

from verifiers.scripts import setup


def _fake_download_factory(downloaded: list[tuple[str, str]]):
    def _download(src: str, dst: str) -> str:
        downloaded.append((src, dst))
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(f"downloaded from {src}\n")
        return dst

    return _download


def test_dedupe_config_destinations_preserves_first_destination() -> None:
    configs = [
        ("repo-a", "configs/a.toml", "configs/out.toml"),
        ("repo-b", "configs/b.toml", "configs/other.toml"),
        ("repo-c", "configs/c.toml", "configs/out.toml"),
    ]
    deduped = setup._dedupe_config_destinations(configs)
    assert deduped == configs[:2]


def test_run_setup_downloads_endpoints_toml_and_default_config_sets(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    downloaded: list[tuple[str, str]] = []
    config_batches: list[list[tuple[str, str, str]]] = []

    monkeypatch.setattr(setup.wget, "download", _fake_download_factory(downloaded))
    monkeypatch.setattr(
        setup,
        "download_configs",
        lambda configs: config_batches.append(list(configs)),
    )
    monkeypatch.setattr(setup, "sync_prime_skills", lambda: None)

    setup.run_setup(skip_install=True, skip_agents_md=True)

    expected_configs = setup._dedupe_config_destinations(
        setup.GEPA_CONFIGS + setup.EVAL_CONFIGS + setup.RL_CONFIGS
    )
    assert downloaded == [(setup.ENDPOINTS_SRC, setup.ENDPOINTS_DST)]
    assert config_batches == [expected_configs]


def test_run_setup_with_prime_rl_downloads_prime_configs_plus_shared_configs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    downloaded: list[tuple[str, str]] = []
    config_batches: list[list[tuple[str, str, str]]] = []

    monkeypatch.setattr(setup.wget, "download", _fake_download_factory(downloaded))
    monkeypatch.setattr(
        setup,
        "download_configs",
        lambda configs: config_batches.append(list(configs)),
    )
    monkeypatch.setattr(setup, "install_prime_rl", lambda: None)
    monkeypatch.setattr(setup, "install_environments_to_prime_rl", lambda: None)
    monkeypatch.setattr(setup, "sync_prime_skills", lambda: None)

    setup.run_setup(skip_install=True, skip_agents_md=True, prime_rl=True)

    expected_configs = setup._dedupe_config_destinations(
        setup.PRIME_RL_CONFIGS + setup.GEPA_CONFIGS + setup.EVAL_CONFIGS
    )
    assert downloaded == [(setup.ENDPOINTS_SRC, setup.ENDPOINTS_DST)]
    assert config_batches == [expected_configs]


def test_sync_prime_skills_creates_dot_prime_tree(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, "LAB_SKILLS", ["create-environments", "brainstorm"])

    downloaded: list[tuple[str, str]] = []
    monkeypatch.setattr(setup.wget, "download", _fake_download_factory(downloaded))

    setup.sync_prime_skills()

    assert (
        tmp_path / ".prime" / "skills" / "create-environments" / "SKILL.md"
    ).exists()
    assert (tmp_path / ".prime" / "skills" / "brainstorm" / "SKILL.md").exists()
    assert downloaded == [
        (
            "https://raw.githubusercontent.com/primeintellect-ai/verifiers/refs/heads/main/skills/create-environments/SKILL.md",
            ".prime/skills/create-environments/SKILL.md",
        ),
        (
            "https://raw.githubusercontent.com/primeintellect-ai/verifiers/refs/heads/main/skills/brainstorm/SKILL.md",
            ".prime/skills/brainstorm/SKILL.md",
        ),
    ]
