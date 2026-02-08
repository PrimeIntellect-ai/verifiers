from __future__ import annotations

from pathlib import Path

from verifiers.scripts import setup


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

    monkeypatch.setattr(
        setup.wget, "download", lambda src, dst: downloaded.append((src, dst))
    )
    monkeypatch.setattr(
        setup,
        "download_configs",
        lambda configs: config_batches.append(list(configs)),
    )

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

    monkeypatch.setattr(
        setup.wget, "download", lambda src, dst: downloaded.append((src, dst))
    )
    monkeypatch.setattr(
        setup,
        "download_configs",
        lambda configs: config_batches.append(list(configs)),
    )
    monkeypatch.setattr(setup, "install_prime_rl", lambda: None)
    monkeypatch.setattr(setup, "install_environments_to_prime_rl", lambda: None)

    setup.run_setup(skip_install=True, skip_agents_md=True, prime_rl=True)

    expected_configs = setup._dedupe_config_destinations(
        setup.PRIME_RL_CONFIGS + setup.GEPA_CONFIGS + setup.EVAL_CONFIGS
    )
    assert downloaded == [(setup.ENDPOINTS_SRC, setup.ENDPOINTS_DST)]
    assert config_batches == [expected_configs]
