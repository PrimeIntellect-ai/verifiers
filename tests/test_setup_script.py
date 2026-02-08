from __future__ import annotations

from pathlib import Path

from verifiers.scripts import setup


def test_run_setup_downloads_endpoints_toml_and_rl_plus_gepa_configs(
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

    assert downloaded == [(setup.ENDPOINTS_SRC, setup.ENDPOINTS_DST)]
    assert config_batches == [setup.GEPA_CONFIGS, setup.RL_CONFIGS]


def test_run_setup_with_prime_rl_downloads_prime_configs_only(
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

    assert downloaded == [(setup.ENDPOINTS_SRC, setup.ENDPOINTS_DST)]
    assert config_batches == [setup.PRIME_RL_CONFIGS, setup.GEPA_CONFIGS]
