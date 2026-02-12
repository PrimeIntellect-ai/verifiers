from __future__ import annotations

import json
import os
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


def test_prepare_agent_skill_dirs_materializes_skills_from_prime(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, "LAB_SKILLS", ["create-environments", "brainstorm"])

    for skill_name in setup.LAB_SKILLS:
        skill_dir = tmp_path / ".prime" / "skills" / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(f"{skill_name}\n")

    setup._prepare_agent_skill_dirs(["codex"])

    for skill_name in setup.LAB_SKILLS:
        source = tmp_path / ".prime" / "skills" / skill_name
        target = tmp_path / ".codex" / "skills" / skill_name
        assert target.exists()
        assert (target / "SKILL.md").exists()
        if target.is_symlink():
            assert target.resolve() == source.resolve()


def test_prepare_agent_skill_dirs_is_safe_with_existing_links(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, "LAB_SKILLS", ["create-environments"])

    source = tmp_path / ".prime" / "skills" / "create-environments"
    source.mkdir(parents=True, exist_ok=True)
    (source / "SKILL.md").write_text("create-environments\n")

    target = tmp_path / ".codex" / "skills" / "create-environments"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(
        os.path.relpath(source, start=target.parent),
        target_is_directory=True,
    )

    setup._prepare_agent_skill_dirs(["codex"])

    assert target.is_symlink()
    assert target.resolve() == source.resolve()
    assert (target / "SKILL.md").exists()


def test_prepare_agent_skill_dirs_uses_mapped_root_for_amp(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, "LAB_SKILLS", ["create-environments"])

    source = tmp_path / ".prime" / "skills" / "create-environments"
    source.mkdir(parents=True, exist_ok=True)
    (source / "SKILL.md").write_text("create-environments\n")

    setup._prepare_agent_skill_dirs(["amp"])

    assert (
        tmp_path / ".agents" / "skills" / "create-environments" / "SKILL.md"
    ).exists()
    assert not (tmp_path / ".amp" / "skills").exists()


def test_prepare_agent_skill_dirs_supports_skill_name_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, "LAB_SKILLS", ["create-environments"])
    monkeypatch.setattr(
        setup,
        "AGENT_SKILL_NAME_MAP",
        {"amp": {"create-environments": "create-envs"}},
    )

    source = tmp_path / ".prime" / "skills" / "create-environments"
    source.mkdir(parents=True, exist_ok=True)
    (source / "SKILL.md").write_text("create-environments\n")

    setup._prepare_agent_skill_dirs(["amp"])

    assert (tmp_path / ".agents" / "skills" / "create-envs" / "SKILL.md").exists()
    assert not (tmp_path / ".agents" / "skills" / "create-environments").exists()


def test_run_setup_prints_post_setup_call_to_action(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(setup.wget, "download", _fake_download_factory([]))
    monkeypatch.setattr(setup, "download_configs", lambda *_: None)
    monkeypatch.setattr(setup, "sync_prime_skills", lambda: None)

    setup.run_setup(skip_install=True, skip_agents_md=True)

    output = capsys.readouterr().out
    assert "Prepared .codex/skills" in output
    assert output.index("Prepared .codex/skills") < output.index("get started")
    assert "get started" in output
    assert "quick commands" in output
    assert "ask codex" in output
    assert "example prompt" not in output
    assert 'ask codex: "I want to train a model for my task domain.' not in output
    assert "idea -> environment -> eval -> training" in output
    assert "prime env init my-env" in output
    assert "prime env install my-env" not in output
    assert "prime eval run my-env -m gpt-5-nano -n 5" in output
    assert "prime eval tui" in output
    assert "prime rl run configs/rl/wiki-search.toml" in output
    assert "prime gepa run my-env -m gpt-5-nano" in output


def test_run_setup_prints_prime_rl_post_setup_call_to_action(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(setup.wget, "download", _fake_download_factory([]))
    monkeypatch.setattr(setup, "download_configs", lambda *_: None)
    monkeypatch.setattr(setup, "sync_prime_skills", lambda: None)
    monkeypatch.setattr(setup, "install_prime_rl", lambda: None)
    monkeypatch.setattr(setup, "install_environments_to_prime_rl", lambda: None)

    setup.run_setup(skip_install=True, skip_agents_md=True, prime_rl=True)

    output = capsys.readouterr().out
    assert "get started" in output
    assert "quick commands" in output
    assert "ask codex" in output
    assert "example prompt" not in output
    assert "prime env install my-env" not in output
    assert "uv run prime-rl configs/prime-rl/wiki-search.toml" in output
    assert "prime rl run configs/rl/wiki-search.toml" not in output


def test_run_setup_persists_lab_choices_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(setup.wget, "download", _fake_download_factory([]))
    monkeypatch.setattr(setup, "download_configs", lambda *_: None)
    monkeypatch.setattr(setup, "sync_prime_skills", lambda: None)

    setup.run_setup(
        skip_install=True,
        skip_agents_md=True,
        agents="codex,cursor,codex",
        no_interactive=True,
    )

    metadata_path = tmp_path / ".prime" / "lab.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["setup_source"] == "prime lab setup"
    assert metadata["choices"] == {
        "agents": ["codex", "cursor"],
        "primary_agent": "codex",
        "use_multiple_agents": True,
    }


def test_run_setup_persists_default_lab_choices_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(setup.wget, "download", _fake_download_factory([]))
    monkeypatch.setattr(setup, "download_configs", lambda *_: None)
    monkeypatch.setattr(setup, "sync_prime_skills", lambda: None)

    setup.run_setup(skip_install=True, skip_agents_md=True, no_interactive=True)

    metadata_path = tmp_path / ".prime" / "lab.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["setup_source"] == "prime lab setup"
    assert metadata["choices"] == {
        "agents": ["codex"],
        "primary_agent": "codex",
        "use_multiple_agents": False,
    }
