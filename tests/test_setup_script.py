import sys

import pytest

from verifiers.scripts import setup


def test_run_setup_warns_to_use_prime_lab_setup(capsys) -> None:
    exit_code = setup.run_setup(skip_install=True, prime_rl=True, agents="codex")

    output = capsys.readouterr()
    assert exit_code == 1
    assert output.out == ""
    assert "vf-setup is deprecated" in output.err
    assert "Upgrade the Prime CLI" in output.err
    assert "prime lab setup" in output.err


def test_main_warns_to_use_prime_lab_setup_and_exits(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["vf-setup", "--skip-install", "--prime-rl"])

    with pytest.raises(SystemExit) as exc_info:
        setup.main()

    output = capsys.readouterr()
    assert exc_info.value.code == 1
    assert output.out == ""
    assert "vf-setup is deprecated" in output.err
    assert "Upgrade the Prime CLI" in output.err
    assert "prime lab setup" in output.err
