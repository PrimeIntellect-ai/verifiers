from verifiers.scripts.tui import PRIME_EVAL_TUI_MESSAGE, main


def test_vf_tui_points_to_prime_eval_tui(capsys) -> None:
    main()

    output = capsys.readouterr()
    assert output.out.strip() == PRIME_EVAL_TUI_MESSAGE
    assert "prime eval tui" in output.out
