from verifiers.scripts.tui import PRIME_EVAL_VIEW_MESSAGE, main


def test_vf_tui_points_to_prime_eval_view(capsys) -> None:
    main()

    output = capsys.readouterr()
    assert output.out.strip() == PRIME_EVAL_VIEW_MESSAGE
    assert "prime eval view" in output.out
