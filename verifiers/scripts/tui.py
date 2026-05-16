"""Compatibility entrypoint for the retired Verifiers eval viewer."""

PRIME_EVAL_TUI_MESSAGE = "vf-tui has moved. Use `prime eval tui`."


def main() -> None:
    print(PRIME_EVAL_TUI_MESSAGE)


if __name__ == "__main__":
    main()
