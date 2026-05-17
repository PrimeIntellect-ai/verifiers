"""Compatibility entrypoint for the retired Verifiers eval viewer."""

PRIME_EVAL_VIEW_MESSAGE = "vf-tui is deprecated. Use `prime eval view`."


def main() -> None:
    print(PRIME_EVAL_VIEW_MESSAGE)


if __name__ == "__main__":
    main()
