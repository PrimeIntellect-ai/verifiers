from harnesses import OpenCode, OpenCodeConfig


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)
