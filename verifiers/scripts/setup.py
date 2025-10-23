import os

import wget

REPO = "primeintellect-ai/verifiers"
ENDPOINTS_SRC = (
    f"https://raw.githubusercontent.com/{REPO}/refs/heads/main/configs/endpoints.py"
)
ENDPOINTS_DST = "configs/endpoints.py"

ZERO3_SRC = (
    f"https://raw.githubusercontent.com/{REPO}/refs/heads/main/configs/zero3.yaml"
)
ZERO3_DST = "configs/zero3.yaml"


CONFIGS = [
    "alphabet-sort",
    "gsm8k",
    "math-python",
    "reverse-text",
    "wordle",
    "tool-test",
]

CONFIGS_SRC = [
    f"https://raw.githubusercontent.com/{REPO}/refs/heads/main/configs/rl/{config}.toml"
    for config in CONFIGS
]
CONFIGS_DST = [f"configs/rl/{config}.toml" for config in CONFIGS]


def main():
    os.makedirs("configs", exist_ok=True)
    os.makedirs("configs/rl", exist_ok=True)
    # create configs/endpoints.py if it doesn't exist
    if not os.path.exists(ENDPOINTS_DST):
        wget.download(ENDPOINTS_SRC, ENDPOINTS_DST)
        print(f"\nDownloaded {ENDPOINTS_DST} from https://github.com/{REPO}")

    else:
        print(f"{ENDPOINTS_DST} already exists")

    # create configs/zero3.yaml if it doesn't exist
    if not os.path.exists(ZERO3_DST):
        # create it
        wget.download(ZERO3_SRC, ZERO3_DST)
        print(f"\nDownloaded {ZERO3_DST} from https://github.com/{REPO}")
    else:
        print(f"{ZERO3_DST} already exists")

    # download all configs
    for src, dst in zip(CONFIGS_SRC, CONFIGS_DST):
        if not os.path.exists(dst):
            wget.download(src, dst)
            print(f"\nDownloaded {dst} from https://github.com/{REPO}")
        else:
            print(f"{dst} already exists")


if __name__ == "__main__":
    main()
