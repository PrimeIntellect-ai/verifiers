import sys
from pathlib import Path

ENVIRONMENTS_DIR = Path(__file__).resolve().parents[1]
if str(ENVIRONMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENTS_DIR))

from emulator_common import load_emulator_environment  # noqa: E402

MANIFEST_PATH = Path(__file__).resolve().parent / "tasks" / "manifest.json"


def load_environment(config=None, **kwargs):
    return load_emulator_environment(MANIFEST_PATH, config=config, **kwargs)
