"""The run identity the launcher hands every process of a run."""

import os


def run_id() -> str:
    """``$VF_RUN_ID``: set by every entrypoint (the `vf-*` CLIs and the prime-rl launchers)
    and inherited by spawned env servers and pool workers."""
    run_id = os.environ.get("VF_RUN_ID")
    if not run_id:
        raise RuntimeError("VF_RUN_ID is unset")
    return run_id
