"""The run identity the launcher hands every process of a run."""

import logging
import os
import uuid

logger = logging.getLogger(__name__)


def run_id() -> str:
    """``$VF_RUN_ID``: set by every entrypoint (the `vf-*` CLIs and the prime-rl launchers)
    and inherited by spawned env servers and pool workers. A process started without one
    mints its own and exports it, so it and its children form a run of their own."""
    run_id = os.environ.get("VF_RUN_ID")
    if not run_id:
        run_id = os.environ["VF_RUN_ID"] = uuid.uuid4().hex
        logger.warning(
            "VF_RUN_ID is unset - run-scoped state such as creation limiters covers only "
            "this process and its children (minted %s)",
            run_id,
        )
    return run_id
