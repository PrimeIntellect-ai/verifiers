"""The scope that state shared by a run's processes is keyed by."""

import logging
import os
import uuid

logger = logging.getLogger(__name__)

_process_scope: str | None = None


def run_scope() -> str:
    """``$VF_RUN_ID``, set by every entrypoint (the `vf-*` CLIs and the prime-rl launchers)
    and inherited by spawned env servers and pool workers. A process started without one
    gets a scope of its own, minted once per process."""
    global _process_scope
    run_id = os.environ.get("VF_RUN_ID")
    if run_id:
        return run_id
    if _process_scope is None:
        _process_scope = uuid.uuid4().hex
        logger.warning(
            "VF_RUN_ID is unset - run-scoped state such as creation limiters covers only "
            "this process (scope %s)",
            _process_scope,
        )
    return _process_scope
