"""EnvServer wiring: archive_dir and archive reach the loaded env."""

from pathlib import Path

import pytest

from verifiers.v1.serve.server import EnvServer
from verifiers.v1.utils.loaders import resolve_env_config


def test_env_server_sets_archive(tmp_path: Path) -> None:
    config = resolve_env_config({"taskset": {"id": "echo-v1"}})
    archive_dir = tmp_path / "artifacts"
    server = EnvServer(
        config=config,
        address="tcp://127.0.0.1:0",
        archive_dir=str(archive_dir),
        archive={"max_mb": 1, "extra": ["/usr"]},
    )
    try:
        assert server.env.archive_dir == archive_dir
        assert server.env.archive_config.max_mb == 1
        assert server.env.archive_config.extra == ["/usr"]
    finally:
        server.frontend.close()
        server.ctx.term()


@pytest.mark.e2e
@pytest.mark.docker
@pytest.mark.parametrize("harness,harness_runtime", [("bash", "docker")], indirect=True)
async def test_serve_archive(run_v1_server, harness, harness_runtime, tmp_path):
    """Archive dumps land under the run dir when the env is served, not in-process."""
    (trace,) = await run_v1_server(
        "echo-agentic-v1",
        harness=harness,
        runtime={"type": harness_runtime},
        output_dir=tmp_path,
        max_turns=10,
        max_tokens=8192,
    )
    assert trace.ok
    manifests = list((tmp_path / "artifacts").glob(f"*/{trace.id}/manifest.json"))
    assert len(manifests) == 1, f"expected one manifest for trace {trace.id}, got {manifests}"
