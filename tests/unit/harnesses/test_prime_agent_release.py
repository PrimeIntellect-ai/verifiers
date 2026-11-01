"""Prime Agent's local `release`: one read per process, whatever the number of harness copies."""

import json

from verifiers.v1.harnesses.prime_agent import (
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
)


def test_harness_copies_install_the_build_first_read(tmp_path):
    def build(sha256):
        manifest = {"binaries": [{"platform": "linux-x64", "file": "prime-agent.tar.gz", "sha256": sha256}]}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    build("a" * 64)
    config = PrimeAgentHarnessConfig(release=tmp_path)
    first = PrimeAgentHarness(config)
    first_build = first._build
    # A dist rebuilt mid-run doesn't reach later copies: they install what the run started with,
    # sharing one read of it.
    build("b" * 64)
    second = PrimeAgentHarness(config)
    assert second._build == first_build
    assert second._local_release[1] is first._local_release[1]
