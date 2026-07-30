"""Tests for router/worker address construction.

See issue #2030: `make_ipc_address` hardcoded `ipc:///tmp/...`. `ipc://` is a Unix domain
socket transport, so on Windows `EnvRouter.__init__` could not bind at all and the env
server died before a single rollout ran.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import zmq

from verifiers.utils.serve_utils import ipc_path_of, make_ipc_address


class TestIpcPathOf:
    def test_returns_path_for_ipc_address(self):
        assert ipc_path_of("ipc:///tmp/vf-abc-responses") == "/tmp/vf-abc-responses"

    @pytest.mark.parametrize(
        "address",
        ["tcp://127.0.0.1:5555", "inproc://x", "pgm://eth0;239.0.0.1:5555"],
    )
    def test_returns_none_for_non_ipc_address(self, address):
        """Only ipc:// has a file to unlink; anything else must not reach os.unlink."""
        assert ipc_path_of(address) is None


class TestMakeIpcAddressWithIpcSupport:
    def test_uses_ipc_and_the_resolved_temp_dir(self):
        with patch.object(zmq, "has", return_value=True):
            address = make_ipc_address("abc123", "responses")
        assert address.startswith("ipc://")
        # not a hardcoded /tmp: it has to follow tempfile.gettempdir()
        expected = Path(tempfile.gettempdir()) / "vf-abc123-responses"
        assert address == f"ipc://{expected}"

    def test_slashes_in_name_are_flattened(self):
        """Worker names embed the env id, which can contain a slash."""
        with patch.object(zmq, "has", return_value=True):
            address = make_ipc_address("abc123", "some/env-0")
        assert ipc_path_of(address) is not None
        assert Path(ipc_path_of(address)).name == "vf-abc123-some--env-0"


class TestMakeIpcAddressWithoutIpcSupport:
    """The Windows case, and any libzmq built without ipc."""

    def test_falls_back_to_loopback_tcp(self):
        with patch.object(zmq, "has", return_value=False):
            address = make_ipc_address("abc123", "responses")
        assert address.startswith("tcp://127.0.0.1:")
        assert int(address.rsplit(":", 1)[1]) > 0

    def test_fallback_address_has_no_file_to_unlink(self):
        with patch.object(zmq, "has", return_value=False):
            address = make_ipc_address("abc123", "responses")
        assert ipc_path_of(address) is None

    def test_distinct_names_get_distinct_ports(self):
        """Router and each worker bind separately, so addresses must not collide."""
        with patch.object(zmq, "has", return_value=False):
            addresses = {
                make_ipc_address("abc123", name)
                for name in ("responses", "stats", "env-0", "env-1")
            }
        assert len(addresses) == 4

    def test_fallback_address_actually_binds(self):
        """The point of the fix: the returned address must be bindable."""
        with patch.object(zmq, "has", return_value=False):
            address = make_ipc_address("abc123", "responses")
        ctx = zmq.Context()
        try:
            socket = ctx.socket(zmq.PULL)
            try:
                socket.bind(address)
            finally:
                socket.close()
        finally:
            ctx.term()

    def test_hardcoded_tmp_ipc_address_is_never_returned(self):
        """Regression guard for the exact string in issue #2030."""
        with patch.object(zmq, "has", return_value=False):
            address = make_ipc_address("abc123", "responses")
        assert not address.startswith("ipc:///tmp/")
