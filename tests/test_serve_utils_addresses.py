"""Tests for router/worker address construction.

See issue #2030: `make_ipc_address` hardcoded `ipc:///tmp/...`. `ipc://` is a Unix domain
socket transport, so on Windows `EnvRouter.__init__` could not bind at all and the env
server died before a single rollout ran.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import zmq

from verifiers.utils.serve_utils import (
    _SUN_PATH_MAX,
    get_free_port,
    ipc_path_of,
    make_ipc_address,
)


class _ScriptedProbe:
    """A stand-in for a probe socket that reports a caller-chosen port."""

    def __init__(self, port):
        self.port = port
        self.closed = False

    def bind(self, address):
        pass

    def getsockname(self):
        return ("127.0.0.1", self.port)

    def close(self):
        self.closed = True


def _scripted_socket_module(ports):
    """Patch target for `serve_utils.socket` that hands out `ports` in order."""
    remaining = list(ports)
    made: list[_ScriptedProbe] = []

    module = MagicMock()
    module.AF_INET = 2
    module.SOCK_STREAM = 1

    def make_probe(family, kind):
        probe = _ScriptedProbe(remaining.pop(0))
        made.append(probe)
        return probe

    module.socket.side_effect = make_probe
    return module, made


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
    def test_uses_ipc_and_a_writable_directory(self):
        with patch.object(zmq, "has", return_value=True):
            address = make_ipc_address("abc123", "responses")
        assert address.startswith("ipc://")
        path = Path(ipc_path_of(address))
        assert path.name == "vf-abc123-responses"
        # not a hardcoded /tmp: whatever directory is chosen has to be writable
        assert os.access(path.parent, os.W_OK)

    def test_falls_back_to_gettempdir_when_tmp_is_not_writable(self):
        """The reason the hardcoded /tmp went away in the first place."""
        with (
            patch.object(zmq, "has", return_value=True),
            patch("verifiers.utils.serve_utils.os.access", return_value=False),
        ):
            address = make_ipc_address("abc123", "responses")
        assert Path(ipc_path_of(address)).parent == Path(tempfile.gettempdir())

    def test_slashes_in_name_are_flattened(self):
        """Worker names embed the env id, which can contain a slash."""
        with patch.object(zmq, "has", return_value=True):
            address = make_ipc_address("abc123", "some/env-0")
        assert ipc_path_of(address) is not None
        assert Path(ipc_path_of(address)).name == "vf-abc123-some--env-0"


class TestIpcPathFitsInSunPath:
    """`sun_path` is 104 bytes on macOS, 108 on Linux. A worker name carries the env
    id, which is arbitrary and often an `org/name` pair, and macOS puts TMPDIR under a
    ~48 byte `/var/folders/...` path. Overflowing means the worker cannot bind after
    the router is already up, which is the late startup failure this fix removes.
    """

    LONG_TMPDIR = "/var/folders/qx/8m4n2z1d5kv7wcpr3hb9xlk80000gn/T"

    def _address_under(self, tmpdir, name, tmp_writable=False):
        """Build an address as if gettempdir() were `tmpdir`, /tmp optionally absent."""
        with (
            patch.object(zmq, "has", return_value=True),
            patch(
                "verifiers.utils.serve_utils.tempfile.gettempdir", return_value=tmpdir
            ),
            patch(
                "verifiers.utils.serve_utils.os.access",
                side_effect=lambda p, _: Path(p) != Path("/tmp") or tmp_writable,
            ),
        ):
            return make_ipc_address("abc123", name)

    def test_a_long_temp_dir_is_passed_over_for_tmp(self):
        """macOS: the per-user TMPDIR costs 44 bytes of name budget for nothing."""
        address = self._address_under(self.LONG_TMPDIR, "org/env", tmp_writable=True)
        assert Path(ipc_path_of(address)).parent == Path("/tmp")

    def test_a_long_name_is_shortened_rather_than_overflowing(self):
        env_id = "some-organization/a-fairly-long-environment-name-v2-hard"
        address = self._address_under(self.LONG_TMPDIR, f"{env_id}-0")
        assert len(ipc_path_of(address).encode()) <= _SUN_PATH_MAX

    def test_shortened_names_stay_distinct_between_workers(self):
        """Truncation alone would collide: the ids differ only past the cut."""
        env_id = "some-organization/a-fairly-long-environment-name-v2-hard"
        addresses = {
            self._address_under(self.LONG_TMPDIR, f"{env_id}-{worker}")
            for worker in range(4)
        }
        assert len(addresses) == 4

    def test_a_name_that_already_fits_is_left_alone(self):
        """No digest suffix on the common case: the paths stay readable."""
        address = self._address_under("/tmp", "env-0", tmp_writable=True)
        assert Path(ipc_path_of(address)) == Path("/tmp/vf-abc123-env-0")

    def test_a_directory_with_no_room_left_fails_loudly(self):
        """Better to say so than to hand back a path that cannot bind."""
        with pytest.raises(RuntimeError, match="cannot fit a socket name"):
            self._address_under("/" + "d" * _SUN_PATH_MAX, "env-0")


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


class TestPortIsNotReissuedBeforeItIsBound:
    """A worker address is built in the parent but bound in the child, after
    `load_environment`. The port is free for that whole window, so a later
    allocation must not be handed it again: the second worker would die on
    `Address already in use`, which is the crash this fix exists to remove.
    """

    def test_get_free_port_skips_an_excluded_port(self):
        """The OS re-offering a port already issued must not be passed through."""
        module, made = _scripted_socket_module([5001, 5001, 5002])
        with patch("verifiers.utils.serve_utils.socket", module):
            port = get_free_port(exclude={5001})
        assert port == 5002
        # every probe is closed, including the ones held open to force a new offer
        assert all(probe.closed for probe in made)

    def test_get_free_port_returns_the_first_acceptable_port(self):
        module, made = _scripted_socket_module([5001])
        with patch("verifiers.utils.serve_utils.socket", module):
            assert get_free_port(exclude={5002}) == 5001
        assert made[0].closed

    def test_get_free_port_gives_up_rather_than_returning_a_used_port(self):
        module, _ = _scripted_socket_module([5001] * 200)
        with (
            patch("verifiers.utils.serve_utils.socket", module),
            pytest.raises(RuntimeError, match="no free port"),
        ):
            get_free_port(exclude={5001})

    def test_make_ipc_address_records_the_port_it_issued(self):
        issued: set[int] = set()
        module, _ = _scripted_socket_module([5001])
        with (
            patch.object(zmq, "has", return_value=False),
            patch("verifiers.utils.serve_utils.socket", module),
        ):
            make_ipc_address("abc123", "responses", issued)
        assert issued == {5001}

    def test_a_shared_set_stops_the_second_worker_reusing_the_first_port(self):
        """The exact race: the OS offers 5001 twice because worker 0 has not bound it."""
        issued: set[int] = set()
        module, _ = _scripted_socket_module([5001, 5001, 5002])
        with (
            patch.object(zmq, "has", return_value=False),
            patch("verifiers.utils.serve_utils.socket", module),
        ):
            first = make_ipc_address("abc123", "env-0", issued)
            second = make_ipc_address("abc123", "env-1", issued)
        assert first == "tcp://127.0.0.1:5001"
        assert second == "tcp://127.0.0.1:5002"
        assert issued == {5001, 5002}

    def test_ipc_path_callers_are_unaffected(self):
        """The set is only touched on the tcp path; ipc addresses use no port."""
        issued: set[int] = set()
        with patch.object(zmq, "has", return_value=True):
            make_ipc_address("abc123", "responses", issued)
        assert issued == set()
