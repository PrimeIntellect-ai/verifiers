"""A box with no way to fetch Node.js gets the host's pinned tarball."""

from types import SimpleNamespace

from verifiers.v1.harnesses import node


class Box:
    def __init__(self, can_fetch: bool):
        self.can_fetch = can_fetch
        self.written: dict[str, bytes] = {}

    async def run(self, argv, env):
        if argv == ["uname", "-m"]:
            return SimpleNamespace(exit_code=0, stdout="x86_64\n", stderr="")
        assert argv[:2] == ["sh", "-c"] and argv[2] == node.CAN_FETCH
        return SimpleNamespace(
            exit_code=0 if self.can_fetch else 1, stdout="", stderr=""
        )

    async def write(self, path, data):
        self.written[path] = data


async def test_a_hardened_box_unpacks_the_hosts_node(monkeypatch):
    installs = []

    async def tarball(machine):
        assert machine == "x86_64\n"
        return b"node tarball"

    async def installed(runtime, **kwargs):
        installs.append(kwargs)

    monkeypatch.setattr(node, "host_node_tarball", tarball)
    monkeypatch.setattr(node, "ensure_installed", installed)
    box = Box(can_fetch=False)
    await node.ensure_node(box)  # type: ignore[arg-type]
    [(path, data)] = box.written.items()
    assert data == b"node tarball" and path.endswith(".tar.gz")
    assert installs[0]["install"] == node.UNPACK and installs[0]["env"] == {
        "VF_NODE_TARBALL": path
    }


async def test_a_box_that_can_fetch_node_installs_it_itself(monkeypatch):
    installs = []

    async def installed(runtime, **kwargs):
        installs.append(kwargs)

    monkeypatch.setattr(node, "ensure_installed", installed)
    box = Box(can_fetch=True)
    await node.ensure_node(box)  # type: ignore[arg-type]
    assert not box.written and installs[0]["install"] == node.INSTALL


def test_the_pinned_tarballs_have_digests():
    assert set(node.NODE_SHA256) == {"x64", "arm64"}
    assert node._ARCHES["x86_64"] == "x64" and node._ARCHES["aarch64"] == "arm64"
