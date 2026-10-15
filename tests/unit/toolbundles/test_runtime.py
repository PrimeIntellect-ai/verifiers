"""The stdlib bundle runtime: tool calls on a state, both interfaces, the MCP server
(reached by the real MCP client), and the CLI."""

import json
import subprocess
import sys
import threading

import pytest

from verifiers.v1.toolbundles import vf_tools
from verifiers.v1.toolbundles.vf_tools import Bundle, Session


def test_signature_derives_the_argument_schema(bundle_dir):
    bundle = Bundle.load(bundle_dir)
    add = bundle.tools["crm_add_contact"].spec()
    assert add["app"] == "crm"
    assert add["inputSchema"] == {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "tier": {"type": "string", "default": "standard"},
        },
        "required": ["name", "email"],
    }
    assert bundle.tools["email_send"].schema["required"] == ["to"]


def test_a_call_changes_the_state_and_ids_are_deterministic(bundle_dir, seed):
    commits = []
    session = Session(Bundle.load(bundle_dir), seed, on_commit=commits.append)
    ok, contact = session.call(
        "crm_add_contact", {"name": "Grace Hopper", "email": "grace@example.com"}
    )
    assert ok and contact["id"] == "contact_1"
    assert session.state["crm"]["contacts"][-1] == contact
    assert len(commits) == 1
    ok, found = session.call("crm_search_contacts", {"query": "grace"})
    assert ok and found == [contact]
    assert len(commits) == 1  # a read commits nothing


def test_a_refused_or_failing_call_leaves_the_state_unchanged(bundle_dir, seed):
    session = Session(Bundle.load(bundle_dir), seed)
    before = json.dumps(session.state, sort_keys=True)
    ok, message = session.call(
        "crm_add_contact", {"name": "Ada", "email": "ada@example.com"}
    )
    assert not ok and "exists" in message
    ok, message = session.call("broken")
    assert not ok and message.startswith("internal error: KeyError")
    assert json.dumps(session.state, sort_keys=True) == before


@pytest.mark.parametrize(
    "arguments",
    [{"name": "x"}, {"name": "x", "email": "y", "extra": 1}, {"name": 1, "email": "y"}],
    ids=["missing", "unknown", "mistyped"],
)
def test_arguments_are_checked_against_the_schema(bundle_dir, seed, arguments):
    ok, _ = Session(Bundle.load(bundle_dir), seed).call("crm_add_contact", arguments)
    assert not ok


def test_search_execute_advertises_two_tools_over_the_same_apis(bundle_dir, seed):
    bundle = Bundle.load(bundle_dir, interface="search_execute")
    assert [t["name"] for t in bundle.advertised()] == ["search_api", "execute"]
    session = Session(bundle, seed)
    ok, found = session.call("search_api", {"query": "send email"})
    assert ok and found[0]["name"] == "email_send"
    ok, _ = session.call(
        "execute", {"api": "email_send", "arguments": {"to": "ada@example.com"}}
    )
    assert ok and session.state["email"]["sent"] == [
        {"to": "ada@example.com", "body": ""}
    ]
    ok, message = session.call("execute", {"api": "nope"})
    assert not ok and "unknown API" in message


@pytest.fixture
def server(bundle_dir, seed, tmp_path):
    state_file = tmp_path / "state.json"
    vf_tools.write_state(str(state_file), seed)
    session = Session(
        Bundle.load(bundle_dir),
        seed,
        on_commit=lambda s: vf_tools.write_state(str(state_file), s),
    )
    httpd = vf_tools.make_server(session, "127.0.0.1", 0, "crm")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/mcp", state_file
    httpd.shutdown()


async def test_the_mcp_client_lists_and_calls_tools(server):
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    url, state_file = server
    async with Client(streamable_http_client(url)) as client:
        tools = (await client.list_tools()).tools
        assert {t.name for t in tools} >= {"crm_add_contact", "email_send"}
        result = await client.call_tool(
            "crm_add_contact", {"name": "Grace", "email": "grace@example.com"}
        )
        assert not result.is_error
        assert json.loads(result.content[0].text)["id"] == "contact_1"
        refused = await client.call_tool("crm_add_contact", {"name": "Grace"})
        assert refused.is_error
    assert json.loads(state_file.read_text())["crm"]["contacts"][-1]["name"] == "Grace"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, vf_tools.__file__, *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_cli_calls_seeds_and_dumps(server, tmp_path, seed):
    url, state_file = server
    called = run_cli("call", "email_send", '{"to": "ada@example.com"}', "--url", url)
    assert called.returncode == 0 and json.loads(called.stdout) == {"queued": True}
    refused = run_cli("call", "email_send", "{}", "--url", url)
    assert refused.returncode == 1
    dump = tmp_path / "out" / "crm.json"
    assert (
        run_cli("dump", "--state", str(state_file), "--out", str(dump)).returncode == 0
    )
    assert json.loads(dump.read_text())["email"]["sent"] == [
        {"to": "ada@example.com", "body": ""}
    ]
    fresh = tmp_path / "fresh.json"
    seed_file = tmp_path / "seed.json"
    seed_file.write_text(json.dumps(seed))
    assert (
        run_cli("seed", "--state", str(fresh), "--seed", str(seed_file)).returncode == 0
    )
    assert json.loads(fresh.read_text()) == seed
