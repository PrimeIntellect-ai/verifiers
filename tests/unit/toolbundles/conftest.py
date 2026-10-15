import textwrap
from pathlib import Path

import pytest

TOOLS = textwrap.dedent(
    '''
    from vf_tools import ToolError, new_id, tool


    @tool(app="crm")
    def crm_add_contact(state, name: str, email: str, tier: str = "standard") -> dict:
        """Add a contact to the CRM."""
        if any(c["email"] == email for c in state["crm"]["contacts"]):
            raise ToolError(f"a contact with email {email} exists")
        contact = {"id": new_id(state, "contact"), "name": name, "email": email, "tier": tier}
        state["crm"]["contacts"].append(contact)
        return contact


    @tool(app="crm")
    def crm_search_contacts(state, query: str, limit: int = 10) -> list:
        """Search contacts by name or email."""
        q = query.lower()
        found = [c for c in state["crm"]["contacts"] if q in c["name"].lower() or q in c["email"]]
        return found[:limit]


    @tool(app="email", schema={"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to"]})
    def email_send(state, **arguments):
        """Send an email."""
        state["email"]["sent"].append({"to": arguments["to"], "body": arguments.get("body", "")})
        return {"queued": True}


    @tool
    def broken(state) -> None:
        state["crm"]["contacts"].append("half-written")
        raise KeyError("bug")
    '''
)

SEED = '{"crm": {"contacts": [{"id": "contact_0", "name": "Ada Lovelace", "email": "ada@example.com", "tier": "gold"}]}, "email": {"sent": []}, "meta": {"now": "2026-10-15T09:00:00Z"}}'


def write_bundle(root: Path, name: str = "crm", interface: str = "direct") -> Path:
    bundle = root / name
    bundle.mkdir(parents=True)
    (bundle / "tools.py").write_text(TOOLS)
    (bundle / "bundle.toml").write_text(
        f'name = "{name}"\nversion = 2\ninterface = "{interface}"\n'
    )
    return bundle


@pytest.fixture
def bundle_dir(tmp_path) -> Path:
    return write_bundle(tmp_path / "bundles")


@pytest.fixture
def seed() -> dict:
    import json

    return json.loads(SEED)


@pytest.fixture
def make_bundle():
    return write_bundle
