"""`[[tools]]` in task.toml: parsing, and how a Harbor task compiles each bundle into an
MCP server entry and an end-state dump hook that the grader box never repeats."""

import tomllib

import pytest

from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask
from verifiers.v1.tasksets.harbor.taskset import parse_task, verifier_box_data
from verifiers.v1.toolbundles import BASE_PORT, bundle_digest, parse_mounts

TASK_TOML = """\
version = "1.0"

[environment]
docker_image = "python:3.12-slim"

[agent]
user = "agent"

[verifier]
environment_mode = "separate"

[[verifier.collect]]
command = "true"

{tools}
"""


def make_task(tmp_path, tools: str, write_bundle):
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "test.sh").write_text("exit 0\n")
    (task / "instruction.md").write_text("Do it.\n")
    (task / "task.toml").write_text(TASK_TOML.format(tools=tools))
    write_bundle(task / "tools", "crm", interface="search_execute")
    (task / "tools" / "seed").mkdir()
    (task / "tools" / "seed" / "crm.json").write_text("{}")
    return task


def test_a_bundle_compiles_to_an_mcp_server_and_a_dump_hook(tmp_path, make_bundle):
    task = make_task(tmp_path, '[[tools]]\nbundle = "crm"\n', make_bundle)
    data = parse_task(task, 0, HarborConfig())
    (mount,) = data.tools
    assert mount.interface == "search_execute" and mount.version == 2
    assert mount.seed == str(task / "tools" / "seed" / "crm.json")
    assert data.mcp_servers == [
        {
            "name": "crm",
            "transport": "streamable-http",
            "url": f"http://127.0.0.1:{BASE_PORT}/mcp",
        }
    ]
    assert data.collect[0].command == mount.dump_command()
    assert data.collect[0].service == "main"
    assert data.collect[1].command == "true"
    assert [(a.source, a.service) for a in data.artifacts][-1] == (
        mount.dump_file,
        "main",
    )
    grader = verifier_box_data(data)
    assert grader.tools == [] and grader.mcp_servers == []
    assert HarborTask(data).deployment_spec() is None


def test_a_sidecar_bundle_gets_a_service_of_its_own(tmp_path, make_bundle):
    task = make_task(
        tmp_path, '[[tools]]\nbundle = "crm"\nplacement = "sidecar"\n', make_bundle
    )
    data = parse_task(task, 0, HarborConfig())
    (mount,) = data.tools
    assert data.mcp_servers[-1]["url"] == f"http://crm:{BASE_PORT}/mcp"
    assert data.collect[0].service == "crm"
    assert [(a.source, a.service) for a in data.artifacts][-1] == (
        mount.dump_file,
        "crm",
    )
    spec = HarborTask(data).deployment_spec()
    assert spec.file is None and set(spec.services) == {"crm"}


def test_a_sidecar_may_not_take_an_authored_service_name(tmp_path, make_bundle):
    task = make_task(
        tmp_path, '[[tools]]\nbundle = "crm"\nplacement = "sidecar"\n', make_bundle
    )
    (task / "environment").mkdir()
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: x\n  crm:\n    image: y\n"
    )
    with pytest.raises(ValueError):
        parse_task(task, 0, HarborConfig())


@pytest.mark.parametrize(
    "tools",
    [
        '[[tools]]\nbundle = "nope"\n',
        '[[tools]]\nbundle = "crm"\nseed = "missing.json"\n',
        '[[tools]]\nbundle = "crm"\nseed = "../../x.json"\n',
        '[[tools]]\nbundle = "crm"\nport = 1\n',
        '[[tools]]\nbundle = "crm"\n[[tools]]\nbundle = "crm"\n',
    ],
    ids=["no-bundle", "no-seed", "escape", "unknown-key", "twice"],
)
def test_bad_declarations_fail_the_parse(tmp_path, tools, make_bundle):
    task = make_task(tmp_path, tools, make_bundle)
    with pytest.raises(ValueError):
        parse_mounts(task, tomllib.loads(tools)["tools"])


def test_the_bundle_must_carry_its_declared_name(tmp_path, make_bundle):
    task = make_task(tmp_path, "", make_bundle)
    (task / "tools" / "crm" / "bundle.toml").write_text('name = "other"\nversion = 1\n')
    with pytest.raises(ValueError):
        parse_mounts(task, [{"bundle": "crm"}])


def test_the_digest_follows_content_not_caches(tmp_path, make_bundle):
    bundle = make_bundle(tmp_path, "crm")
    before = bundle_digest(bundle)
    (bundle / "__pycache__").mkdir()
    (bundle / "__pycache__" / "tools.cpython-312.pyc").write_bytes(b"x")
    assert bundle_digest(bundle) == before
    (bundle / "tools.py").write_text((bundle / "tools.py").read_text() + "\n# edit\n")
    assert bundle_digest(bundle) != before
