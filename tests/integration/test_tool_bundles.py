"""A Harbor task with a `[[tools]]` bundle on local Docker, without a model: setup
installs, seeds and starts the server; a non-root agent changes state only through the
tools; a restored box resumes from the state on disk; the end state is dumped, travels
to a separate grader and is graded there. The untouched seed scores 0."""

import json
import shutil
import textwrap

import pytest

import verifiers.v1 as vf
from verifiers.v1.rollout import resolve_agent_runtime
from verifiers.v1.runtimes import make_backend
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask
from verifiers.v1.tasksets.harbor.taskset import parse_task
from verifiers.v1.utils.decorators import invoke

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="needs Docker")

IMAGE = "python:3.12-slim"

TOOLS = textwrap.dedent(
    '''
    from vf_tools import ToolError, new_id, tool


    @tool(app="crm")
    def crm_add_contact(state, name: str, email: str) -> dict:
        """Add a contact."""
        contact = {"id": new_id(state, "contact"), "name": name, "email": email}
        state["crm"]["contacts"].append(contact)
        return contact


    @tool(app="crm")
    def crm_list_contacts(state) -> list:
        """List contacts."""
        return state["crm"]["contacts"]
    '''
)

TEST_SH = textwrap.dedent(
    """\
    python3 - <<'EOF'
    import json
    state = json.load(open("/logs/tools/crm.json"))
    names = [c["name"] for c in state["crm"]["contacts"]]
    open("/logs/verifier/reward.txt", "w").write("1" if "Grace" in names else "0")
    EOF
    """
)


def make_task(
    tmp_path, *, placement: str = "box", agent_user: bool = True
) -> HarborTask:
    root = tmp_path / f"crm-{placement}"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test.sh").write_text(TEST_SH)
    (root / "instruction.md").write_text("Add Grace to the CRM.\n")
    (root / "task.toml").write_text(
        f'version = "1.0"\n[environment]\ndocker_image = "{IMAGE}"\n'
        + ('[agent]\nuser = "agent"\n' if agent_user else "")
        + '[verifier]\nenvironment_mode = "separate"\n'
        f'[[tools]]\nbundle = "crm"\nplacement = "{placement}"\n'
    )
    bundle = root / "tools" / "crm"
    bundle.mkdir(parents=True)
    (bundle / "bundle.toml").write_text('name = "crm"\nversion = 1\n')
    (bundle / "tools.py").write_text(TOOLS)
    (root / "tools" / "seed").mkdir()
    (root / "tools" / "seed" / "crm.json").write_text(
        json.dumps({"crm": {"contacts": [{"id": "c0", "name": "Ada", "email": "a@x"}]}})
    )
    return HarborTask(parse_task(root, 0, HarborConfig()))


@pytest.fixture
def task(tmp_path) -> HarborTask:
    return make_task(tmp_path)


async def graded(task: HarborTask, act) -> vf.Trace:
    trace = vf.Trace(
        task=vf.TraceTask.of(task),
        agent=vf.AgentInfo(
            config=vf.AgentConfig(
                model="unused", runtime=vf.DockerConfig(image=IMAGE, workdir="/app")
            )
        ),
    )
    backend = make_backend(vf.DockerConfig(image=IMAGE, workdir="/app"))
    deployment = backend.deploy(
        spec=task.deployment_spec(), config=vf.DeploymentConfig(trust_compose=True)
    )
    try:
        await deployment.start()
        runtime = deployment.workspace
        hooks = {"trace": trace, "runtime": runtime, "deployment": deployment}
        await invoke(task.setup, hooks)
        await act(task, runtime)
        await task.finalize(trace, runtime)
        await vf.grade_submission(task, trace, runtime, deployment=deployment)
    finally:
        await deployment.close()
    return trace


async def test_tools_change_state_and_the_grader_reads_the_end_state(task):
    seen = {}

    async def act(task, runtime):
        agent = await resolve_agent_runtime(runtime, "agent")
        added = await agent.run(
            [
                "vf-tools",
                "call",
                "crm_add_contact",
                '{"name": "Grace", "email": "g@x"}',
            ],
            {},
        )
        seen["added"] = added
        seen["read_state"] = await agent.run(
            ["cat", "/var/lib/vf-tools/crm/state.json"], {}
        )
        seen["read_bundle"] = await agent.run(["ls", "/opt/vf-tools/crm"], {})
        # A box restored from a checkpoint has the files but no process.
        await runtime.run(["pkill", "-f", "vf_tools.py serve"], {})
        await invoke(task.restore, {"runtime": runtime})
        listed = await agent.run(["vf-tools", "call", "crm_list_contacts"], {})
        seen["listed"] = json.loads(listed.stdout)

    trace = await graded(task, act)
    assert seen["added"].exit_code == 0, seen["added"].stderr
    assert json.loads(seen["added"].stdout)["id"] == "contact_1"
    assert seen["read_state"].exit_code != 0
    assert seen["read_bundle"].exit_code != 0
    assert [c["name"] for c in seen["listed"]] == ["Ada", "Grace"]
    assert trace.reward == 1, [e.message for e in trace.errors]


async def test_the_untouched_seed_scores_zero(task):
    async def act(task, runtime):
        return None

    trace = await graded(task, act)
    assert trace.reward == 0, [e.message for e in trace.errors]


async def test_a_sidecar_keeps_the_state_out_of_a_root_agents_box(tmp_path):
    """A sidecar-placed bundle runs in its own Compose service: a root agent reaches it
    only through the tools, and its dump travels from that service to the grader."""
    task = make_task(tmp_path, placement="sidecar", agent_user=False)
    seen = {}

    async def act(task, runtime):
        seen["added"] = await runtime.run(
            [
                "vf-tools",
                "call",
                "crm_add_contact",
                '{"name": "Grace", "email": "g@x"}',
            ],
            {},
        )
        seen["state"] = await runtime.run(["ls", "/var/lib/vf-tools/crm"], {})

    trace = await graded(task, act)
    assert seen["added"].exit_code == 0, seen["added"].stderr
    assert seen["state"].exit_code != 0
    assert trace.reward == 1, [e.message for e in trace.errors]


ENV_TOOLS = textwrap.dedent(
    '''
    import os
    import pwd

    from vf_tools import tool


    @tool
    def add_note(state, text: str) -> int:
        """Append a note to the team's notes file."""
        with open("/app/data/notes.txt", "a") as handle:
            return handle.write(text + "\\n")


    @tool
    def whoami(state) -> str:
        return pwd.getpwuid(os.getuid()).pw_name


    @tool
    def tamper(state) -> int:
        with open("/app/elsewhere.txt", "w") as handle:
            return handle.write("x")
    '''
)

ENV_TEST_SH = textwrap.dedent(
    """\
    python3 - <<'EOF'
    text = open("/logs/tools/notes.out").read()
    open("/logs/verifier/reward.txt", "w").write("1" if "hello" in text else "0")
    EOF
    """
)


async def test_an_environment_bundle_runs_confined_as_its_own_user(tmp_path):
    """An environment-kind bundle works on the task's own files: its server runs as an
    unprivileged user of its own, writes only under its declared path, and its declared
    dump command observes the result for the grader."""
    root = tmp_path / "notes-task"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test.sh").write_text(ENV_TEST_SH)
    (root / "instruction.md").write_text("Leave a note saying hello.\n")
    (root / "task.toml").write_text(
        f'version = "1.0"\n[environment]\ndocker_image = "{IMAGE}"\n'
        '[agent]\nuser = "agent"\n[verifier]\nenvironment_mode = "separate"\n'
        '[[tools]]\nbundle = "notes"\n'
    )
    bundle = root / "tools" / "notes"
    bundle.mkdir(parents=True)
    (bundle / "bundle.toml").write_text(
        'name = "notes"\nversion = 1\nowner = "task"\nkind = "environment"\n'
        'endpoints = ["/app/data"]\ndump = "cat /app/data/notes.txt"\n'
    )
    (bundle / "tools.py").write_text(ENV_TOOLS)
    task = HarborTask(parse_task(root, 0, HarborConfig()))
    seen = {}

    async def act(task, runtime):
        agent = await resolve_agent_runtime(runtime, "agent")
        for name, arguments in (
            ("add_note", '{"text": "hello"}'),
            ("whoami", "{}"),
            ("tamper", "{}"),
        ):
            seen[name] = await agent.run(["vf-tools", "call", name, arguments], {})
        seen["code"] = await agent.run(["ls", "/opt/vf-tools/notes"], {})

    trace = await graded(task, act)
    assert seen["add_note"].exit_code == 0, seen["add_note"].stdout
    assert json.loads(seen["whoami"].stdout) == "vf-tools-notes"
    assert seen["tamper"].exit_code != 0
    assert seen["code"].exit_code != 0
    assert trace.reward == 1, [e.message for e in trace.errors]
