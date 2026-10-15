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
    state = json.load(open("/logs/artifacts/tools/crm.json"))
    names = [c["name"] for c in state["crm"]["contacts"]]
    open("/logs/verifier/reward.txt", "w").write("1" if "Grace" in names else "0")
    EOF
    """
)


@pytest.fixture
def task(tmp_path) -> HarborTask:
    root = tmp_path / "crm-task"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test.sh").write_text(TEST_SH)
    (root / "instruction.md").write_text("Add Grace to the CRM.\n")
    (root / "task.toml").write_text(
        f'version = "1.0"\n[environment]\ndocker_image = "{IMAGE}"\n'
        '[agent]\nuser = "agent"\n[verifier]\nenvironment_mode = "separate"\n'
        '[[tools]]\nbundle = "crm"\n'
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


async def graded(task: HarborTask, act) -> vf.Trace:
    trace = vf.Trace(
        task=vf.TraceTask.of(task),
        agent=vf.AgentInfo(config=vf.AgentConfig(model="unused")),
    )
    backend = make_backend(vf.DockerConfig(image=IMAGE, workdir="/app"))
    deployment = backend.deploy(spec=task.deployment_spec())
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
