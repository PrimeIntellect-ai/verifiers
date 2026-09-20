"""Harbor source tasks backed by a persistent, reviewed Worlds repository."""

import asyncio
import fnmatch
import io
import tarfile
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field

import verifiers.v1 as vf
from harbor_swarm.repository import archive
from verifiers.v1.envs.swarm import SwarmTask, WorldConnection
from verifiers.v1.runtimes import RuntimeConfig, provision_runtime
from verifiers.v1.tasksets.harbor.taskset import (
    HarborConfig,
    HarborData,
    HarborTask,
    HarborTaskConfig,
    parse_task,
)


class HarborSwarmData(HarborData):
    workspace: str
    editable: list[str]
    public_check: str


class HarborSwarmTask(SwarmTask, vf.Task[HarborSwarmData]):
    review_repository = "solution"
    seed_files: dict[str, str]
    seed_oid: str
    snapshot: dict
    seed_runtime: RuntimeConfig

    def harbor(self) -> HarborTask:
        return HarborTask(HarborData.model_validate(self.data.model_dump()))

    def runtime_env(self) -> dict[str, str]:
        return {**self.harbor().runtime_env(), **super().runtime_env()}

    async def prepare_world(self, world: WorldConnection) -> None:
        async with (
            asyncio.timeout(600),
            provision_runtime(
                self.seed_runtime, env=self.harbor().runtime_env()
            ) as runtime,
        ):
            await runtime.prepare_setup()
            await self.harbor().setup(runtime)
            collected = await vf.collect(runtime, self.data.artifacts)
            payload = collected[self.data.workspace]
            if payload is None:
                raise ValueError("Missing seed workspace")
            self.seed_files = {}
            root = self.data.workspace.lstrip("/") + "/"
            with tarfile.open(fileobj=io.BytesIO(payload)) as source:
                for entry in source:
                    if entry.isdir():
                        continue
                    if not entry.isfile() or not entry.name.startswith(root):
                        raise ValueError("Seed workspace must contain regular files")
                    stream = source.extractfile(entry)
                    assert stream is not None
                    self.seed_files[entry.name[len(root) :]] = stream.read().decode()
        repo = await world.mutate("create_repository", name="solution")
        initial = repo["branches"][0]["oid"]
        edits = {"README.md": None, **self.seed_files}
        change = await world.mutate(
            "seed_repository",
            repository="solution",
            branch="main",
            base_oid=initial,
            expected_head=initial,
            files=edits,
            message="Seed task workspace",
        )
        self.seed_oid = change["commit_oid"]
        await world.mutate(
            "create_issue",
            repository="solution",
            title=self.data.name or "Task",
            body=self.data.prompt_text,
        )
        await world.mutate("join", conversation="general")
        await world.mutate(
            "send",
            conversation="general",
            body="Task instructions are in solution issue #1. Coordinate here; submit through reviewed PRs.",
        )

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await self.harbor().setup(runtime)
        await super().setup(trace, runtime)
        await vf.restore(runtime, archive(self.data.workspace, self.seed_files))

    def participant(self, world, index, accounts):
        participant = super().participant(world, index, accounts)
        # Add only the world host to the benchmark's existing network allowlist.
        if "*" not in participant.data.network_allow:
            participant.data = participant.data.model_copy(
                update={
                    "network_allow": [
                        *participant.data.network_allow,
                        urlsplit(world.url).hostname,
                    ],
                }
            )
        return participant

    def participant_prompt(self, index, accounts):
        return (
            self.data.prompt_text
            + f"""

You share repository solution with your team. Your local workspace is {self.data.workspace}.
Only files matching {self.data.editable} may change. Use public checks: {self.data.public_check}
Coordinate implementation through general and the issue/PR tools. Solvers implement and test;
the coordinator integrates reviewed PRs and organizes the final unanimous submission.
Follow the task's rules for reference implementations. Repository tools run on the world server:
- repository {{"repository":"solution"}} lists branches and exact commit IDs.
- read_tree {{"repository":"solution","commit_oid":"OID"}} returns files at an exact commit.
- commit_files {{"repository":"solution","branch":"my-branch","base_oid":"OID", "message":"Summary","files":{{"path":"UTF-8 content"}}}} --mutate creates a branch.
  To update it, also pass expected_head equal to its current OID and base_oid. Null file content deletes a file.
  Main is protected. Read files and apply changes locally using Python; no Git client is necessary.
- create_pr {{"repository":"solution","head":"my-branch","title":"Summary"}} --mutate
- pr {{"repository":"solution","number":1}} shows exact head/base IDs and diff.
- review_pr {{"repository":"solution","number":1,"expected_head":"OID","expected_base":"OID","verdict":"approve"}} --mutate (another author must review).
- merge_pr {{"repository":"solution","number":1,"expected_head":"OID","expected_base":"OID"}} --mutate
- join {{"conversation":"general"}} --mutate; send {{"conversation":"general","body":"text"}} --mutate
- read {{"conversation":"general"}}; issue {{"repository":"solution","number":1}}
Work from the latest main, review each other's changes, and test the assembled main before proposing it.
The coordinator must not merely collect votes: independently inspect and run the public check.
Final hidden grading runs once in a fresh sandbox after the team stops. Hidden scores are not available during solving.
"""
        )

    def validate_files(self, files):
        changed = {
            name
            for name in self.seed_files.keys() | files.keys()
            if self.seed_files.get(name) != files.get(name)
        }
        invalid = [
            name
            for name in changed
            if not any(
                fnmatch.fnmatchcase(name, pattern) for pattern in self.data.editable
            )
        ]
        if invalid:
            raise ValueError(
                "Changes outside task submission paths: " + ", ".join(sorted(invalid))
            )

    async def check_revision(self, world, commit_oid):
        snapshot = await world.call(
            "read_tree", repository="solution", commit_oid=commit_oid
        )
        try:
            self.validate_files(snapshot["files"])
        except ValueError as error:
            return {"passed": False, "commit_oid": commit_oid, "detail": str(error)}
        async with provision_runtime(
            self.seed_runtime, env=self.harbor().runtime_env()
        ) as runtime:
            await runtime.prepare_setup()
            await self.harbor().setup(runtime)
            await vf.restore(runtime, archive(self.data.workspace, snapshot["files"]))
            await runtime.prepare_execution([])
            result = await runtime.run(["sh", "-c", self.data.public_check], {})
            return {
                "passed": result.exit_code == 0,
                "commit_oid": commit_oid,
                "exit_code": result.exit_code,
                "output": (result.stdout + result.stderr)[-8000:],
            }

    async def capture(self, world):
        review = await world.call("review_round", round_id=self.review_round)
        oid = review["accepted_oid"]
        if not oid:
            repo = await world.call("repository", repository="solution")
            oid = next(
                branch["oid"] for branch in repo["branches"] if branch["name"] == "main"
            )
        self.snapshot = await world.call(
            "read_tree", repository="solution", commit_oid=oid
        )
        return {
            "repository": "solution",
            "commit_oid": oid,
            "accepted": review["status"] == "accepted",
        }

    async def evaluate(self, submission):
        raise RuntimeError(
            "HarborSwarmEnv grades repository artifacts through Harbor in a fresh runtime"
        )


class HarborSwarmConfig(vf.TasksetConfig):
    task: HarborTaskConfig = HarborTaskConfig()
    task_dir: Path | None = None
    workspace: str = "/app/zig-git"
    editable: list[str] = Field(default_factory=lambda: ["src/*.zig"])
    public_check: str = "zig build"


class HarborSwarmTaskset(vf.Taskset[HarborSwarmTask, HarborSwarmConfig]):
    def load(self):
        if self.config.task_dir is None:
            raise ValueError(
                "Set --env.taskset.task-dir to a local Harbor task with a separate verifier"
            )
        data = parse_task(
            self.config.task_dir.resolve(),
            0,
            HarborConfig(require_image=True, ignore_timeouts=False),
        )
        if data.verifier is None:
            raise ValueError(
                "Declare a separate Harbor verifier to grade the shared repository in a fresh runtime."
            )
        if data.collect or data.mcp_servers:
            raise ValueError(
                "Repository adaptation currently supports file artifacts without collect hooks or task MCP servers"
            )
        if (
            len(data.artifacts) != 1
            or data.artifacts[0].source != self.config.workspace
        ):
            raise ValueError(
                "Declare exactly one Harbor artifact at the configured workspace"
            )
        row = HarborSwarmData(
            **data.model_dump(),
            workspace=self.config.workspace,
            editable=self.config.editable,
            public_check=self.config.public_check,
        )
        return [HarborSwarmTask(row, self.config.task)]
