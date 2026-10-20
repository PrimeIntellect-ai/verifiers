"""Where an isolated verifier's box runs: its own image's WORKDIR, or the solver's."""

from types import SimpleNamespace

from verifiers.v1.configs.scoring import ScoringConfig
from verifiers.v1.grading import verifier_runtime
from verifiers.v1.runtimes import PrimeConfig
from verifiers.v1.tasksets.harbor.taskset import HarborArtifact, HarborData, HarborTask

SOLVER = SimpleNamespace(config=PrimeConfig(image="solver:1", workdir="/app"))
POLICY = ScoringConfig(mode="isolated")


def grader(**data) -> HarborTask:
    return HarborTask(HarborData(idx=0, prompt="p", **data))


def test_a_verifier_on_its_own_image_runs_in_that_images_workdir():
    config = verifier_runtime(grader(image="verifier:1"), SOLVER, PrimeConfig(), POLICY)
    # None: the box takes the image's WORKDIR, as `docker run` of it would.
    assert (config.image, config.workdir) == ("verifier:1", None)
    declared = grader(image="verifier:1", workdir="/srv")
    assert verifier_runtime(declared, SOLVER, PrimeConfig(), POLICY).workdir == "/srv"


def test_a_verifier_on_the_solvers_image_or_restoring_relative_files_keeps_its_workdir():
    config = verifier_runtime(grader(), SOLVER, PrimeConfig(), POLICY)
    assert (config.image, config.workdir) == ("solver:1", "/app")
    relative = grader(image="verifier:1", artifacts=[HarborArtifact(source="out")])
    assert verifier_runtime(relative, SOLVER, PrimeConfig(), POLICY).workdir == "/app"
