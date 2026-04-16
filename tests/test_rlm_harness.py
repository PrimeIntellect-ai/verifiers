from verifiers.envs.experimental.composable.harnesses.rlm import (
    DEFAULT_RLM_BRANCH,
    DEFAULT_RLM_REPO_URL,
    build_install_script,
    rlm_harness,
)


def test_build_install_script_defaults():
    script = build_install_script()
    assert f"{DEFAULT_RLM_BRANCH}/install.sh" in script
    assert f"RLM_REPO_URL={DEFAULT_RLM_REPO_URL}" in script
    assert "RLM_REPO_BRANCH=main" in script


def test_build_install_script_custom_branch_and_repo():
    script = build_install_script("github.com/example/rlm.git", "feature/x")
    assert "raw.githubusercontent.com/example/rlm/feature/x/install.sh" in script
    assert "RLM_REPO_URL=github.com/example/rlm.git" in script
    assert "RLM_REPO_BRANCH=feature/x" in script


def test_rlm_harness_threads_branch_to_install_script():
    harness = rlm_harness(
        rlm_repo_url="github.com/example/rlm.git",
        rlm_branch="feature/x",
    )
    assert harness.install_script is not None
    assert (
        "raw.githubusercontent.com/example/rlm/feature/x/install.sh"
        in harness.install_script
    )
    assert "RLM_REPO_BRANCH=feature/x" in harness.install_script
