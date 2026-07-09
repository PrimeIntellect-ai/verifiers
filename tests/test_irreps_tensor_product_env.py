import importlib.util
from pathlib import Path


def _load_module():
    env_path = (
        Path(__file__).parent.parent
        / "environments"
        / "irreps_tensor_product"
        / "irreps_tensor_product.py"
    )
    spec = importlib.util.spec_from_file_location("irreps_tensor_product", env_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decompose_examples():
    module = _load_module()

    assert module._format_decomposition(module._decompose((2, "e"), (3, "o"))) == (
        "1o + 2o + 3o + 4o + 5o"
    )
    assert module._format_decomposition(module._decompose((1, "o"), (1, "o"))) == (
        "0e + 1e + 2e"
    )
    assert module._format_decomposition(module._decompose((0, "e"), (4, "o"))) == "4o"


def test_normalize_decomposition_allows_whitespace_only():
    module = _load_module()

    assert module._normalize_decomposition(" 1o+2o +  3o ") == "1o + 2o + 3o"
    assert module._normalize_decomposition("1o + 2x") is None
    assert module._normalize_decomposition("1o + ") is None


def test_load_environment_builds_standalone_datasets():
    module = _load_module()

    env = module.load_environment(
        num_train_examples=2,
        num_eval_examples=3,
        seed=17,
        min_l=1,
        max_l=2,
        include_even=True,
        include_odd=False,
    )

    assert len(env.dataset) == 2
    assert len(env.eval_dataset) == 3
    row = env.dataset[0]
    assert set(row).issuperset({"question", "answer", "info"})
    assert row["info"]["source_dataset"] == "irreps_tensor_product"
    assert all(term.endswith("e") for term in row["info"]["terms"])


def test_load_environment_is_deterministic():
    module = _load_module()

    first = module.load_environment(num_train_examples=4, num_eval_examples=2, seed=9)
    second = module.load_environment(num_train_examples=4, num_eval_examples=2, seed=9)

    assert first.dataset[:] == second.dataset[:]
    assert first.eval_dataset[:] == second.eval_dataset[:]


def test_exact_rubric_scores_canonical_and_rejects_wrong_parity():
    module = _load_module()
    env = module.load_environment(num_train_examples=1, num_eval_examples=1, seed=3)
    row = env.eval_dataset[0]
    gold = row["answer"]
    wrong = gold.replace("e", "#").replace("o", "e").replace("#", "o")
    reward_func = env.rubric._get_reward_funcs()[0]

    gold_score = reward_func(
        completion=[{"role": "assistant", "content": f"<answer>{gold}</answer>"}],
        answer=gold,
    )
    whitespace_score = reward_func(
        completion=[
            {
                "role": "assistant",
                "content": f"<answer>{gold.replace(' + ', '+')}</answer>",
            }
        ],
        answer=gold,
    )
    wrong_score = reward_func(
        completion=[{"role": "assistant", "content": f"<answer>{wrong}</answer>"}],
        answer=gold,
    )

    assert gold_score == 1.0
    assert whitespace_score == 1.0
    assert wrong_score == 0.0


def test_invalid_config_raises_value_error():
    module = _load_module()

    try:
        module.load_environment(max_l=-1)
    except ValueError as exc:
        assert "max_l" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    try:
        module.load_environment(include_even=False, include_odd=False)
    except ValueError as exc:
        assert "parity" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
