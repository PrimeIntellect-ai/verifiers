from verifiers.utils.env_utils import env_module_name


def test_env_module_name_normalizes_hub_refs():
    assert env_module_name("primeintellect/reverse-text") == "reverse_text"
    assert env_module_name("primeintellect/reverse-text@v1") == "reverse_text"
