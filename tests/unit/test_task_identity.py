"""Task identity across verifiers' network-policy refactor (#2902)."""

import verifiers.v1 as vf
from verifiers.v1.durable import task_parts
from verifiers.v1.task import identity_data, task_key


def test_a_network_policy_keeps_the_identity_it_had_as_network_allow_and_block():
    # Before #2902 a no-network task dumped network_allow=[] and network_block=[]; it is
    # now network={"allow": [], "block": ["*"]}. Its key and call fingerprint must not
    # change, or every finished call runs again and every job id changes.
    new = {
        "prompt": "p",
        "network": {"allow": [], "block": ["*"]},
        "scoring": {"mode": "shared", "network": None},
    }
    old = {
        "prompt": "p",
        "network_allow": [],
        "network_block": [],
        "scoring": {"mode": "shared"},
    }
    assert identity_data(new) == old
    hosts = {"prompt": "p", "network": {"allow": ["pypi.org"], "block": []}}
    assert identity_data(hosts) == {
        "prompt": "p",
        "network_allow": ["pypi.org"],
        "network_block": [],
    }
    verifier = {"verifier": {"image": "v", "network": {"allow": [], "block": ["*"]}}}
    assert identity_data(verifier) == {"verifier": {"image": "v", "network_allow": []}}


def test_a_declared_grading_network_is_part_of_identity():
    # Grading under a different network can grade differently: that changes identity.
    declared = {
        "prompt": "p",
        "scoring": {"mode": "shared", "network": {"allow": [], "block": ["*"]}},
    }
    assert identity_data(declared)["scoring"]["network"] == {
        "allow": [],
        "block": ["*"],
    }


def test_task_key_and_parts_hash_the_identity_shape():
    task = vf.Task(vf.TaskData(prompt="p", network=vf.NetworkPolicyConfig(allow=[])))
    dumped = task.data.model_dump(mode="json", exclude_none=True)
    assert task.hash == task_key(identity_data(dumped))
    assert (
        "network" not in identity_data(dumped)
        and identity_data(dumped)["network_allow"] == []
    )
    assert task_parts(task)["data"]


def test_task_data_nested_in_task_data_keeps_its_identity_too():
    """A QC task records the task it audits: that record reads in its old shape as well."""
    record = {
        "name": "t",
        "network": {"allow": [], "block": ["*"]},
        "verifier": {"network": {"allow": [], "block": ["*"]}},
    }
    old_record = {
        "name": "t",
        "network_allow": [],
        "network_block": [],
        "verifier": {"network_allow": []},
    }
    data = {"name": "qc", "network": {"allow": ["*"], "block": []}, "record": record}
    assert identity_data(data) == {
        "name": "qc",
        "network_allow": ["*"],
        "network_block": [],
        "record": old_record,
    }
