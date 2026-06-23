"""Run the merged F2P+P2P pytest ids and print 1.0 iff every expected id passed.

Run inside the task's repo (cwd) by the testbed python (which has pytest + the project
installed) — NOT a uv script, since the tests need the project's own environment. argv[1]
is the path to a JSON file of pytest node ids (a file, not inline, so a large id list can't
overflow the sandbox exec command line). We write JUnit XML and match each expected id against
it (a pytest node id and the JUnit classname/name don't line up, so we try a few forms),
printing the score on the last line. Any failure prints 0.0.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET

import pytest

XML = "/tmp/scaleswe_results.xml"


def _normalize(value: str) -> str:
    parts = value.strip().split("::")
    if parts and parts[0].endswith(".py"):
        parts[0] = parts[0][:-3]
    return ".".join(parts).replace("/", ".").strip(".")


def _all_passed(xml_content: str, expected: list[str]) -> bool:
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return False
    exact = set(expected)
    norm = {_normalize(t): t for t in expected}
    fp = {re.sub(r"\s+", "", _normalize(t)): t for t in expected}
    matched: dict[str, str] = {}
    found: set[str] = set()
    for tc in root.iter("testcase"):
        if tc.find("skipped") is not None:
            continue
        name, classname = tc.get("name", ""), tc.get("classname", "")
        file_attr = tc.get("file", "")
        status = (
            "failed"
            if tc.find("failure") is not None or tc.find("error") is not None
            else "passed"
        )
        for candidate in (
            f"{file_attr}::{name}" if file_attr else "",
            _normalize(f"{classname}.{name}"),
            re.sub(r"\s+", "", _normalize(f"{classname}.{name}")),
            f"{classname.replace('.', '/')}.py::{name}",
        ):
            original = (
                candidate
                if candidate in exact
                else norm.get(candidate) or fp.get(candidate)
            )
            if original:
                matched[original] = status
                found.add(original)
                break
    return (
        bool(found)
        and all(status == "passed" for status in matched.values())
        and not [t for t in expected if t not in found]
    )


def main() -> None:
    expected = json.load(open(sys.argv[1]))
    if not expected:
        print(0.0)
        return
    pytest.main(
        ["-vv", f"--junitxml={XML}", "-o", "addopts=", "--rootdir=.", *expected]
    )
    try:
        xml_content = open(XML).read()
    except OSError:
        print(0.0)
        return
    print(1.0 if _all_passed(xml_content, expected) else 0.0)


if __name__ == "__main__":
    main()
