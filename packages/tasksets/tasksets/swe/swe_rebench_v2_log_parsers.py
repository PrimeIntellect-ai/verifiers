# ruff: noqa
# Vendored from https://github.com/SWE-rebench/SWE-rebench-V2
#   lib/agent/log_parsers.py @ main (2026-04)
#   lib/agent/swe_constants.py @ main (inlined as TestStatus below)
#
# Upstream is not pip-installable (no pyproject.toml / setup.py), so we
# vendor the parser module verbatim to preserve grading fidelity against
# the nebius/SWE-rebench-V2 dataset. The only edit vs. upstream is the
# replacement of `from lib.agent.swe_constants import TestStatus` with
# an inline enum so this file has no out-of-tree imports. Ruff is
# disabled for this file so the vendor stays byte-for-byte comparable
# to upstream on future syncs.

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from enum import Enum
from typing import Any


class TestStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
ANSI_COLOR_DELIM_RE = re.compile(r"\x1B\[[0-9;]*m")
OCAML_STATUS_PREFIX_RE = re.compile(
    r"^(" + "|".join(re.escape(name) for name in TestStatus.__members__) + r")\s+(.*)$"
)
OCAML_STATUS_PRECEDENCE = {
    TestStatus.PASSED.value: 0,
    TestStatus.SKIPPED.value: 1,
    TestStatus.FAILED.value: 2,
    TestStatus.ERROR.value: 3,
}


def ansi_escape(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return ANSI_ESCAPE_RE.sub("", text)


def parse_log_pytest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if any(line.startswith(x.value) for x in TestStatus):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_options(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework with options

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    test_status_map = {}
    for line in log.split("\n"):
        if any(line.startswith(x.value) for x in TestStatus):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            has_option = option_pattern.search(test_case[1])
            if has_option:
                main, option = has_option.groups()
                if (
                    option.startswith("/")
                    and not option.startswith("//")
                    and "*" not in option
                ):
                    option = "/" + option.split("/")[-1]
                test_name = f"{main}[{option}]"
            else:
                test_name = test_case[1]
            test_status_map[test_name] = test_case[0]
    return test_status_map


def parse_log_django(log: str) -> dict[str, str]:  # noqa: PLR0912
    """
    Parser for test logs generated with Django tester framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    lines = log.split("\n")

    prev_test = None
    for line in lines:
        line = line.strip()

        # This isn't ideal but the test output spans multiple lines
        if "--version is equivalent to version" in line:
            test_status_map["--version is equivalent to version"] = (
                TestStatus.PASSED.value
            )

        # Log it in case of error
        if " ... " in line:
            prev_test = line.split(" ... ")[0]

        pass_suffixes = (" ... ok", " ... OK", " ...  OK")
        for suffix in pass_suffixes:
            if line.endswith(suffix):
                # TODO: Temporary, exclusive fix for django__django-7188
                # The proper fix should involve somehow getting the test results to
                # print on a separate line, rather than the same line
                if line.strip().startswith(
                    "Applying sites.0002_alter_domain_unique...test_no_migrations"
                ):
                    line = line.split("...", 1)[-1].strip()
                test = line.rsplit(suffix, 1)[0]
                test_status_map[test] = TestStatus.PASSED.value
                break
        if " ... skipped" in line:
            test = line.split(" ... skipped")[0]
            test_status_map[test] = TestStatus.SKIPPED.value
        if line.endswith(" ... FAIL"):
            test = line.split(" ... FAIL")[0]
            test_status_map[test] = TestStatus.FAILED.value
        if line.startswith("FAIL:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.FAILED.value
        if line.endswith(" ... ERROR"):
            test = line.split(" ... ERROR")[0]
            test_status_map[test] = TestStatus.ERROR.value
        if line.startswith("ERROR:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.ERROR.value

        if line.lstrip().startswith("ok") and prev_test is not None:
            # It means the test passed, but there's some additional output (including new lines)
            # between "..." and "ok" message
            test = prev_test
            test_status_map[test] = TestStatus.PASSED.value

    # TODO: This is very brittle, we should do better
    # There's a bug in the django logger, such that sometimes a test output near the end gets
    # interrupted by a particular long multiline print statement.
    # We have observed this in one of 3 forms:
    # - "{test_name} ... Testing against Django installed in {*} silenced.\nok" # noqa: ERA001
    # - "{test_name} ... Internal Server Error: \/(.*)\/\nok" # noqa: ERA001
    # - "{test_name} ... System check identified no issues (0 silenced).\nok" # noqa: ERA001
    patterns = [
        r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
        r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
        r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            test_name = match.group(1)
            test_status_map[test_name] = TestStatus.PASSED.value
    return test_status_map


def parse_log_pytest_v2(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = "".join(chr(char) for char in range(1, 32))
    translator = str.maketrans("", "", escapes)
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        line = line.translate(translator)
        if any(line.startswith(x.value) for x in TestStatus):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            test_status_map[test_case[1]] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any(line.endswith(x.value) for x in TestStatus):
            test_case = line.split()
            test_status_map[test_case[0]] = test_case[1]
    return test_status_map


def parse_log_seaborn(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with seaborn testing framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if line.startswith(TestStatus.FAILED.value):
            test_case = line.split()[1]
            test_status_map[test_case] = TestStatus.FAILED.value
        elif f" {TestStatus.PASSED.value} " in line:
            parts = line.split()
            if parts[1] == TestStatus.PASSED.value:
                test_case = parts[0]
                test_status_map[test_case] = TestStatus.PASSED.value
        elif line.startswith(TestStatus.PASSED.value):
            parts = line.split()
            test_case = parts[1]
            test_status_map[test_case] = TestStatus.PASSED.value
    return test_status_map


def parse_log_sympy(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    pattern = r"(_*) (.*)\.py:(.*) (_*)"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f"{match[1]}.py:{match[2]}"
        test_status_map[test_case] = TestStatus.FAILED.value
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(("[FAIL]", "[OK]")):
                line = line[: line.rfind("[")]
                line = line.strip()
            if line.endswith(" E"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.ERROR.value
            if line.endswith(" F"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.FAILED.value
            if line.endswith(" ok"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.PASSED.value
    return test_status_map


def parse_log_matplotlib(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        line = line.replace("MouseButton.LEFT", "1")
        line = line.replace("MouseButton.RIGHT", "3")
        if any(line.startswith(x.value) for x in TestStatus):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_nebo(log: str) -> dict[str, str]:
    """
    Enhanced parser that handles both gw-prefixed and non-prefixed test lines
    """
    test_status_map = {}
    escapes = "".join(chr(char) for char in range(1, 32))
    translator = str.maketrans("", "", escapes)

    # Pattern 1: [gwX] [Y%] STATUS TEST_NAME in TIME
    pattern_gw = re.compile(r"^\[gw\d+\]\s+\[[^\]]*\]\s+(\w+)\s+(.*)$")
    # Pattern 2: TEST_NAME STATUS [Y%] in TIME
    pattern_standard = re.compile(
        r"^([^\s].*?)\s+("
        + "|".join(re.escape(x.value) for x in TestStatus)
        + r")\s+\[[^\]]*\]\s+in\s+\d+(?:\.\d+)?s$"
    )
    # Pattern 3: STATUS TEST_NAME
    pattern_status_first = re.compile(
        r"^(" + "|".join(re.escape(x.value) for x in TestStatus) + r")\s+(.*)$"
    )

    for line in log.splitlines():
        # Clean ANSI escape codes and non-printable characters
        line = re.sub(r"\[(\d+)m", "", line)
        line = line.translate(translator).strip()

        # Handle gw-prefixed format: [gwX] [Y%] STATUS TEST_NAME
        match_gw = pattern_gw.match(line)
        if match_gw:
            status_str = match_gw.group(1)
            if status_str in TestStatus.__members__:
                test_name = match_gw.group(2).split(" in ")[0].strip()
                test_status_map[test_name] = status_str
                continue

        # Handle standard format: TEST_NAME STATUS [Y%] in TIME
        match_standard = pattern_standard.match(line)
        if match_standard:
            test_name = match_standard.group(1).strip()
            status_str = match_standard.group(2).strip()
            test_status_map[test_name] = status_str
            continue

        # Handle status-first format: STATUS TEST_NAME
        match_status_first = pattern_status_first.match(line)
        if match_status_first:
            status_str = match_status_first.group(1).strip()
            test_name = match_status_first.group(2).split(" in ")[0].strip()
            test_status_map[test_name] = status_str
            continue

    return test_status_map


def parse_test_report(xml_content: str) -> dict[str, str]:
    """Parse a single XML report and return test status dictionary."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return {}

    results = {}
    for testcase in root.findall(".//testcase"):
        classname = testcase.get("classname")
        name = testcase.get("name")
        full_name = f"{classname}::{name}"

        if testcase.find("failure") is not None:
            status = TestStatus.FAILED.value
        elif testcase.find("skipped") is not None:
            status = TestStatus.SKIPPED.value
        elif testcase.find("error") is not None:
            status = TestStatus.ERROR.value
        else:
            status = TestStatus.PASSED.value

        results[full_name] = status

    return results


def parse_combined_test_reports(content: str) -> dict[str, str]:
    """Parse all XML reports in content and return combined test status dictionary."""
    combined_results = {}
    start_marker = "<?xml"
    end_marker = "</testsuites>"
    pos = 0

    while pos < len(content):
        # Find next XML start marker
        xml_start = content.find(start_marker, pos)
        if xml_start == -1:
            break  # No more XML documents

        # Find corresponding XML end marker
        xml_end = content.find(end_marker, xml_start)
        if xml_end == -1:
            # Skip incomplete XML
            pos = xml_start + len(start_marker)
            continue

        # Adjust end position to include full end marker
        xml_end += len(end_marker)
        xml_doc = content[xml_start:xml_end]

        try:
            results = parse_test_report(xml_doc)
            combined_results.update(results)  # Later reports overwrite earlier ones
        except ET.ParseError:
            # Skip invalid XML
            pass

        # Move to next position after current XML document
        pos = xml_end

    return combined_results


def parse_log_gotest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with 'go test'

    Args:
        log (str): log content
        test_spec (TestSpec): test spec (unused)
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    # Pattern to match test result lines
    pattern = r"^--- (PASS|FAIL|SKIP): (.+) \((.+)\)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name, _duration = match.groups()
            if status == "PASS":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == "FAIL":
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status == "SKIP":
                test_status_map[test_name] = TestStatus.SKIPPED.value

    return test_status_map


def parse_log_elixir(log: str) -> dict[str, str]:
    """Parse ExUnit output and return {full_test_name: status}.

    Rules:
      * Lines like: "* test <name> [L#42]" or with timing "(12.3ms)" -> PASSED (tentative)
      * Lines like: "* test <name> (skipped) [L#42]" -> SKIPPED
      * Failure headers: "1) test <name> (<Module>)" -> FAILED (overrides prior PASS)
    """
    results: dict[str, str] = {}

    # Regexes
    skipped_re = re.compile(r"^\*\s+test\s+(.*?)\s+\(skipped\)\s+\[L#\d+\]$")
    passed_timed_re = re.compile(
        r"^\*\s+test\s+(.*?)\s+\([0-9]+(?:\.[0-9]+)?ms\)\s+\[L#\d+\]$"
    )
    passed_basic_re = re.compile(r"^\*\s+test\s+(.*?)\s+\[L#\d+\]$")
    failure_header_re = re.compile(r"^\d+\)\s+test\s+(.*?)\s+\([^)]+\)$")

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := skipped_re.match(line):
            results[m.group(1)] = TestStatus.SKIPPED.value
            continue
        if m := failure_header_re.match(line):
            results[m.group(1)] = TestStatus.FAILED.value
            continue
        if m := passed_timed_re.match(line):
            results.setdefault(m.group(1), TestStatus.PASSED.value)
            continue
        if m := passed_basic_re.match(line):
            results.setdefault(m.group(1), TestStatus.PASSED.value)
            continue
    return results


def parse_log_ruby_v1(log: str) -> dict[str, str]:
    """Parse Ruby MiniTest (rake test) output and return {full_test_name: status}.

    Enhancements:
      * Captures suite header lines like: "UnitTestStorageXMLCollections" (no leading spaces, CamelCase/word chars)
      * Each test is keyed as "<SuiteName>::<test_method_name>" when a current suite is known,
        otherwise just the test_method_name.

    Status line examples:
        "  test_put_url_path_is_properly_escaped                           PASS (0.02s)"
        "  test_collection_get_arguments                                   SKIP (0.01s)"
        "  test_something                                                 FAIL (0.12s)"
        "  test_other                                                     ERROR (0.00s)"

    Normalized statuses: PASSED, FAILED, ERROR, SKIPPED.
    If a test appears multiple times, a worse status (FAILED/ERROR) overrides earlier PASS/SKIP.
    """
    results: dict[str, str] = {}

    # Regex for status lines
    status_re = re.compile(
        r"^\s*(test[^A-Z\n]+?)\s+(PASS|PASS(?:ED)?|FAIL|FAILURE|ERROR|SKIP|SKIPPED)\b(?:\s*\([0-9.]+s\))?$"
    )
    # Regex for suite header: single word (letters, digits, underscore) starting with uppercase, no spaces
    suite_re = re.compile(r"^[A-Z][A-Za-z0-9_]*(?:[A-Z][A-Za-z0-9_]*)*$")

    severity_rank = {"PASSED": 0, "SKIPPED": 0, "FAILED": 1, "ERROR": 2}

    def norm(status: str) -> str:
        s = status.upper()
        if s in ("PASS", "PASSED"):
            return "PASSED"
        if s in ("FAIL", "FAILURE"):
            return "FAILED"
        if s in ("SKIP", "SKIPPED"):
            return "SKIPPED"
        return s  # ERROR

    current_suite: str | None = None

    for raw in log.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("Finished in "):
            continue
        # Detect suite header (must not start with two spaces like test lines)
        if not line.startswith(" ") and suite_re.match(line):
            current_suite = line
            continue
        m = status_re.match(line)
        if not m:
            continue
        test_name, status_token = m.group(1).strip(), m.group(2)
        full_name = f"{current_suite}::{test_name}" if current_suite else test_name
        status_norm = norm(status_token)
        prev = results.get(full_name)
        if prev is None or severity_rank.get(status_norm, 0) > severity_rank.get(
            prev, 0
        ):
            results[full_name] = status_norm
    return results


def parse_log_redis(log: str) -> dict[str, str]:
    """
    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    pattern = r"^\[(ok|err|skip|ignore)\]:\s(.+?)(?:\s\((\d+\s*m?s)\))?$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name, _duration = match.groups()
            if status == "ok":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == "err":
                # Strip out file path information from failed test names
                test_name = re.sub(r"\s+in\s+\S+$", "", test_name)
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status in ("skip", "ignore"):
                test_status_map[test_name] = TestStatus.SKIPPED.value

    return test_status_map


def parse_log_jq(log: str) -> dict[str, str]:
    """
    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    pattern = r"^\s*(PASS|FAIL):\s(.+)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == "PASS":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == "FAIL":
                test_status_map[test_name] = TestStatus.FAILED.value
    return test_status_map


def parse_log_doctest(log: str) -> dict[str, str]:
    """
    Assumes test binary runs with -s -r=xml.
    """
    test_status_map = {}

    # Extract XML content
    start_tag = "<doctest"
    end_tag = "</doctest>"
    start_index = log.find(start_tag)
    end_index = (
        log.find(end_tag, start_index) + len(end_tag) if start_index != -1 else -1
    )

    if start_index != -1 and end_index != -1:
        xml_string = log[start_index:end_index]
        root = ET.fromstring(xml_string)

        for testcase in root.findall(".//TestCase"):
            testcase_name = testcase.get("name")
            for subcase in testcase.findall(".//SubCase"):
                subcase_name = subcase.get("name")
                name = f"{testcase_name} > {subcase_name}"

                expressions = subcase.findall(".//Expression")
                subcase_passed = all(
                    expr.get("success") == "true" for expr in expressions
                )

                if subcase_passed:
                    test_status_map[name] = TestStatus.PASSED.value
                else:
                    test_status_map[name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_micropython_test(log: str) -> dict[str, str]:
    test_status_map = {}

    pattern = r"^(pass|FAIL|skip)\s+(.+)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == "pass":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == "FAIL":
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status == "skip":
                test_status_map[test_name] = TestStatus.SKIPPED.value

    return test_status_map


def parse_log_googletest(log: str) -> dict[str, str]:
    test_status_map = {}

    pattern = r"^.*\[\s*(OK|FAILED)\s*\]\s(.*)\s\(.*\)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == "OK":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == "FAILED":
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_minitest(log: str) -> dict[str, str]:
    """
    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    pattern = r"^(.+)\. .*=.*(\.|F|E).*$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            test_name, outcome = match.groups()
            if outcome == ".":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif outcome in ["F", "E"]:
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_cucumber(log: str) -> dict[str, str]:
    """
    Assumes --format progress is used.
    """
    test_status_map = {}

    pattern = r"^(.*) \.+(\.|F)"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            test_name, outcome = match.groups()
            if outcome == ".":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif outcome == "F":
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_ruby_unit(log: str) -> dict[str, str]:
    test_status_map = {}

    pattern = r"^\s*(?:test: )?(.+):\s+(\.|E\b|F\b|O\b)"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            test_name, outcome = match.groups()
            if outcome == ".":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif outcome in ["E", "F"]:
                test_status_map[test_name] = TestStatus.FAILED.value
            elif outcome == "O":
                test_status_map[test_name] = TestStatus.SKIPPED.value

    return test_status_map


def parse_log_rspec_transformed_json(log: str) -> dict[str, str]:
    test_status_map = {}

    pattern = r"(.+) - (passed|failed)"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            test_name, outcome = match.groups()
            if outcome == "passed":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif outcome == "failed":
                test_status_map[test_name] = TestStatus.FAILED.value
            elif outcome == "pending":
                test_status_map[test_name] = TestStatus.SKIPPED.value
            else:
                raise ValueError(f"Unknown outcome: {outcome}")

    return test_status_map


def parse_log_cargo(log: str) -> dict[str, str]:
    """
    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    pattern = r"^test\s+(\S+)\s+\.\.\.\s+(\w+)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            test_name, outcome = match.groups()
            if outcome == "ok":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif outcome == "FAILED":
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_phpunit(log: str) -> dict[str, str]:
    """
    Parser for phpunit logs with the --testdox option.
    Args:
        log (str): log content
        test_spec (TestSpec): test spec (unused)
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    suite = None

    suite_pattern = r"^(\w.+) \(.+\)$"
    test_pattern = r"^\s*([✔✘↩])\s*(.*)$"
    # Strip trailing timing suffixes like "[1.34 ms]" or "[123 ms]" from test names.
    # PHPUnit --testdox appends these brackets when --display-incomplete or timing is enabled.
    _timing_suffix_re = re.compile(
        r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE
    )

    for line in log.split("\n"):
        suite_match = re.match(suite_pattern, line)
        if suite_match:
            suite = suite_match.groups()[0]
            continue

        test_match = re.match(test_pattern, line)
        if test_match:
            status, test_name = test_match.groups()
            # Remove timing suffix before building the key
            test_name = _timing_suffix_re.sub("", test_name).strip()
            full_test_name = f"{suite} > {test_name}"

            if status == "✔":
                test_status_map[full_test_name] = TestStatus.PASSED.value
            elif status == "✘":
                test_status_map[full_test_name] = TestStatus.FAILED.value
            elif status == "↩":
                test_status_map[full_test_name] = TestStatus.SKIPPED.value

    return test_status_map


def parse_log_maven(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with 'mvn test'.
    Annoyingly maven will not print the tests that have succeeded. For this log
    parser to work, each test must be run individually, and then we look for
    BUILD (SUCCESS|FAILURE) in the logs.

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    current_test_name = "---NO TEST NAME FOUND YET---"

    # Get the test name from the command used to execute the test.
    # Assumes we run evaluation with set -x
    test_name_pattern = r"^.*-Dtest=(\S+).*$"
    result_pattern = r"^.*BUILD (SUCCESS|FAILURE)$"

    for line in log.split("\n"):
        test_name_match = re.match(test_name_pattern, line.strip())
        if test_name_match:
            current_test_name = test_name_match.groups()[0]

        result_match = re.match(result_pattern, line.strip())
        if result_match:
            status = result_match.groups()[0]
            if status == "SUCCESS":
                test_status_map[current_test_name] = TestStatus.PASSED.value
            elif status == "FAILURE":
                test_status_map[current_test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_ant(log: str) -> dict[str, str]:
    test_status_map = {}

    pattern = r"^\s*\[junit\]\s+\[(PASS|FAIL|ERR)\]\s+(.*)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == "PASS":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status in ["FAIL", "ERR"]:
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_logs_kotlin_junit(log: str) -> dict[str, str]:
    """Parse JUnit/Maven Surefire output and return {test_class: status}.

    Rules:
      * Lines like: "Running <test_class>" -> mark test as seen
      * Lines like: "Tests run: X, Failures: Y, Errors: Z, Skipped: W" -> determine status:
        - If Failures > 0 or Errors > 0 -> FAILED
        - If Skipped > 0 (and no failures/errors) -> SKIPPED
        - Otherwise -> PASSED
    """
    results: dict[str, str] = {}

    # Regexes
    running_re = re.compile(r"^Running\s+(.+)$")
    summary_re = re.compile(
        r"^Tests run:\s+(\d+),\s+Failures:\s+(\d+),\s+Errors:\s+(\d+),\s+Skipped:\s+(\d+)"
    )

    current_test = None

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Check for "Running <test_class>"
        if m := running_re.match(line):
            current_test = m.group(1)
            continue

        # Check for summary line
        if m := summary_re.match(line):
            if current_test:
                tests_run = int(m.group(1))
                failures = int(m.group(2))
                errors = int(m.group(3))
                skipped = int(m.group(4))

                # Determine status based on the counts
                if failures > 0 or errors > 0:
                    results[current_test] = TestStatus.FAILED.value
                elif skipped > 0:
                    results[current_test] = TestStatus.SKIPPED.value
                else:
                    results[current_test] = TestStatus.PASSED.value

                current_test = None
            continue

    return results


def parse_log_gradle_custom(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with 'gradle test'. Assumes that the
    pre-install script to update the gradle config has run.
    """
    test_status_map = {}

    pattern = r"^([^>].+?)\s+(PASSED|FAILED)(?:\s+\(\d+(?:\.\d+)?s\))?$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            test_name, status = match.groups()
            if status == "PASSED":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == "FAILED":
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_calypso(log: str) -> dict[str, str]:
    """
    Parser for test logs generated by Calypso test suite
    """
    test_status_map = {}
    suite = []

    def get_test_name(suite: list[tuple[str, int]], match_pattern: str, line: str):
        test_names = " - ".join([x[0] for x in suite])
        if not (matched := re.match(match_pattern, line)):
            raise ValueError(f"Pattern {match_pattern} doesn't match line: {line}")

        return " - ".join([test_names, matched.group(1)]).strip()

    for log_chunk in log.split(" ./node_modules/.bin/jest ")[1:]:
        for line in log_chunk.split("\n"):
            if any(line.startswith(x) for x in ["Test Suites", "  ● "]):
                break
            if line.strip().startswith("✓"):
                # Test passed
                match_pattern = (
                    r"^\s+✓\s(.*)\(\d+ms\)$"
                    if re.search(r"\(\d+ms\)", line) is not None
                    else r"^\s+✓\s(.*)"
                )
                test_status_map[get_test_name(suite, match_pattern, line)] = (
                    TestStatus.PASSED.value
                )
            elif line.strip().startswith("✕"):
                # Test failed
                match_pattern = (
                    r"^\s+✕\s(.*)\(\d+ms\)$"
                    if re.search(r"\(\d+ms\)", line) is not None
                    else r"^\s+✕\s(.*)"
                )
                test_status_map[get_test_name(suite, match_pattern, line)] = (
                    TestStatus.FAILED.value
                )
            elif len(line) - len(line.lstrip()) > 0:
                # Adjust suite name
                indent = len(line) - len(line.lstrip())
                if len(suite) == 0:
                    # If suite is empty, initialize it
                    suite = [(line.strip(), indent)]
                else:
                    while len(suite) > 0 and suite[-1][-1] >= indent:
                        # Pop until the last element with indent less than current indent
                        suite.pop()
                    suite.append((line.strip(), indent))

    return test_status_map


def parse_log_chart_js(log: str) -> dict[str, str]:
    """
    Parser for test logs generated by ChartJS test suite
    """
    log = ansi_escape(log)
    test_status_map = {}
    failure_case_patterns = [
        # use [^\S\r\n] to avoid overlapping Chrome groups on separate lines
        (r"Chrome\s[\d\.]+[^\S\r\n]\(.+?\)[^\S\r\n](.*)FAILED$", re.MULTILINE),
    ]
    for failure_case_pattern, flags in failure_case_patterns:
        failures = re.findall(failure_case_pattern, log, flags)
        if len(failures) == 0:
            continue
        for failure in failures:
            test_status_map[failure] = TestStatus.FAILED.value
    return test_status_map


def parse_log_marked(log: str) -> dict[str, str]:
    """
    Parser for test logs generated by Marked test suite
    """
    test_status_map = {}
    for line in log.split("\n"):
        if match := re.search(r"^\d+\)\s(.*)", line):
            test = match.group(1)
            test_status_map[test.strip()] = TestStatus.FAILED.value
    return test_status_map


def parse_log_p5js(log: str) -> dict[str, str]:
    def remove_json_blocks(log_content: str) -> str:
        filtered_lines = []
        in_json_block = False
        in_json_list_block = False
        for line in log_content.split("\n"):
            stripped_line = line.rstrip()  # Remove trailing whitespace
            if stripped_line.endswith("{"):
                in_json_block = True
                continue
            if stripped_line.endswith("["):
                in_json_list_block = True
                continue
            if stripped_line == "}" and in_json_block:
                in_json_block = False
                continue
            if stripped_line == "]" and in_json_list_block:
                in_json_list_block = False
                continue
            if in_json_block or in_json_list_block:
                continue
            if stripped_line.startswith("{") and stripped_line.endswith("}"):
                continue
            if stripped_line.startswith("[") and stripped_line.endswith("]"):
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines)

    def remove_xml_blocks(log_content: str) -> str:
        xml_pat = re.compile(r"<(\w+)>[\s\S]*?<\/\1>", re.MULTILINE)
        match = xml_pat.search(log_content)
        while match:
            # count the number of opening tags in the match
            opening_tags = match.group().count(rf"<{match.group(1)}>") - 1
            opening_tags = max(opening_tags, 0)
            start = match.start()
            end = match.end()
            log_content = (
                log_content[:start]
                + f"<{match.group(1)}>" * opening_tags
                + log_content[end:]
            )
            match = xml_pat.search(log_content)
        return log_content

    def is_valid_fail(match: re.Match) -> bool:
        last_line_indent = 0
        for line in match.group(2).split("\n"):
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= last_line_indent:
                return False
            last_line_indent = line_indent
        return True

    log = ansi_escape(log)
    log = remove_json_blocks(log)
    log = remove_xml_blocks(log)
    test_results = {}

    # Parse failing tests
    fail_pattern = re.compile(r"^\s*(\d+)\)(.{0,1000}?):", re.MULTILINE | re.DOTALL)
    for match in fail_pattern.finditer(log):
        if is_valid_fail(match):
            test_names = list(map(str.strip, match.group(2).split("\n")))
            full_name = ":".join(test_names)
            test_results[full_name] = TestStatus.FAILED.value

    return test_results


def parse_log_react_pdf(log: str) -> dict[str, str]:
    """
    Parser for test logs generated by Carbon test suite
    """
    test_status_map = {}
    # Match PASS/FAIL followed by test name and optional timing like (1.23ms), (1.23 s), (1.23s)
    _timing_suffix = r"(?:\s\([\d.]+\s*(?:ms|s)\))?"
    patterns = [
        (re.compile(rf"^PASS\s(.*?){_timing_suffix}$"), TestStatus.PASSED.value),
        (re.compile(rf"^FAIL\s(.*?){_timing_suffix}$"), TestStatus.FAILED.value),
    ]
    for line in log.split("\n"):
        for pattern, status in patterns:
            if matched := pattern.search(line):
                test_name = matched.group(1).rstrip()
                test_status_map[test_name] = status
                break
    return test_status_map


def parse_log_jest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Jest. Assumes --verbose flag.

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    pattern = r"^\s*(✓|✕|○)\s(.+?)(?:\s\((\d+\s*m?s)\))?$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status_symbol, test_name, _duration = match.groups()
            if status_symbol == "✓":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status_symbol == "✕":
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status_symbol == "○":
                test_status_map[test_name] = TestStatus.SKIPPED.value
    return test_status_map


def parse_log_jest_json(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Jest. Assumes the --json flag has been
    piped into JEST_JSON_JQ_TRANSFORM. Unlike --verbose, tests with the same name
    in different describe blocks print with different names.
    """
    test_status_map = {}

    pattern = r"^\[(PASSED|FAILED)\]\s(.+)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == "PASSED":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == "FAILED":
                test_status_map[test_name] = TestStatus.FAILED.value
    return test_status_map


def parse_log_vitest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with vitest. Assumes --reporter=verbose flag.
    """
    test_status_map = {}

    pattern = r"^\s*(✓|×|↓)\s(.+?)(?:\s(\d+\s*m?s?|\[skipped\]))?$"  # noqa: RUF001

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status_symbol, test_name, _duration_or_skipped = match.groups()
            if status_symbol == "✓":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status_symbol == "×":  # noqa: RUF001
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status_symbol == "↓":
                test_status_map[test_name] = TestStatus.SKIPPED.value
    return test_status_map


def parse_log_karma(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Karma. Handles duplicate test names in
    different describe blocks. Logic is brittle.
    """
    test_status_map = {}
    current_indent = -1
    current_suite = []
    started = False

    pattern = r"^(\s*)?([✔✖])?\s(.*)$"

    for line in log.split("\n"):
        if line.startswith("SUMMARY:"):
            # Individual test logs end here
            return test_status_map

        if "Starting browser" in line:
            started = True
            continue

        if not started:
            continue

        match = re.match(pattern, line)
        if match:
            indent, status, name = match.groups()

            if indent and not status:
                new_indent = len(indent)
                if new_indent > current_indent:
                    current_indent = new_indent
                    current_suite.append(name)
                elif new_indent < current_indent:
                    current_indent = new_indent
                    current_suite.pop()
                    continue

            if status in ("✔", "✖"):
                full_test_name = " > ".join(current_suite + [name])
                test_status_map[full_test_name] = (
                    TestStatus.PASSED.value
                    if status == "✔"
                    else TestStatus.FAILED.value
                )

    return test_status_map


def parse_log_tap(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with TAP

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    # Pattern to match TAP result lines
    pattern = r"^(ok|not ok) (\d+) (.+)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, _test_number, test_name = match.groups()
            if status == "ok":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == "not ok":
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_cpp(log: str) -> dict[str, str]:
    """Parse pytest output and return {test_name: status}.

    Rules:
      * Lines like: "tests/test_ujson.py::test_name PASSED ..." -> PASSED
      * Lines like: "tests/test_ujson.py::test_name FAILED ..." -> FAILED
      * Lines like: "tests/test_ujson.py::test_name SKIPPED ..." -> SKIPPED
      * Lines like: "tests/test_ujson.py::test_name ERROR ..." -> ERROR
      * Failed tests in FAILURES section override PASSED if present
      * Test names are extracted as the part after the last '::' for consistency
    """
    results: dict[str, str] = {}

    # Regexes for test results (match up to PASSED/FAILED etc., ignore rest)
    passed_re = re.compile(r"^(.*?)\s+PASSED")
    failed_re = re.compile(r"^(.*?)\s+FAILED")
    skipped_re = re.compile(r"^(.*?)\s+SKIPPED")
    error_re = re.compile(r"^(.*?)\s+ERROR")

    # Also check FAILURES section for failed tests
    in_failures = False
    failure_test_re = re.compile(
        r"^___________________________ (.*?) ___________________________$"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Check for FAILURES section start
        if line.startswith(
            "=================================== FAILURES ==================================="
        ):
            in_failures = True
            continue

        if in_failures:
            if m := failure_test_re.match(line):
                test_name = m.group(1)
                results[test_name] = TestStatus.FAILED.value
            continue

        # Parse individual test lines
        if m := passed_re.match(line):
            full_name = m.group(1)
            test_name = full_name.split("::")[-1] if "::" in full_name else full_name
            results.setdefault(test_name, TestStatus.PASSED.value)
            continue
        if m := failed_re.match(line):
            full_name = m.group(1)
            test_name = full_name.split("::")[-1] if "::" in full_name else full_name
            results[test_name] = TestStatus.FAILED.value
            continue
        if m := skipped_re.match(line):
            full_name = m.group(1)
            test_name = full_name.split("::")[-1] if "::" in full_name else full_name
            results[test_name] = TestStatus.SKIPPED.value
            continue
        if m := error_re.match(line):
            full_name = m.group(1)
            test_name = full_name.split("::")[-1] if "::" in full_name else full_name
            results[test_name] = TestStatus.ERROR.value
            continue

    return results


def parse_log_cpp_v2(log: str) -> dict[str, str]:
    """Parse C++ test output and return {test_name: status}.

    Rules:
      * Lines like: "Test <name>                   passed" -> PASSED (tentative)
      * Lines like: "Test <name>                   failed" -> FAILED (overrides prior PASS)
      * Lines like: "Test <name>                   skipped" -> SKIPPED
    """
    results: dict[str, str] = {}

    # Regexes
    passed_re = re.compile(r"^Test\s+(.*?)\s+passed$")
    failed_re = re.compile(r"^Test\s+(.*?)\s+failed$")
    skipped_re = re.compile(r"^Test\s+(.*?)\s+skipped$")

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := passed_re.match(line):
            results.setdefault(m.group(1), TestStatus.PASSED.value)
            continue
        if m := failed_re.match(line):
            results[m.group(1)] = TestStatus.FAILED.value
            continue
        if m := skipped_re.match(line):
            results[m.group(1)] = TestStatus.SKIPPED.value
            continue
    return results


def parse_log_cpp_v3(log: str) -> dict[str, str]:
    """Parse C++/googletest-like output and return {test_name: status}.

    This function targets log snippets like the sample in the notebook where lines
    look like:
      "[12/414] File: Save and load black gray-scale jpeg image... OK"
      "[39/414] image_function::GetThreshold (form 1)... OK"
      "[123/414] SomeTest::TestName... FAILED"

    Also handles Botan test runner group-summary lines:
      "AES-128 ran 17370 tests in 34.33 msec all ok"
      "ChaCha ran 2048 tests in 12.1 msec 3 tests failed"

    Rules implemented:
      - Lines ending with "... OK" or "... OK" indicate PASSED.
      - Lines ending with "... FAILED" or containing "FAIL" indicate FAILED.
      - Lines containing "SKIPPED" indicate SKIPPED.
      - The parser extracts the human-friendly test name found after the optional prefix
        like "[12/414]" and before the ellipsis '...'. If there is a parenthesized
        suffix like "(form 1)", it will be included in the test name.
      - Botan summary lines: timing is stripped from the key so that the key matches
        the normalised form used in the evaluation dataset.

    Returns:
      dict mapping extracted test name to one of: 'PASSED','FAILED','SKIPPED','ERROR'.
    """
    results: dict[str, str] = {}

    # Botan group-summary: "NAME ran N tests in X.XX msec all ok"
    # or "NAME ran N tests in X.XX msec N tests failed"
    _botan_re = re.compile(
        r"^(.*?)\s+ran\s+(\d+)\s+tests\s+in\s+[\d.]+\s+(?:msec|sec)\s+(.+)$",
        re.IGNORECASE,
    )
    _botan_fail_re = re.compile(r"\bfailed\b", re.IGNORECASE)

    # Patterns
    # Example: [12/414] File: Save and load ... OK
    line_re = re.compile(r"^(?:\[[0-9]+/[0-9]+\]\s*)?(.*?)\s*\.\.\.\s*(\w+)$")
    # Fallback to detect failures or skips embedded differently
    failed_re = re.compile(r"\bFAILED\b|\bFAIL\b|\bFailure\b", re.IGNORECASE)
    skipped_re = re.compile(r"\bSKIPPED\b", re.IGNORECASE)
    passed_token_re = re.compile(r"\bOK\b|\bPASSED\b", re.IGNORECASE)

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Botan group-summary line — check before generic patterns.
        bm = _botan_re.match(line)
        if bm:
            name, count, tail = bm.group(1).strip(), bm.group(2), bm.group(3).strip()
            # Key without timing: "NAME ran N tests <tail>" (tail = "all ok" / "N tests failed")
            key = f"{name} ran {count} tests {tail}"
            if _botan_fail_re.search(tail):
                results[key] = TestStatus.FAILED.value
            else:
                results[key] = TestStatus.PASSED.value
            continue

        # Direct structured match: capture name before '... <STATUS>'
        m = line_re.match(line)
        if m:
            name = m.group(1).strip()
            status_token = m.group(2).upper()
            if skipped_re.search(status_token) or skipped_re.search(line):
                results[name] = TestStatus.SKIPPED.value
            elif failed_re.search(status_token) or failed_re.search(line):
                results[name] = TestStatus.FAILED.value
            elif passed_token_re.search(status_token) or passed_token_re.search(line):
                results[name] = TestStatus.PASSED.value
            else:
                results[name] = TestStatus.ERROR.value
            continue

        # If no structured '... STATUS' but contains keywords, try to extract a name
        if skipped_re.search(line):
            # try to take whole line as name
            results[line] = TestStatus.SKIPPED.value
            continue
        if failed_re.search(line):
            results[line] = TestStatus.FAILED.value
            continue
        if passed_token_re.search(line):
            results[line] = TestStatus.PASSED.value
            continue

    return results


def parse_log_cpp_v4(log: str) -> dict[str, str]:
    """Minimal parser: only handles summary lines of the exact form used in the notebook sample."""
    results: dict[str, str] = {}
    # Capture: digits/digits Test   #n: <name> ... <Status>
    re_summary = re.compile(
        r"^\s*\d+/\d+\s+Test\s+#\d+:\s+(.+?)\s+\.+\s+(Passed|Failed|Skipped|Timeout)\b",
        re.IGNORECASE,
    )
    for line in log.splitlines():
        if not line.strip():
            continue
        m = re_summary.match(line)
        if not m:
            continue
        name = m.group(1).strip()
        status = m.group(2).strip().upper()
        if status == "PASSED":
            results[name] = TestStatus.PASSED.value
        elif status in ("FAILED", "TIMEOUT"):
            results[name] = TestStatus.FAILED.value
        else:
            results[name] = TestStatus.SKIPPED.value
    return results


def parse_lue_nvim(log: str) -> dict[str, str]:
    results: dict[str, str] = {}

    # Precompiled regexes
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    success_re = re.compile(r"^Success\s*\|\|\s*(.+?)\s*$")
    fail_re = re.compile(r"^(?:Fail|Failed)\s*\|\|\s*(.+?)\s*$")
    skip_re = re.compile(r"^(?:Skip|Skipped)\s*\|\|\s*(.+?)\s*$", re.IGNORECASE)

    for raw in log.splitlines():
        if not raw.strip():
            continue
        # Remove ANSI escapes then trim whitespace & trailing tabs
        line = ansi_re.sub("", raw).rstrip()

        # Ignore summary counters (they have a colon right after keyword and no '||')
        if ":" in line.split("\t")[0] and "||" not in line:
            # e.g., "Success:" or "Failed :" or "Errors :"
            # ensure we don't skip real test lines which always have '||'
            continue

        # Match order: fail overrides, skip, success (success uses setdefault)
        m_fail = fail_re.match(line)
        if m_fail:
            name = m_fail.group(1).strip()
            results[name] = TestStatus.FAILED.value
            continue

        m_skip = skip_re.match(line)
        if m_skip:
            name = m_skip.group(1).strip()
            # Only set if not already failed
            results.setdefault(name, TestStatus.SKIPPED.value)
            continue

        m_success = success_re.match(line)
        if m_success:
            name = m_success.group(1).strip()
            # Don't overwrite FAIL (or SKIPPED) with PASS
            results.setdefault(name, TestStatus.PASSED.value)
            continue

    return results


def _mvn_failure_status(line: str) -> str:
    if "Exception" in line and "AssertionError" not in line:
        return TestStatus.ERROR.value
    return TestStatus.FAILED.value


def _mvn_summary_class(line: str, current_running: str | None) -> str | None:
    if " in " in line:
        return line.rsplit(" in ", 1)[-1].strip().rstrip(".:;")
    return current_running


def _mvn_status_from_summary_counts(failures: int, errors: int, skipped: int) -> str:
    if failures == 0 and errors == 0:
        if skipped == 0:
            return TestStatus.PASSED.value
        return TestStatus.SKIPPED.value
    return TestStatus.FAILED.value


def parse_java_mvn(log: str) -> dict[str, str]:
    """Parse Maven logs into {identifier: status}.

    Supports both:
      * Surefire/Failsafe style lines:
        - `[INFO] Running <class>`
        - `Tests run: X, Failures: Y, Errors: Z, Skipped: W`
        - `[ERROR] <class>.<method>:<line> ...`
      * `mvn -Dtest=...` flows that only emit `BUILD SUCCESS|FAILURE`
        (delegates to `parse_log_maven` as a fallback signal source).
    """
    results: dict[str, str] = {}

    failure_re = re.compile(
        r"^\[ERROR\]\s+(?P<class>[A-Za-z0-9_$.]+)\.(?P<method>[A-Za-z0-9_$.]+?)(?::(?P<line>\d+))?\b"
    )
    running_re = re.compile(r"^(?:\[INFO\]\s+)?Running\s+([A-Za-z0-9_$.]+)$")
    summary_re = re.compile(
        r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
    )

    seen_running: set[str] = set()
    current_running: str | None = None

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        rm = running_re.match(line)
        if rm:
            running_name = rm.group(1)
            current_running = running_name
            seen_running.add(running_name)
            continue
        fm = failure_re.match(line)
        if fm:
            clazz = fm.group("class")
            method = fm.group("method")
            test_id = f"{clazz}.{method}"
            results[test_id] = _mvn_failure_status(line)
            results.setdefault(clazz, TestStatus.FAILED.value)
            continue

        sm = summary_re.search(line)
        if sm:
            _tests_run, failures, errors, skipped = map(int, sm.groups())
            clazz = _mvn_summary_class(line, current_running)
            if clazz:
                status = _mvn_status_from_summary_counts(failures, errors, skipped)
                results.setdefault(clazz, status)
            current_running = None

    # Fallback for logs where only junit XML snippets are emitted inline.
    for test_name, status in parse_log_junit(log).items():
        results.setdefault(test_name, status)

    # Fallback for logs where only `-Dtest=<name>` + BUILD SUCCESS/FAILURE exists.
    for test_name, status in parse_log_maven(log).items():
        results.setdefault(test_name, status)

    if not results and seen_running:
        for clazz in seen_running:
            results.setdefault(clazz, TestStatus.PASSED.value)
    return results


def parse_log_sbt(log: str) -> dict[str, str]:
    """Parse Scala sbt test output (JUnit XML format) and return {full_test_name: status}.

    Rules:
      * Parse XML testsuite elements from the log
      * Extract testcase elements with their name and classname
      * Determine status: PASSED (no failure/error/skipped), FAILED, ERROR, or SKIPPED
    """
    return _parse_junit_testcases_from_text(log, joiner=" ")


parse_log_junit = parse_log_sbt


def _parse_junit_testcases_from_text(log: str, joiner: str = " ") -> dict[str, str]:
    """Parse junit-like testcase fragments from raw text.

    Works with:
      * attribute order variations (name before classname, etc.);
      * both self-closing and expanded testcase tags;
      * concatenated XML documents in noisy logs.
    """
    results: dict[str, str] = {}

    open_tag_re = re.compile(r"<testcase\b([^>]*)>", re.DOTALL)
    attr_re = re.compile(r'(\w+)="([^"]*)"')

    pos = 0
    while True:
        match = open_tag_re.search(log, pos)
        if match is None:
            break

        raw_attrs = match.group(1) or ""
        attrs = dict(attr_re.findall(raw_attrs))
        name = attrs.get("name")
        classname = attrs.get("classname", "")

        if not name:
            pos = match.end()
            continue

        is_self_closing = raw_attrs.strip().endswith("/")
        content = ""
        if not is_self_closing:
            close_pos = log.find("</testcase>", match.end())
            if close_pos != -1:
                content = log[match.end() : close_pos]
                pos = close_pos + len("</testcase>")
            else:
                # Malformed logs: fallback to a bounded window.
                content = log[match.end() : match.end() + 4096]
                pos = match.end()
        else:
            pos = match.end()

        full_name = f"{classname}{joiner}{name}".strip() if classname else name

        if "<failure" in content:
            status = TestStatus.FAILED.value
        elif "<error" in content:
            status = TestStatus.ERROR.value
        elif "<skipped" in content:
            status = TestStatus.SKIPPED.value
        else:
            status = TestStatus.PASSED.value

        results[full_name] = status

    return results


def parse_java_mvn_v2(log: str) -> dict[str, str]:
    """Parse Maven Surefire / Failsafe style logs and extract test module status.

    Returns a mapping {module_or_test_identifier: status} where status in
    (PASSED, FAILED, SKIPPED, ERROR).

    Heuristics:
    - Lines containing ' ... SUCCESS ' => PASSED
    - Lines containing ' ... FAILURE ' => FAILED
    - Lines containing ' ... SKIPPED' => SKIPPED
    - Summary line 'Tests run: X, Failures: Y, Errors: Z, Skipped: W' is parsed but
      only used to potentially add an aggregate key '__suite__' if helpful.
    - If summary shows Errors > 0 but we didn't mark any specific test as ERROR,
      we won't guess individual names (lack of detail in provided excerpt).

    The key chosen is the module segment between the Maven prefix '[INFO] ' and the status word.
    Multiple spaces or alignment dots are stripped.
    """

    results: dict[str, str] = {}

    # Regex to capture module name and status keywords
    # Example line:
    # [INFO] Piranha - HTTP - Implementation .................... FAILURE [  1.241 s]
    line_re = re.compile(
        r"^\[INFO\]\s+(?P<name>.+?)\s+\.\.+\s+(?P<status>SUCCESS|FAILURE|SKIPPED)(?:\s|\[|$)"
    )
    # Summary line:
    summary_re = re.compile(
        r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = line_re.match(line)
        if m:
            name = m.group("name")
            status_word = m.group("status")
            if status_word == "SUCCESS":
                status = TestStatus.PASSED.value
            elif status_word == "FAILURE":
                status = TestStatus.FAILED.value
            elif status_word == "SKIPPED":
                status = TestStatus.SKIPPED.value
            else:  # Should not happen
                status = TestStatus.ERROR.value
            results[name] = status
            continue
        sm = summary_re.search(line)
        if sm:
            tests_run, failures, errors, skipped = map(int, sm.groups())
            # Optionally store an aggregate entry
            aggregate_status = TestStatus.PASSED.value
            if failures > 0:
                aggregate_status = TestStatus.FAILED.value
            elif errors > 0:
                aggregate_status = TestStatus.ERROR.value
            elif skipped == tests_run:
                aggregate_status = TestStatus.SKIPPED.value
            results.setdefault("__suite__", aggregate_status)

    return results


def parse_log_php_v1(log: str) -> dict[str, str]:  # noqa: PLR0912, PLR0915
    """Parse a PHPUnit style test execution log into {test_name: status}.

    Adjustments (update):
      * Remove trailing timing tokens like "0.03s" from captured names.
      * Ignore summary line starting with "Tests:".
      * Ignore suite-level PASS lines (e.g. "PASS  Tests\\Foo\\BarTest").
      * Keep individual test method lines that start with a check mark (✓ / ✔) or cross (Unicode U+2A2F) for failures.

    Heuristics:
      * Passing test line: leading optional spaces then (✓|✔) then name then timing.
      * Failing test line: leading optional spaces then (Unicode U+2A2F|x) then name then timing.
      * Skipped test line: leading dash "-" then name then timing OR contains inline '(skipped)'.
      * Suite FAIL line ("FAIL  ClassName") marks the suite as FAILED (kept) if desired context.
      * Summary lines and dividers are ignored.
    """
    results: dict[str, str] = {}
    cross_mark = chr(0x2A2F)

    def clean(name: str) -> str:
        # Drop trailing timing tokens and extra spaces
        name = name.rstrip()
        name = re.sub(r"\s{2,}\d+(?:\.\d+)?s\s*$", "", name)  # two+ spaces then timing
        name = re.sub(
            r"\s+\d+(?:\.\d+)?s\s*$", "", name
        )  # fallback single space timing
        return name.strip()

    pass_line_re = re.compile(r"^\s*(?:✓|✔)\s+(?P<name>.+?)\s{2,}\d+\.?\d*s\b.*$")
    pass_line_no_timing_re = re.compile(r"^\s*(?:✓|✔)\s+(?P<name>.+?)\s*$")

    fail_line_re = re.compile(
        rf"^\s*(?:{cross_mark}|x)\s+(?P<name>.+?)\s{{2,}}\d+\.?\d*s\b.*$",
        re.IGNORECASE,
    )
    fail_line_no_timing_re = re.compile(
        rf"^\s*(?:{cross_mark}|x)\s+(?P<name>.+?)\s*$", re.IGNORECASE
    )

    suite_fail_re = re.compile(r"^\s*FAIL(?:ED)?\s+(?P<name>\S.+)$", re.IGNORECASE)
    # (We intentionally DO NOT capture PASS <suite> lines)

    skipped_line_re = re.compile(r"^\s*-\s+(?P<name>.+?)\s{2,}\d+\.?\d*s\b.*$")
    skipped_line_no_timing_re = re.compile(r"^\s*-\s+(?P<name>.+?)\s*$")

    inline_skipped_marker = re.compile(r"\(skipped\)|\bSKIPPED\b", re.IGNORECASE)

    for raw in log.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        stripped = line.strip()

        # Ignore dividers and summary lines
        if stripped.startswith(("___", "---", "Tests:")):
            continue
        if stripped.startswith("Duration:"):
            continue

        # Suite-level FAIL
        m = suite_fail_re.match(line)
        if m:
            name = clean(m.group("name"))
            results[name] = TestStatus.FAILED.value
            continue

        # Failing test cases (with or without timing)
        for rx in (fail_line_re, fail_line_no_timing_re):
            m = rx.match(line)
            if m:
                name = clean(m.group("name"))
                if name:  # avoid empty
                    results[name] = TestStatus.FAILED.value
                break
        else:
            # Skipped test cases
            matched_skip = False
            for rx in (skipped_line_re, skipped_line_no_timing_re):
                m = rx.match(line)
                if m:
                    name = clean(m.group("name"))
                    if name:
                        results[name] = TestStatus.SKIPPED.value
                    matched_skip = True
                    break
            if matched_skip:
                continue

            # Passing test cases (no PASS suite lines)
            for rx in (pass_line_re, pass_line_no_timing_re):
                m = rx.match(line)
                if m:
                    name = clean(m.group("name"))
                    if name:
                        results.setdefault(name, TestStatus.PASSED.value)
                    break
            else:
                # Inline skipped marker inside an otherwise pass-looking line
                if inline_skipped_marker.search(line):
                    # Heuristic: strip timing and leading markers
                    token = re.sub(rf"^\s*(?:✓|✔|{cross_mark}|x|-)\s+", "", line)
                    token = clean(token)
                    if token and token not in results:
                        results[token] = TestStatus.SKIPPED.value
                # Otherwise ignore noise / stack traces / diffs
                continue

    return results


def parse_log_ruby_v2(log: str) -> dict[str, str]:
    """Parse Ruby test output and return {full_test_name: status}.

    This function parses Ruby test logs similar to parse_log_elixir.

    Rules:
      * Lines like: "TestClass#test_method = 5.30 s = ." -> PASSED
      * Lines like: "TestClass#test_method = 5.30 s = F" -> FAILED
      * Lines like: "TestClass#test_method = 5.30 s = E" -> ERROR (treated as FAILED)
      * Lines like: "TestClass#test_method = 5.30 s = S" -> SKIPPED
      * Failure sections starting with "Failure:" provide additional context but test status
        is determined by the summary line with = symbol
    """
    results: dict[str, str] = {}

    # Regex patterns for Ruby test output
    # Match lines like: "TestClass#test_method = 5.30 s = ."
    test_result_re = re.compile(r"^(.+?#.+?)\s+=\s+[\d.]+\s+s\s+=\s+([.FES])$")

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Check for test result lines
        match = test_result_re.match(line)
        if match:
            test_name = match.group(1)
            status_char = match.group(2)

            if status_char == ".":
                results[test_name] = TestStatus.PASSED.value
            elif status_char == "F":
                results[test_name] = TestStatus.FAILED.value
            elif status_char == "E":
                results[test_name] = TestStatus.FAILED.value  # Treat errors as failures
            elif status_char == "S":
                results[test_name] = TestStatus.SKIPPED.value

    return results


def _set_test_status(results: dict[str, str], name: str, status: str) -> None:
    """Set test status by converting a status word (e.g. 'PASSED') to TestStatus enum value."""
    results[name] = getattr(TestStatus, status).value


def _update_status_by_precedence(
    results: dict[str, str], name: str, status: str
) -> None:
    """Set test status, overriding only if new status has higher precedence."""
    if not name:
        return
    current = results.get(name)
    if current is None or OCAML_STATUS_PRECEDENCE.get(
        status, -1
    ) >= OCAML_STATUS_PRECEDENCE.get(current, -1):
        results[name] = status


def parse_log_haskell(log: str) -> dict[str, str]:
    """Parse common Haskell test outputs (tasty/hspec/HUnit) and return {test_name: status}.

    Heuristics supported:
      - Tasty-style: "<name>: OK|FAIL|ERROR|SKIP [..]"
      - Hspec checkmarks: "✓ <name>" -> PASSED, "✗ <name>" -> FAILED
      - Hspec brackets: "<name> [✔]" -> PASSED, "<name> [✗]" -> FAILED
      - HUnit headers: "### Failure in: <name>", "### Error in: <name>"
      - Pending/skip as "SKIP" or "PENDING" map to SKIPPED
    Later FAIL/ERROR overrides prior PASS of the same test.
    """
    results: dict[str, str] = {}

    # Strip ANSI escape codes to normalize parsing
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    # Tasty-style per-test line: "  Foo.Bar.baz: OK (0.01s)"
    tasty_line = re.compile(
        r"^\s*(?P<name>.+?)\s*:\s*(?P<status>OK|PASS|FAIL|ERROR|SKIP|PENDING)(?:\b|\s|$)"
    )

    # Hspec checkmarks (may include timing like (0.01s))
    hspec_pass = re.compile(r"^\s*(?:✓|✔)\s+(?P<name>.+?)(?:\s+\(.*?\))?\s*$")
    hspec_fail = re.compile(r"^\s*(?:✗|✘)\s+(?P<name>.+?)\s*$")

    # Hspec brackets style: "  test name [✔]" or "  test name [✗]"
    hspec_bracket_pass = re.compile(r"^\s+(?P<name>.+?)\s+\[(?:✓|✔)\]\s*$")
    hspec_bracket_fail = re.compile(r"^\s+(?P<name>.+?)\s+\[(?:✗|✘)\]\s*$")

    # Hspec failure list headers like: "1) Some feature does X"
    hspec_failure_header = re.compile(r"^\s*\d+\)\s+(?P<name>.+?)\s*$")

    # HUnit headers
    hunit_failure = re.compile(r"^###\s+Failure\s+in:\s+(?P<name>.+)$")
    hunit_error = re.compile(r"^###\s+Error\s+in:\s+(?P<name>.+)$")

    # Skip/pending variants
    pending_inline = re.compile(r"^\s*(?P<name>.+?)\s*:\s*(?:SKIP|PENDING)\b.*$")

    for raw in log.splitlines():
        line = ansi_re.sub("", raw).rstrip()
        if not line:
            continue
        # Ignore obvious non-test noise
        if any(
            k in line for k in ("Test suite ", "Linking ", "Building ", "Running  ")
        ):
            continue

        if m := tasty_line.match(line):
            name = m.group("name").strip()
            status_word = m.group("status")
            status_map = {
                "OK": "PASSED",
                "PASS": "PASSED",
                "FAIL": "FAILED",
                "ERROR": "ERROR",
                "SKIP": "SKIPPED",
                "PENDING": "SKIPPED",
            }
            _set_test_status(results, name, status_map[status_word])
            continue

        if m := pending_inline.match(line):
            _set_test_status(results, m.group("name").strip(), "SKIPPED")
            continue

        if m := hspec_bracket_pass.match(line):
            name = m.group("name").strip()
            results.setdefault(name, TestStatus.PASSED.value)
            continue

        if m := hspec_bracket_fail.match(line):
            _set_test_status(results, m.group("name").strip(), "FAILED")
            continue

        if m := hspec_pass.match(line):
            # Tentatively pass unless overridden later by a failure
            name = m.group("name").strip()
            results.setdefault(name, TestStatus.PASSED.value)
            continue

        if m := hspec_fail.match(line):
            _set_test_status(results, m.group("name").strip(), "FAILED")
            continue

        if m := hspec_failure_header.match(line):
            _set_test_status(results, m.group("name").strip(), "FAILED")
            continue

        if m := hunit_failure.match(line):
            _set_test_status(results, m.group("name").strip(), "FAILED")
            continue

        if m := hunit_error.match(line):
            _set_test_status(results, m.group("name").strip(), "ERROR")
            continue

    return results


def parse_log_haskell_v2(log: str) -> dict[str, str]:  # noqa: C901, PLR0912, PLR0915
    lines = log.splitlines()

    # 1) Collect explicit failures from the Failures: section
    failed_fullnames = set()
    failures_idx = None
    for idx, ln in enumerate(lines):
        if ln.strip() == "Failures:":
            failures_idx = idx
            break

    if failures_idx is not None:
        failure_header_re = re.compile(r"^\s*\d+\)\s+(.*)$")
        i = failures_idx + 1
        while i < len(lines):
            s = lines[i].strip()
            if not s:
                i += 1
                continue
            if s.startswith(
                (
                    "To rerun use:",
                    "Randomized with seed",
                    "Finished in ",
                    "Test suite ",
                )
            ):
                break
            m = failure_header_re.match(s)
            if m:
                header = m.group(1).strip()
                parts = [p.strip() for p in header.split(",")]
                if parts:
                    full = ".".join([parts[0]] + parts[1:])
                    failed_fullnames.add(full)
            i += 1

    # 2) Find the tree section (between RUNNING and Failures/summary)
    start_idx = None
    for idx, ln in enumerate(lines):
        if ln.startswith("Test suite ") and ": RUNNING..." in ln:
            start_idx = idx + 1
            break
    if start_idx is None:
        # Fallback heuristic for when RUNNING marker isn't present
        for idx, ln in enumerate(lines):
            if (
                ln
                and not ln.startswith(" ")
                and "." in ln
                and not any(
                    k in ln
                    for k in (
                        "Downloading",
                        "Building",
                        "Installing",
                        "Completed",
                        "Starting",
                        "Preprocessing",
                        "Configuring",
                        "Linking",
                        "Resolving dependencies",
                        "Build profile",
                        "In order, the following will be built",
                        "Error:",
                        "Test suite ",
                        "Running ",
                    )
                )
            ):
                start_idx = idx
                break
    end_idx = failures_idx if failures_idx is not None else len(lines)

    def is_info(text: str) -> bool:
        return text == "Golden and Actual output didn't change"

    def leading_spaces(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    # Helper to peek next significant tree line (skips blanks and info lines)
    def next_sig_index(k: int):
        m = k + 1
        while m < end_idx:
            t = lines[m].rstrip("\n")
            s = t.strip()
            if not s or is_info(s):
                m += 1
                continue
            return m
        return None

    results: dict[str, str] = {}
    stack: list[str] = []

    i = start_idx or 0
    while i < end_idx:
        raw = lines[i].rstrip("\n")
        s = raw.strip()
        if not s or is_info(s):
            i += 1
            continue

        # Compute level by indentation (2 spaces per level in Hspec output)
        indent = leading_spaces(raw)
        level = indent // 2

        # Normalize leaf name and detect inline status (FAILED/PENDING)
        status_hint = None
        m_fail = re.match(r"^(.*?)\s+FAILED\s*\[\d+\]$", s)
        m_pending = re.match(r"^(.*?)\s+(?:PENDING|SKIPPED)\b.*$", s, re.IGNORECASE)
        if m_fail:
            name_core = m_fail.group(1).strip()
            status_hint = TestStatus.FAILED.value
        elif m_pending:
            name_core = m_pending.group(1).strip()
            status_hint = TestStatus.SKIPPED.value
        else:
            name_core = s

        # Update hierarchy stack at this level
        if level < len(stack):
            stack[level] = name_core
            del stack[level + 1 :]
        elif level == len(stack):
            stack.append(name_core)
        else:
            while len(stack) < level:
                stack.append("")
            stack.append(name_core)

        # Determine if this is a leaf example: next significant line not deeper
        j = next_sig_index(i)
        is_leaf = True
        if j is not None:
            next_indent = leading_spaces(lines[j])
            if next_indent > indent:
                # Child group/example exists; current is a container
                is_leaf = False

        if is_leaf:
            full_name = ".".join([seg for seg in stack if seg])
            if status_hint == TestStatus.FAILED.value:
                results[full_name] = TestStatus.FAILED.value
            elif status_hint == TestStatus.SKIPPED.value:
                results[full_name] = TestStatus.SKIPPED.value
            else:
                results[full_name] = TestStatus.PASSED.value

        i += 1

    # Override with failures discovered in the failures section
    for full in failed_fullnames:
        results[full] = TestStatus.FAILED.value

    return results


_JS_DURATION_SUFFIX_RE = re.compile(
    r"\s*(?:\(\s*)?\d+(?:\.\d+)?\s*(?:ms|s)\s*(?:\))?\s*$",
    re.IGNORECASE,
)


def _strip_js_duration_suffix(name: str) -> str:
    """Remove trailing timing information such as ``(53ms)`` from a test name."""

    return _JS_DURATION_SUFFIX_RE.sub("", name).strip()


def parse_log_js(log: str) -> dict[str, str]:
    """Parse Mocha/Jest output and return {full_test_name: status}.

    Rules:
      * Lines like: "    ✔ should send results to beats" -> PASSED
      * Lines like: "    - test-summary for single suite - teams" -> SKIPPED
      * Lines like: "    [W] 1) should mention group name in slack:" -> FAILED
    """
    results: dict[str, str] = {}

    # Regexes
    passed_re = re.compile(r"^\s*✔\s+(.*?)$")
    skipped_re = re.compile(r"^\s*-\s+(.*?)$")
    failed_re = re.compile(r"^\s*\[W\]\s*\d+\)\s+(.*?)$")
    failed_header_re = re.compile(r"^\s*\d+\)\s+(.*?):$")

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := skipped_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name:
                results[name] = TestStatus.SKIPPED.value
            continue
        if m := failed_header_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name:
                results[name] = TestStatus.FAILED.value
            continue
        if m := failed_re.match(line):
            name = _strip_js_duration_suffix(m.group(1).rstrip())
            if name:
                results[name] = TestStatus.FAILED.value
            continue
        if m := passed_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name:
                results.setdefault(name, TestStatus.PASSED.value)
            continue
    return results


def parse_log_js_2(log: str) -> dict[str, str]:
    """Parse Mocha output and return {full_test_name: status}.

    Rules:
      * Lines like: "  ✔ <name>" -> PASSED (tentative)
      * Lines like: "  1) <name>" -> FAILED (overrides prior PASS)
      * Lines like: "  - <name>" -> SKIPPED (overrides prior PASS)
    """
    results: dict[str, str] = {}

    # Regexes
    passed_re = re.compile(r"^\s*✔\s+(.*?)$")
    failed_re = re.compile(r"^\s*\d+\)\s+(.*?)$")
    skipped_re = re.compile(r"^\s*-\s+(.*?)$")

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := skipped_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name:
                results[name] = TestStatus.SKIPPED.value
            continue
        if m := failed_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name:
                results[name] = TestStatus.FAILED.value
            continue
        if m := passed_re.match(line):
            name = _strip_js_duration_suffix(m.group(1))
            if name:
                results.setdefault(name, TestStatus.PASSED.value)
            continue
    return results


def parse_log_js_3(log: str) -> dict[str, str]:
    """Parse TAP-formatted JS test output into {full_test_name: status}.

    The parser tracks nested suites by concatenating names with ``::`` so that
    leaf test names remain unique even when repeated across suites.
    """
    results: dict[str, str] = {}
    stack: list[str] = []

    tap_line_re = re.compile(
        r"^(?P<status>not ok|ok)\s+\d+\s+-\s+(?P<name>.*?)(?:\s+#.*)?$"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Handle closing braces signalling the end of a nested block.
        if line.replace("}", "") == "":
            for _ in range(line.count("}")):
                if stack:
                    stack.pop()
            continue

        opens_context = line.endswith("{")
        if opens_context:
            line = line[:-1].rstrip()

        match = tap_line_re.match(line)
        if not match:
            continue

        status_word = match.group("status")
        raw_name = match.group("name")
        name, *_ = re.split(r"\s+#", raw_name, maxsplit=1)
        name = _strip_js_duration_suffix(name.strip())
        if not name:
            continue
        skip_marker = any(
            token in raw_name.lower() for token in ("# skip", "# skipped", "# todo")
        )

        full_name = " :: ".join((*stack, name)) if stack else name

        if skip_marker:
            results[full_name] = TestStatus.SKIPPED.value
        else:
            status_value = (
                TestStatus.PASSED.value
                if status_word == "ok"
                else TestStatus.FAILED.value
            )
            if status_value == TestStatus.FAILED.value or full_name not in results:
                results[full_name] = status_value

        if opens_context:
            stack.append(name)

    return results


def parse_log_js_4(log: str) -> dict[str, str]:
    """Parse JS test runner output and return {full_test_name: status}."""
    results: dict[str, str] = {}
    mult_sign = chr(0x00D7)
    pass_symbols = ("✔", "✓")
    fail_symbols = ("✘", "✖", mult_sign)
    skip_symbols = ("○", "◌", "◦", "⚪")
    skip_markers = ("(skipped)", "[skip]", "[skipped]", "[pending]", "[todo]")

    def normalize(name: str) -> str:
        return _strip_js_duration_suffix(name.strip())

    def strip_tag(payload: str) -> str:
        payload = payload.strip()
        if (payload.startswith("[") and "]: " in payload) or (
            payload.startswith("[") and "]:" in payload
        ):
            payload = payload.split("]:", 1)[1].strip()
        if payload.startswith(":"):
            payload = payload[1:].strip()
        return payload

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        symbol = line[0]
        payload = line[1:].strip() if len(line) > 1 else ""

        if symbol in pass_symbols:
            name = normalize(strip_tag(payload))
            if name:
                results.setdefault(name, TestStatus.PASSED.value)
            continue

        if symbol in fail_symbols:
            name = normalize(strip_tag(payload))
            if name:
                results[name] = TestStatus.FAILED.value
            continue

        if symbol in skip_symbols:
            name = normalize(strip_tag(payload))
            if name:
                results[name] = TestStatus.SKIPPED.value
            continue

        lower = line.lower()
        if any(marker in lower for marker in skip_markers):
            candidate = line
            for marker in skip_markers:
                candidate = candidate.replace(marker, "")
            name = normalize(candidate)
            if name:
                results[name] = TestStatus.SKIPPED.value

    return results


def parse_log_gradlew_v1(log: str) -> dict[str, str]:
    """Parse Gradle JUnit XML output and return {full_test_name: status}.

    Parses XML testsuite elements and extracts testcase results.
    Test name format: "testname (classname)" or just "testname" if no class.

    Status mapping:
      * <testcase> with no <failure>, <error>, or <skipped> -> PASSED
      * <testcase> with <skipped> -> SKIPPED
      * <testcase> with <failure> -> FAILED
      * <testcase> with <error> -> ERROR
    """
    results: dict[str, str] = {}

    # Find all XML blocks in the log
    lines = log.split("\n")
    xml_blocks = []
    current_block = []
    in_xml = False

    for line in lines:
        if line.strip().startswith("<?xml"):
            in_xml = True
            current_block = [line]
        elif in_xml:
            current_block.append(line)
            if line.strip().startswith("</testsuite>"):
                xml_blocks.append("\n".join(current_block))
                current_block = []
                in_xml = False

    # Parse each XML block
    for xml_block in xml_blocks:
        try:
            root = ET.fromstring(xml_block)

            # Process each testcase
            for testcase in root.findall(".//testcase"):
                name = testcase.get("name", "")
                classname = testcase.get("classname", "")

                # Format: "testname (classname)" to match common test output format
                full_name = f"{name} ({classname})" if classname and name else name

                # Determine status
                if testcase.find("skipped") is not None:
                    status = TestStatus.SKIPPED.value
                elif testcase.find("failure") is not None:
                    status = TestStatus.FAILED.value
                elif testcase.find("error") is not None:
                    status = TestStatus.ERROR.value
                else:
                    status = TestStatus.PASSED.value

                results[full_name] = status

        except ET.ParseError:
            # Skip malformed XML blocks
            continue

    return results


def parse_log_julia(log: str) -> dict[str, str]:  # noqa: PLR0912, PLR0915
    """Parse Julia Test output and return {test_name: status}.

    Parses the test summary table that appears at the end of Julia test runs.
    Format: "Test Summary: | Pass Fail Error Broken Total Time"

    Rules:
      * If a test has Error count > 0 -> ERROR
      * If a test has Fail count > 0 -> FAILED
      * Otherwise (only Pass or no failures) -> PASSED
      * Priority: ERROR > FAILED > PASSED
      * Tests can be hierarchical (indented with spaces)
    """
    results: dict[str, str] = {}

    # Regex to match test summary lines that look like "  name  | numbers ..."
    test_line_re = re.compile(r"^(\s*)(.+?)\s+\|(.+)$")

    in_summary = False
    has_fail_column = False
    has_error_column = False

    for raw in log.splitlines():
        # Detect start of test summary section and column headers
        if "Test Summary:" in raw:
            in_summary = True
            # Check which columns are present in the header
            if "| Pass" in raw:
                header_part = raw.split("|")[1] if "|" in raw else raw
                has_fail_column = "Fail" in header_part
                has_error_column = "Error" in header_part
            continue

        if not in_summary:
            continue

        # Match test result lines
        if match := test_line_re.match(raw):
            test_name = match.group(2).strip()
            columns_text = match.group(3)

            # Parse columns - they appear in order: Pass, Fail, Error, Broken, Total, Time
            # Split by whitespace and filter out time values
            parts = columns_text.split()

            # Filter out time values (contain 's', 'm', or ':')
            numeric_parts = [p for p in parts if p.isdigit()]

            if len(numeric_parts) < 2:
                continue

            # The last numeric value is Total
            total = int(numeric_parts[-1])

            # Determine status based on what failures are present
            status = TestStatus.PASSED.value

            if len(numeric_parts) == 2:
                # Two numbers: either [Pass, Total] or [Fail/Error, Total]
                first_num = int(numeric_parts[0])

                # Check if Pass column has a value (look for leading spaces pattern)
                # When Pass is missing, there are many leading spaces (typically 10+)
                leading_spaces = len(columns_text) - len(columns_text.lstrip())

                if leading_spaces >= 10:
                    # Pass column is empty, so first number is Error/Fail count
                    # We can't distinguish between Error and Fail in this format
                    # Default to ERROR for this case
                    status = TestStatus.ERROR.value
                elif first_num != total:
                    # Pass is present but doesn't equal Total (shouldn't happen in valid output)
                    status = TestStatus.ERROR.value

            elif len(numeric_parts) == 3:
                # Three numbers: Could be [Pass, Fail, Total] OR [Pass, Error, Total]
                # Depends on which columns are present in the header
                middle_count = int(numeric_parts[1])

                if middle_count > 0:
                    if has_error_column and not has_fail_column:
                        # Header has Error column but no Fail column: [Pass, Error, Total]
                        status = TestStatus.ERROR.value
                    elif has_fail_column and not has_error_column:
                        # Header has Fail column but no Error column: [Pass, Fail, Total]
                        status = TestStatus.FAILED.value
                    else:
                        # Both or neither - default to ERROR (more severe)
                        status = TestStatus.ERROR.value

            elif len(numeric_parts) == 4:
                # Four numbers: [Pass, Fail, Error, Total]
                fail_count = int(numeric_parts[1])
                error_count = int(numeric_parts[2])

                if error_count > 0:
                    status = TestStatus.ERROR.value
                elif fail_count > 0:
                    status = TestStatus.FAILED.value

            elif len(numeric_parts) >= 5:
                # Five or more: [Pass, Fail, Error, Broken, Total, ...]
                fail_count = int(numeric_parts[1])
                error_count = int(numeric_parts[2])

                if error_count > 0:
                    status = TestStatus.ERROR.value
                elif fail_count > 0:
                    status = TestStatus.FAILED.value

            results[test_name] = status

    return results


def parse_log_npx(log: str) -> dict[str, str]:
    """Parse NPX test output (Mocha/Jest style) and return {full_test_name: status}.

    Rules:
      * Lines with ✔ indicate PASSED tests
      * Lines with numbers like "1)" indicate FAILED tests
      * Each individual test line is treated independently based on its marker
    """
    results: dict[str, str] = {}

    # Parse individual test lines
    # Pattern for passed tests: "✔ test name" (with optional timing)
    passed_re = re.compile(r"^\s*✔\s+(.+?)(?:\s+\(\d+(?:ms)?\))?$")
    # Pattern for failed tests: "1) test name" at start of line
    failed_re = re.compile(r"^\s*\d+\)\s+(.+)$")

    for raw in log.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        # Check for passed test
        if m := passed_re.match(line):
            test_name = m.group(1).strip()
            results[test_name] = TestStatus.PASSED.value
            continue

        # Check for failed test markers
        if m := failed_re.match(line):
            test_name = m.group(1).strip()
            # Remove any trailing module name in parentheses
            test_name = re.sub(r"\s+\([^)]+\)$", "", test_name)
            results[test_name] = TestStatus.FAILED.value
            continue

    return results


def parse_log_r(log: str) -> dict[str, str]:
    """Parse R testthat output and return {full_test_name: status}.

    Rules:
      * Parse test context lines like "✔ | 60 | expansion" (PASSED) or "✖ | 2 2 9 | render" (FAILED)
      * Also parse the "── Failed tests ──" section for detailed failure info
      * Lines like: "Failure ('test-file.R:8:5'): test name" -> FAILED
      * Lines like: "Error ('test-file.R:25:5'): test name" -> ERROR
    """
    results: dict[str, str] = {}

    # Regexes for parsing test context summary lines such as "✔ | 60 | expansion"
    passed_context_re = re.compile(r"^✔\s+\|[^|]+\|\s+(.+)$")
    failed_context_re = re.compile(r"^✖\s+\|[^|]+\|\s+(.+)$")

    # Regexes for parsing failed tests section
    failure_re = re.compile(r"^Failure\s+\([^)]+\):\s+(.+)$")
    error_re = re.compile(r"^Error\s+\([^)]+\):\s+(.+)$")

    timing_suffix_re = re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*[a-zA-Z]+\]$")

    def strip_timing_suffix(name: str) -> str:
        return timing_suffix_re.sub("", name).strip()

    # First pass: collect test context statuses (passed/failed contexts)
    for raw in log.splitlines():
        line = raw.strip()

        # Match passed test contexts
        if m := passed_context_re.match(line):
            context_name = strip_timing_suffix(m.group(1).strip())
            results[context_name] = TestStatus.PASSED.value
            continue

        # Match failed test contexts
        if m := failed_context_re.match(line):
            context_name = strip_timing_suffix(m.group(1).strip())
            results[context_name] = TestStatus.FAILED.value
            continue

    # Second pass: collect detailed failure information from failed tests section
    in_failed_section = False
    for raw in log.splitlines():
        line = raw.strip()

        # Detect start of failed tests section
        if line.startswith(("── Failed tests ──", "== Failed tests ==")):
            in_failed_section = True
            continue

        # Detect end of failed section (results summary or end markers)
        if in_failed_section and line.startswith(("[ FAIL", "══ Results ══")):
            break

        if in_failed_section and line:
            # Match failure lines
            if m := failure_re.match(line):
                test_name = strip_timing_suffix(m.group(1).strip())
                results[test_name] = TestStatus.FAILED.value
                continue
            # Match error lines
            if m := error_re.match(line):
                test_name = strip_timing_suffix(m.group(1).strip())
                results[test_name] = TestStatus.ERROR.value
                continue

    return results


def parse_log_r_v2(log: str) -> dict[str, str]:
    """Parse R CMD check logs into {check_name: status}.

    The parser focuses on the trailing summary table produced by R CMD check.
    Each line that starts with "* checking" is treated as a "test" entry.
    """
    results: dict[str, str] = {}
    if not log:
        return results

    line_with_status = re.compile(r"^\* checking (.+?) \.\.\. ([A-Z]+)$")
    line_without_status = re.compile(r"^\* checking (.+?) \.\.\.$")

    status_map: dict[str, str] = {
        "OK": TestStatus.PASSED.value,
        "NOTE": TestStatus.PASSED.value,
        "NOTES": TestStatus.PASSED.value,
        "WARNING": TestStatus.FAILED.value,
        "WARNINGS": TestStatus.FAILED.value,
        "ERROR": TestStatus.ERROR.value,
        "ERRORS": TestStatus.ERROR.value,
        "FAIL": TestStatus.FAILED.value,
        "FAILED": TestStatus.FAILED.value,
        "SKIPPED": TestStatus.SKIPPED.value,
    }

    pending_check: str | None = None
    pending_buffer: list[str] = []

    for raw in log.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue

        if pending_check:
            token = stripped.strip(".:!").split()[0].upper()
            if token in status_map:
                results[pending_check] = status_map[token]
                pending_check = None
                pending_buffer.clear()
            else:
                pending_buffer.append(stripped)
            continue

        if match := line_with_status.match(stripped):
            check_name, token = match.groups()
            token = token.upper()
            results[check_name] = status_map.get(token, TestStatus.PASSED.value)
            continue

        if match := line_without_status.match(stripped):
            pending_check = match.group(1)
            pending_buffer.clear()
            continue

    if pending_check and pending_check not in results:
        results[pending_check] = TestStatus.PASSED.value

    return results


def parse_log_lein(log: str) -> dict[str, str]:  # noqa: PLR0912
    """Parse Leiningen (Clojure) test output into {test_namespace: status}."""
    results: dict[str, str] = {}
    current_namespace: str | None = None
    lein_re = re.compile(r"^lein test (.+)$")

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        if m := lein_re.match(line):
            payload = m.group(1).strip()
            if not payload:
                current_namespace = None
                continue

            tokens = payload.split()
            if tokens and tokens[0] == ":only":
                only_target = " ".join(tokens[1:]).strip()
                if not only_target:
                    current_namespace = None
                    continue
                base = only_target.split("/", 1)[0].strip()
                if base:
                    results.setdefault(base, TestStatus.PASSED.value)
                    current_namespace = base
                else:
                    current_namespace = None
            else:
                for token in tokens:
                    base = token.strip()
                    if base:
                        results.setdefault(base, TestStatus.PASSED.value)
                        current_namespace = base
            continue

        if line.startswith("FAIL in") and current_namespace:
            results[current_namespace] = TestStatus.FAILED.value
            continue
        if line.startswith("ERROR in") and current_namespace:
            results[current_namespace] = TestStatus.ERROR.value
            continue

    return results


def _iter_dart_protocol_events(log: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(decoded, dict):
            events.append(decoded)
        elif isinstance(decoded, list):
            events.extend(item for item in decoded if isinstance(item, dict))
    return events


def _handle_dart_test_start(
    event: dict[str, Any], test_id_to_name: dict[int, str]
) -> None:
    test_info = event.get("test")
    if not isinstance(test_info, dict):
        return

    test_id = test_info.get("id")
    test_name = test_info.get("name")
    if test_id is None or not test_name or test_name.startswith("loading "):
        return

    test_id_to_name[test_id] = test_name


def _dart_done_status(event: dict[str, Any]) -> str | None:
    result = event.get("result")
    if result == "success":
        return TestStatus.PASSED.value
    if result == "failure":
        return TestStatus.FAILED.value
    if result == "error":
        return TestStatus.ERROR.value
    if event.get("skipped", False):
        return TestStatus.SKIPPED.value
    return None


def _handle_dart_test_done(
    event: dict[str, Any], test_id_to_name: dict[int, str], results: dict[str, str]
) -> None:
    test_id = event.get("testID")
    if test_id is None or event.get("hidden", False):
        return
    if test_id not in test_id_to_name:
        return

    status = _dart_done_status(event)
    if status is None:
        return

    test_name = test_id_to_name[test_id]
    results[test_name] = status


def parse_log_dart(log: str) -> dict[str, str]:
    """Parse Dart test output and return {test_name: status}.

    Dart test output is a stream of JSON objects with events.
    - testStart events have test.id and test.name
    - testDone events have testID and result (success, failure, error, or skipped)

    We map test IDs to names, then to their final result status.
    """
    results: dict[str, str] = {}
    test_id_to_name: dict[int, str] = {}

    for event in _iter_dart_protocol_events(log):
        event_type = event.get("type")
        if event_type == "testStart":
            _handle_dart_test_start(event, test_id_to_name)
            continue
        if event_type == "testDone":
            _handle_dart_test_done(event, test_id_to_name, results)

    return results


def parse_log_dart_v2(log: str) -> dict[str, str]:
    """Parse Dart test output and return {full_test_name: status}.

    Rules:
      * Lines like: "[pkg]: HH:MM +N: /path/to/test.dart: Test Name" indicate test execution
      * A test passes if we see the line with an incremented counter (+N -> +N+1)
      * Lines with "loading" are ignored
      * "All tests passed!" indicates successful completion
      * Failed tests would typically show error messages (not present in this sample)
    """
    results: dict[str, str] = {}

    # Regex to match test execution lines
    # Format: [package]: HH:MM +N: /path/to/test.dart: Test Name
    test_line_re = re.compile(
        r"^\[[\w_]+\]:\s+"  # Package name in brackets
        r"\d{2}:\d{2}\s+"  # Time HH:MM
        r"\+(\d+):\s+"  # Counter +N
        r"(/[^:]+\.dart):\s+"  # File path
        r"(.+)$"  # Test name
    )

    # Track test occurrences: {(file_path, test_name): [counter1, counter2, ...]}
    test_occurrences: dict[tuple[str, str], list[int]] = {}

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Skip loading lines
        if "loading" in line.lower():
            continue

        # Match test execution lines
        if m := test_line_re.match(line):
            counter = int(m.group(1))
            file_path = m.group(2)
            test_name = m.group(3)

            # Create a unique key for the test
            test_key = (file_path, test_name)

            if test_key not in test_occurrences:
                test_occurrences[test_key] = []
            test_occurrences[test_key].append(counter)

    # Analyze test occurrences
    for (file_path, test_name), counters in test_occurrences.items():
        # Full test name includes the file path
        full_name = f"{file_path}: {test_name}"

        # A test passes if it appears twice with consecutive counters
        # (once at start, once at completion with incremented counter)
        if len(counters) >= 2:
            # Check if counters are consecutive (indicating pass)
            counters_sorted = sorted(counters)
            if counters_sorted[-1] == counters_sorted[-2] + 1:
                results[full_name] = TestStatus.PASSED.value
            else:
                # Multiple occurrences but not consecutive - uncertain
                results[full_name] = TestStatus.PASSED.value
        elif len(counters) == 1:
            # Only appeared once - might be incomplete or failed
            # In the sample data, all tests appear twice, so single occurrence is unusual
            results[full_name] = TestStatus.ERROR.value

    return results


def parse_log_dart_v3(log: str) -> dict[str, str]:
    """Parse Dart/Flutter test output and return {full_test_name: status}.

    Rules:
      * Lines like: "00:01 +64: /path/to/test.dart: test name" -> PASSED
      * Lines like: "00:02 +88 -1: /path/to/test.dart: test name [E]" -> FAILED
      * Lines like: "00:06 +200 -3: loading /path/to/test.dart [E]" -> ERROR
      * Failure indicators: "[E]" suffix or negative counter (-N)
    """
    results: dict[str, str] = {}

    # Regex patterns for Dart test output
    # Pattern: timestamp +pass_count optional(-fail_count): /path/to/test.dart: test name optional([E])
    test_line_re = re.compile(
        r"^\d+:\d+\s+\+\d+(?:\s+-\d+)?:\s+(/[^:]+\.dart):\s+(.+?)(?:\s+\[E\])?$"
    )
    loading_error_re = re.compile(
        r"^\d+:\d+\s+\+\d+\s+-\d+:\s+loading\s+(/[^:]+\.dart)\s+\[E\]$"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Check for loading/compilation errors
        if m := loading_error_re.match(line):
            test_path = m.group(1)
            # Use file path as test name for loading errors
            results[test_path] = TestStatus.ERROR.value
            continue

        # Check for regular test lines
        if m := test_line_re.match(line):
            test_path = m.group(1)
            test_name = m.group(2).strip()

            # Create full test name
            full_test_name = f"{test_path}: {test_name}"

            # Determine status based on line format
            if "[E]" in line:
                # Explicit error marker
                results[full_test_name] = TestStatus.FAILED.value
            elif " -" in line.split(":")[0]:
                # Has negative counter (failures)
                # Only mark as failed if not already marked
                results.setdefault(full_test_name, TestStatus.FAILED.value)
            else:
                # No failure indicators, treat as passed
                results.setdefault(full_test_name, TestStatus.PASSED.value)
            continue

    return results


def parse_log_scala(log: str) -> dict[str, str]:
    """Parse ScalaTest output and return {test_name: status}.

    Rules:
      * Lines like: "[info] - should <name> (<time>)" -> PASSED
      * Lines like: "[info] - should <name> *** FAILED *** (<time>)" -> FAILED
      * Lines like: "[info] - should <name> !!! CANCELED !!! (<time>)" -> SKIPPED
      * Lines like: "[info] - should <name> !!! IGNORED !!!" -> SKIPPED
    """
    results: dict[str, str] = {}

    # Regex patterns for ScalaTest output
    # Match failed tests: [info] - should test name *** FAILED *** (time)
    failed_re = re.compile(
        r"^\[info\]\s+-\s+(.*?)\s+\*\*\*\s+FAILED\s+\*\*\*\s+\([^)]+\)$"
    )
    # Match canceled tests: [info] - should test name !!! CANCELED !!! (time)
    canceled_re = re.compile(
        r"^\[info\]\s+-\s+(.*?)\s+!!!\s+CANCELED\s+!!!\s+\([^)]+\)$"
    )
    # Match ignored tests: [info] - should test name !!! IGNORED !!!
    ignored_re = re.compile(r"^\[info\]\s+-\s+(.*?)\s+!!!\s+IGNORED\s+!!!")
    # Match passed tests: [info] - should test name (time)
    passed_re = re.compile(r"^\[info\]\s+-\s+(.*?)\s+\([^)]+\)$")

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Check for failed tests first (most specific)
        if m := failed_re.match(line):
            results[m.group(1)] = TestStatus.FAILED.value
            continue

        # Check for canceled tests
        if m := canceled_re.match(line):
            results[m.group(1)] = TestStatus.SKIPPED.value
            continue

        # Check for ignored tests
        if m := ignored_re.match(line):
            results[m.group(1)] = TestStatus.SKIPPED.value
            continue

        # Check for passed tests (least specific, matches any test with timing)
        if m := passed_re.match(line):
            # Only set if not already marked as failed/skipped
            results.setdefault(m.group(1), TestStatus.PASSED.value)
            continue

    return results


def parse_log_scala_v2(log: str) -> dict[str, str]:
    """Parse Scala test output (ANSI format) and return {test_name: status}.

    Rules:
      * Lines like: "[32m  + [0m[32mtest name[0m [90mtime[0m" -> PASSED
      * Lines like: "[31m  x [0m[31mtest name[0m [90mtime[0m" -> FAILED (red color code)
      * Summary line provides overall counts but individual test status from markers

    The format uses ANSI color codes:
      - [32m = green (passed)
      - [31m = red (failed)
      - [90m = gray (timing)
      - [0m = reset
    """
    results: dict[str, str] = {}

    # Strip ANSI color codes for easier parsing
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m|\[(?:[0-9]{1,2}m)")

    # Regex patterns for test output
    # Match passed tests: "  +  test name  time"
    passed_re = re.compile(r"^\s*\+\s+(.*?)\s+[\d.]+[mμn]?s\s*$")
    # Match failed tests: "  x  test name  time"
    failed_re = re.compile(r"^\s*x\s+(.*?)\s+[\d.]+[mμn]?s\s*$")
    # Alternative: match tests with color codes still in place
    passed_color_re = re.compile(
        r"^\s*\+\s+\[0m\[32m(.*?)\[0m\s+\[90m[\d.]+[mμn]?s\[0m\s*$"
    )
    failed_color_re = re.compile(
        r"^\s*x\s+\[0m\[31m(.*?)\[0m\s+\[90m[\d.]+[mμn]?s\[0m\s*$"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Strip ANSI codes for cleaner matching
        clean_line = ansi_escape.sub("", line)

        # Check for failed tests (with or without color codes)
        if m := failed_color_re.match(line):
            results[m.group(1)] = TestStatus.FAILED.value
            continue
        if m := failed_re.match(clean_line):
            results[m.group(1)] = TestStatus.FAILED.value
            continue

        # Check for passed tests (with or without color codes)
        if m := passed_color_re.match(line):
            results[m.group(1)] = TestStatus.PASSED.value
            continue
        if m := passed_re.match(clean_line):
            results[m.group(1)] = TestStatus.PASSED.value
            continue

    return results


def parse_log_scala_v3(log: str) -> dict[str, str]:
    """Parse Scala test output and return {full_test_name: status}.

    Handles ScalaTest "[info] - ..." lines and recognizes failure markers.
    Combines suite names with test names (e.g., "recover: should return the value").
    """
    results: dict[str, str] = {}

    status_map = {
        "PASSED": TestStatus.PASSED.value,
        "FAILED": TestStatus.FAILED.value,
        "ERROR": TestStatus.ERROR.value,
        "ABORTED": TestStatus.ERROR.value,
        "SKIPPED": TestStatus.SKIPPED.value,
        "IGNORED": TestStatus.SKIPPED.value,
        "CANCELED": TestStatus.SKIPPED.value,
        "CANCELLED": TestStatus.SKIPPED.value,
        "PENDING": TestStatus.SKIPPED.value,
    }

    test_line_re = re.compile(r"^\[info\]\s+-\s+(.*?)(?:\s+\*{3}\s+([A-Z]+)\s+\*{3})?$")
    suite_line_re = re.compile(r"^\[info\]\s+([^\-].*)$")

    current_suite = None

    for raw in log.splitlines():
        line = raw.strip()
        if not line.startswith("[info]"):
            continue

        # Check if this is a test line (starts with "- ")
        if line.startswith("[info] -"):
            if match := test_line_re.match(line):
                test_name = match.group(1).strip()
                # Remove timing information in parentheses at the end
                test_name = re.sub(r"\s*\(\d+\s+\w+\)$", "", test_name)

                status_token = match.group(2)

                # Combine suite name with test name if suite exists
                full_test_name = (
                    f"{current_suite}: {test_name}" if current_suite else test_name
                )

                if status_token:
                    normalized = status_token.upper()
                    status = status_map.get(normalized, normalized)
                    results[full_test_name] = status
                else:
                    results.setdefault(full_test_name, TestStatus.PASSED.value)
        # This might be a suite name line
        elif match := suite_line_re.match(line):
            potential_suite = match.group(1).strip()
            # Filter out non-suite lines (like "Run completed", "Total number", etc.)
            if potential_suite and not any(
                skip in potential_suite
                for skip in [
                    "Run completed",
                    "Total number",
                    "Suites:",
                    "Tests:",
                    "All tests",
                    "compiling",
                    "done compiling",
                    "loading",
                    "welcome to",
                    "set current",
                ]
            ):
                current_suite = potential_suite

    return results


def parse_log_ocaml(log: str) -> dict[str, str]:
    """Parse Alcotest/QCheck output and return {full_test_name: status}."""
    results: dict[str, str] = {}
    status_map: dict[str, str] = {
        "OK": TestStatus.PASSED.value,
        "PASS": TestStatus.PASSED.value,
        "FAIL": TestStatus.FAILED.value,
        "FAILED": TestStatus.FAILED.value,
        "ERROR": TestStatus.ERROR.value,
        "ERR": TestStatus.ERROR.value,
        "SKIP": TestStatus.SKIPPED.value,
        "SKIPPED": TestStatus.SKIPPED.value,
        "TODO": TestStatus.SKIPPED.value,
    }
    entry_re = re.compile(
        r"^\s*\[(?P<status>[A-Z]+)\]\s+(?P<suite>.*?)\s+(?P<index>\d+)\s+(?P<name>.+)$"
    )
    # Recognize Alcotest/QCheck status lines like "[OK] suite idx description".
    for raw in log.splitlines():
        line = raw.rstrip()
        match = entry_re.match(line)
        if not match:
            continue
        status_token = match.group("status")
        suite = match.group("suite").strip()
        index = match.group("index").strip()
        name = match.group("name").strip()
        test_name = " ".join(part for part in (suite, index, name) if part)
        mapped_status = status_map.get(status_token, TestStatus.ERROR.value)
        if mapped_status == TestStatus.PASSED.value:
            results.setdefault(test_name, mapped_status)
        else:
            results[test_name] = mapped_status
    return results


class _OcamlDuneLogParser:
    running_re = re.compile(r"^Running\[(?P<id>\d+)\]:\s+\((?P<command>.+)\)$")
    output_re = re.compile(r"^Output\[(?P<id>\d+)]:")
    fail_tokens = ("FAIL", "ERROR", "EXCEPTION", "CRASH", "FATAL")
    skip_tokens = ("SKIP", "SKIPPED")

    def __init__(self) -> None:
        self.results: dict[str, str] = {}
        self.run_lookup: dict[int, str] = {}
        self.active_test: str | None = None
        self.seen_output_tests: set[str] = set()

    @staticmethod
    def _extract_test_name(command_segment: str) -> str:
        command = command_segment.split("&&")[-1].strip()
        if command.startswith("exec "):
            command = command[5:].strip()
        if command.startswith("./"):
            command = command[2:]
        parts = command.split()
        binary = parts[0] if parts else ""
        return binary.rsplit("/", 1)[-1] if binary else ""

    def _update_status(self, name: str, status: str) -> None:
        _update_status_by_precedence(self.results, name, status)

    def handle_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return
        if match := self.running_re.match(line):
            run_id = int(match.group("id"))
            test_name = self._extract_test_name(match.group("command"))
            self.run_lookup[run_id] = test_name or f"run_{run_id}"
            self.active_test = None
            return
        if match := self.output_re.match(line):
            self.active_test = self.run_lookup.get(int(match.group("id")))
            return
        if not self.active_test:
            return
        self.seen_output_tests.add(self.active_test)
        upper = line.upper()
        if any(token in upper for token in self.fail_tokens):
            self._update_status(self.active_test, TestStatus.FAILED.value)
        elif any(token in upper for token in self.skip_tokens):
            self._update_status(self.active_test, TestStatus.SKIPPED.value)
        elif upper == "OK":
            self._update_status(self.active_test, TestStatus.PASSED.value)

    def finalize(self) -> dict[str, str]:
        for test_name in self.seen_output_tests:
            self.results.setdefault(test_name, TestStatus.PASSED.value)
        return self.results


def parse_log_ocaml_v2(log: str) -> dict[str, str]:
    """Parse dune test logs and return {test_binary: status}."""
    parser = _OcamlDuneLogParser()
    for raw_line in log.splitlines():
        parser.handle_line(raw_line)
    return parser.finalize()


def parse_log_ocaml_v3(log: str) -> dict[str, str]:
    """Parse inline OCaml test output and return {test_case: status}.

    Rules:
      * Lines starting with "[OK]" (or similar pass tokens) mark the case as PASSED.
      * Lines starting with "[SKIP]" (or aliases) mark the case as SKIPPED unless it failed later.
      * Lines starting with failure tokens ("[FAIL]", "[ERROR]", "[CRASH]", "[KO]", etc.) mark the case as FAILED and override previous entries.
    """
    results: dict[str, str] = {}
    status_tokens: dict[str, str] = {
        "OK": TestStatus.PASSED.value,
        "PASS": TestStatus.PASSED.value,
        "PASSED": TestStatus.PASSED.value,
        "SUCCESS": TestStatus.PASSED.value,
        "SKIP": TestStatus.SKIPPED.value,
        "SKIPPED": TestStatus.SKIPPED.value,
        "TODO": TestStatus.SKIPPED.value,
        "PENDING": TestStatus.SKIPPED.value,
        "DISABLED": TestStatus.SKIPPED.value,
        "FAIL": TestStatus.FAILED.value,
        "FAILED": TestStatus.FAILED.value,
        "ERROR": TestStatus.FAILED.value,
        "EXCEPTION": TestStatus.FAILED.value,
        "CRASH": TestStatus.FAILED.value,
        "FATAL": TestStatus.FAILED.value,
        "KO": TestStatus.FAILED.value,
        "PANIC": TestStatus.FAILED.value,
    }
    status_line_re = re.compile(
        r"^\[(?P<token>[A-Za-z][A-Za-z0-9_-]*)\]\s+(?P<body>.+)$"
    )

    for raw_line in log.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("["):
            continue
        match = status_line_re.match(line)
        if not match:
            continue
        token = match.group("token").upper()
        status = status_tokens.get(token)
        if status is None:
            continue
        body = match.group("body")
        test_name = re.sub(r"\s+", " ", body).strip()
        if not test_name:
            continue
        _update_status_by_precedence(results, test_name, status)

    return results


def parse_log_ocaml_v4(log: str) -> dict[str, str]:
    """Parse ocamlbuild logs and return {test_name: status}."""
    results: dict[str, str] = {}

    for raw in log.splitlines():
        line = ANSI_COLOR_DELIM_RE.sub(" ", raw)
        line = ANSI_ESCAPE_RE.sub("", line).strip()
        if not line:
            continue
        if not (prefix_match := OCAML_STATUS_PREFIX_RE.match(line)):
            continue
        status_token, remainder = prefix_match.groups()
        remainder = remainder.strip()
        if not remainder:
            continue
        parts = re.split(r"\s{2,}", remainder, maxsplit=1)
        name = parts[0].strip() if parts else ""
        if not name and remainder:
            tokens = remainder.split()
            if tokens:
                name = tokens[0]
        if not name:
            continue
        try:
            normalized = TestStatus[status_token].value
        except KeyError:
            continue
        _update_status_by_precedence(results, name, normalized)
    return results


def parse_logs_r_junit(log: str) -> dict[str, str]:
    """Parse R testthat JUnit XML output and return {full_test_name: status}.

    Parses JUnit XML format from testthat::test_local(..., reporter = 'junit').
    Returns a dict mapping test names to their status (PASSED, FAILED, SKIPPED, ERROR).
    """
    results: dict[str, str] = {}

    try:
        # Find the XML content in the log
        xml_start = log.find("<?xml")
        if xml_start == -1:
            return results

        xml_content = log[xml_start:]
        # Try to find the end of XML (closing </testsuites>)
        xml_end = xml_content.find("</testsuites>")
        if xml_end != -1:
            xml_content = xml_content[: xml_end + len("</testsuites>")]

        root = ET.fromstring(xml_content)

        # Iterate through all testsuites and testcases
        for testsuite in root.findall(".//testsuite"):
            for testcase in testsuite.findall("testcase"):
                test_name = testcase.get("name", "")
                classname = testcase.get("classname", "")

                # Create full test name (classname::test_name)
                full_name = f"{classname}::{test_name}" if classname else test_name

                # Check for failure, error, or skipped elements
                if testcase.find("failure") is not None:
                    results[full_name] = TestStatus.FAILED.value
                elif testcase.find("error") is not None:
                    results[full_name] = TestStatus.ERROR.value
                elif testcase.find("skipped") is not None:
                    results[full_name] = TestStatus.SKIPPED.value
                else:
                    # No failure/error/skipped element means the test passed
                    results[full_name] = TestStatus.PASSED.value

    except ET.ParseError:
        # If XML parsing fails, return empty results
        pass

    return results


def parse_log_swift(log: str) -> dict[str, str]:
    """Parse Swift XCTest output and return {full_test_name: status}.

    Rules:
      * Lines like: "Test Case 'ClassName.testName' passed (X.X seconds)" -> PASSED
      * Lines like: "Test Case 'ClassName.testName' failed (X.X seconds)" -> FAILED
      * Extract test name from the quoted portion
    """
    results: dict[str, str] = {}

    # Regex to match test completion lines
    test_result_re = re.compile(
        r"^Test Case '([^']+)'\s+(passed|failed)\s+\([0-9.]+\s+seconds\)$"
    )

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue

        if m := test_result_re.match(line):
            test_name = m.group(1)
            status = m.group(2).upper()

            if status == "PASSED":
                results[test_name] = TestStatus.PASSED.value
            elif status == "FAILED":
                results[test_name] = TestStatus.FAILED.value

    return results


def parse_log_csharp(log: str) -> dict[str, str]:
    """Parse xUnit.net output and return {full_test_name: status}.

    Rules:
      * Lines like: "  Passed <test_name> [<time>]" -> PASSED
      * Lines like: "  Failed <test_name> [<time>]" -> FAILED
      * Lines like: "  Skipped <test_name>" -> SKIPPED
      * Lines like: "[xUnit.net timestamp]     <test_name> [FAIL]" -> FAILED (overrides prior status)
    """
    results: dict[str, str] = {}

    # Regexes for xUnit.net output
    passed_re = re.compile(r"^\s+Passed\s+(.+?)\s+\[.+?\]$")
    failed_re = re.compile(r"^\s+Failed\s+(.+?)\s+\[.+?\]$")
    skipped_re = re.compile(r"^\s+Skipped\s+(.+?)(?:\s+\[.+?\])?$")
    # Alternative failure format: [xUnit.net timestamp]     TestName [FAIL]
    xunit_fail_re = re.compile(r"^\[xUnit\.net\s+[\d:\.]+\]\s+(.+?)\s+\[FAIL\]$")

    for raw in log.splitlines():
        line = raw.rstrip()
        if not line:
            continue

        if m := xunit_fail_re.match(line):
            # This format indicates a failure, overrides any previous status
            results[m.group(1)] = TestStatus.FAILED.value
            continue

        if m := failed_re.match(line):
            results[m.group(1)] = TestStatus.FAILED.value
            continue

        if m := skipped_re.match(line):
            results[m.group(1)] = TestStatus.SKIPPED.value
            continue

        if m := passed_re.match(line):
            # Only set if not already marked as failed
            results.setdefault(m.group(1), TestStatus.PASSED.value)
            continue

    return results


parse_log_astroid = parse_log_pytest
parse_log_flask = parse_log_pytest
parse_log_marshmallow = parse_log_pytest
parse_log_pvlib = parse_log_pytest
parse_log_pyvista = parse_log_pytest
parse_log_sqlfluff = parse_log_pytest
parse_log_xarray = parse_log_pytest

parse_log_pydicom = parse_log_pytest_options
parse_log_requests = parse_log_pytest_options
parse_log_pylint = parse_log_pytest_options

parse_log_astropy = parse_log_pytest_v2
parse_log_scikit = parse_log_pytest_v2
parse_log_sphinx = parse_log_pytest_v2

parse_log_pydantic = parse_log_pytest
parse_log_dvc = parse_log_pytest
parse_lua_nvim = parse_lue_nvim
parse_log_lua_nvim = parse_lue_nvim
parse_log_java_mvn = parse_java_mvn
parse_log_java_mvn_v2 = parse_java_mvn_v2
parse_log_r_junit = parse_logs_r_junit
parse_log_kotlin_junit = parse_logs_kotlin_junit
parse_log_ocaml_v5 = parse_log_ocaml_v4

MAP_REPO_TO_PARSER = {
    "astropy/astropy": parse_log_astropy,
    "django/django": parse_log_django,
    "marshmallow-code/marshmallow": parse_log_marshmallow,
    "matplotlib/matplotlib": parse_log_matplotlib,
    "mwaskom/seaborn": parse_log_seaborn,
    "pallets/flask": parse_log_flask,
    "psf/requests": parse_log_requests,
    "pvlib/pvlib-python": parse_log_pvlib,
    "pydata/xarray": parse_log_xarray,
    "pydicom/pydicom": parse_log_pydicom,
    "pylint-dev/astroid": parse_log_astroid,
    "pylint-dev/pylint": parse_log_pylint,
    "pytest-dev/pytest": parse_log_pytest,
    "pyvista/pyvista": parse_log_pyvista,
    "scikit-learn/scikit-learn": parse_log_scikit,
    "sqlfluff/sqlfluff": parse_log_sqlfluff,
    "sphinx-doc/sphinx": parse_log_sphinx,
    "sympy/sympy": parse_log_sympy,
    "pydantic/pydantic": parse_log_pydantic,
    "iterative/dvc": parse_log_dvc,
}

MAP_REPO_TO_PARSER = defaultdict(lambda: parse_log_pytest, MAP_REPO_TO_PARSER)

NAME_TO_PARSER = {
    "parse_log_pytest": parse_log_pytest,
    "parse_log_pytest_options": parse_log_pytest_options,
    "parse_log_pytest_v2": parse_log_pytest_v2,
    "parse_log_pytest_nebo": parse_log_pytest_nebo,
    "parse_combined_test_reports": parse_combined_test_reports,
    "parse_log_gotest": parse_log_gotest,
    "parse_log_redis": parse_log_redis,
    "parse_log_jq": parse_log_jq,
    "parse_log_doctest": parse_log_doctest,
    "parse_log_micropython_test": parse_log_micropython_test,
    "parse_log_googletest": parse_log_googletest,
    "parse_log_minitest": parse_log_minitest,
    "parse_log_cucumber": parse_log_cucumber,
    "parse_log_ruby_unit": parse_log_ruby_unit,
    "parse_log_rspec_transformed_json": parse_log_rspec_transformed_json,
    "parse_log_cargo": parse_log_cargo,
    "parse_log_phpunit": parse_log_phpunit,
    "parse_log_maven": parse_log_maven,
    "parse_log_ant": parse_log_ant,
    "parse_log_gradle_custom": parse_log_gradle_custom,
    "parse_log_calypso": parse_log_calypso,
    "parse_log_chart_js": parse_log_chart_js,
    "parse_log_marked": parse_log_marked,
    "parse_log_p5js": parse_log_p5js,
    "parse_log_react_pdf": parse_log_react_pdf,
    "parse_log_jest": parse_log_jest,
    "parse_log_jest_json": parse_log_jest_json,
    "parse_log_vitest": parse_log_vitest,
    "parse_log_karma": parse_log_karma,
    "parse_log_tap": parse_log_tap,
    "parse_log_elixir": parse_log_elixir,
    "parse_log_ruby_v1": parse_log_ruby_v1,
    "parse_log_cpp": parse_log_cpp,
    "parse_log_cpp_v2": parse_log_cpp_v2,
    "parse_log_cpp_v3": parse_log_cpp_v3,
    "parse_log_cpp_v4": parse_log_cpp_v4,
    "parse_lue_nvim": parse_lue_nvim,
    "parse_lua_nvim": parse_lua_nvim,
    "parse_log_lua_nvim": parse_log_lua_nvim,
    "parse_java_mvn": parse_java_mvn,
    "parse_log_java_mvn": parse_log_java_mvn,
    "parse_java_mvn_v2": parse_java_mvn_v2,
    "parse_log_java_mvn_v2": parse_log_java_mvn_v2,
    "parse_log_php_v1": parse_log_php_v1,
    "parse_log_ruby_v2": parse_log_ruby_v2,
    "parse_log_haskell": parse_log_haskell,
    "parse_log_haskell_v2": parse_log_haskell_v2,
    "parse_log_js": parse_log_js,
    "parse_log_js_2": parse_log_js_2,
    "parse_log_js_3": parse_log_js_3,
    "parse_log_js_4": parse_log_js_4,
    "parse_log_gradlew_v1": parse_log_gradlew_v1,
    "parse_log_julia": parse_log_julia,
    "parse_log_npx": parse_log_npx,
    "parse_log_r": parse_log_r,
    "parse_log_r_v2": parse_log_r_v2,
    "parse_log_lein": parse_log_lein,
    "parse_log_dart": parse_log_dart,
    "parse_log_dart_v2": parse_log_dart_v2,
    "parse_log_dart_v3": parse_log_dart_v3,
    "parse_log_scala": parse_log_scala,
    "parse_log_scala_v2": parse_log_scala_v2,
    "parse_log_scala_v3": parse_log_scala_v3,
    "parse_log_ocaml": parse_log_ocaml,
    "parse_log_ocaml_v2": parse_log_ocaml_v2,
    "parse_log_ocaml_v3": parse_log_ocaml_v3,
    "parse_log_ocaml_v4": parse_log_ocaml_v4,
    "parse_log_ocaml_v5": parse_log_ocaml_v5,
    "parse_log_sbt": parse_log_sbt,
    "parse_log_junit": parse_log_junit,
    "parse_logs_r_junit": parse_logs_r_junit,
    "parse_logs_kotlin_junit": parse_logs_kotlin_junit,
    "parse_log_r_junit": parse_log_r_junit,
    "parse_log_kotlin_junit": parse_log_kotlin_junit,
    "parse_log_swift": parse_log_swift,
    "parse_log_csharp": parse_log_csharp,
}
