"""Provider-side network restrictions, checked before a request leaves the gateway.

Inspection never changes a native request. A restriction that cannot be enforced by
its existing provider settings rejects the exchange instead of changing the prompt.
"""

import re
from glob import has_magic
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, ValidationError

from verifiers.v1.configs.runtime import (
    NetworkPolicyConfig,
    intersect_network_hosts,
    parse_network_rule,
)
from verifiers.v1.errors import RolloutError

PROVIDER_CAPABILITY_POLICY_CODE = "provider_capability_unavailable"


class ProviderPolicyError(RolloutError):
    status_code = 400

    def __init__(self, paths: list[str]):
        self.paths = list(dict.fromkeys(paths))
        super().__init__(
            "Request is incompatible with the network policy: " + ", ".join(self.paths)
        )


class RequestPolicy:
    def __init__(self, policy: NetworkPolicyConfig):
        self.policy = policy
        self.violations: list[str] = []

    def blocked_url(self, value: object) -> bool:
        if not isinstance(value, str):
            return True
        if value.lower().startswith("data:"):
            return False
        try:
            url = AnyHttpUrl(value)
        except ValidationError:
            return True
        return not self.policy.permits(url.scheme, url.host.strip("[]"), url.port)

    def blocked(self, value, path: str) -> str | None:
        if isinstance(value, list):
            return next(
                (
                    blocked
                    for i, item in enumerate(value)
                    if (blocked := self.blocked(item, f"{path}[{i}]"))
                ),
                None,
            )
        if not isinstance(value, dict):
            return None
        caller = value.get("caller")
        if caller is not None and not (
            isinstance(caller, dict) and caller.get("type") == "direct"
        ):
            return f"{path}.caller.type"
        return self.blocked_part(value, path)

    def content(self, value, path: str) -> None:
        if blocked := self.blocked(value, path):
            self.violations.append(blocked)

    def tools(self, value, path: str = "tools") -> list[dict]:
        if value is not None and not isinstance(value, list):
            self.violations.append(path)
            return []
        accepted = []
        for index, tool in enumerate(value or []):
            item_path = f"{path}[{index}]"
            if self.tool(tool, item_path):
                accepted.append(tool)
            else:
                self.violations.append(item_path)
        return accepted

    def tool(self, value, path: str) -> bool:
        raise NotImplementedError

    def blocked_part(self, value: dict, path: str) -> str | None:
        raise NotImplementedError

    def inspect(self, body: dict) -> None:
        raise NotImplementedError

    def choice(self, choice, tools: list[dict], path: str = "tool_choice") -> None:
        if not isinstance(choice, dict):
            return
        kind = choice.get("type")
        if kind == "allowed_tools":
            nested = choice.get("allowed_tools", choice)
            if not isinstance(nested, dict) or not isinstance(
                nested.get("tools"), list
            ):
                self.violations.append(path)
                return
            for index, item in enumerate(nested["tools"]):
                self.choice(item, tools, f"{path}.tools[{index}]")
        elif kind not in (
            "function",
            "custom",
            "auto",
            "any",
            "none",
            "tool",
        ) and not any(t.get("type") == kind for t in tools):
            self.violations.append(path)


def provider_domains(
    policy: NetworkPolicyConfig, requested: object = None
) -> list[str]:
    """Translate allow/block rules to provider filters without changing their scope.

    Provider filters include subdomains, so exact hosts need a covering wildcard rule.
    Empty results mean the policy cannot be represented; never send an empty filter.
    """
    rules = policy.block or policy.allow
    if requested is not None and not isinstance(requested, list):
        return []
    hosts, requested_domains = [], []
    for entries, output, is_filter in (
        (rules, hosts, False),
        (requested or [], requested_domains, True),
    ):
        for rule in entries:
            if not isinstance(rule, str):
                return []
            try:
                url, host, port = parse_network_rule(rule)
            except ValueError:
                return []
            if is_filter and (
                url.username is not None or url.path or url.query or url.fragment
            ):
                return []
            domain = host if is_filter else host.removeprefix("*.")
            if (
                url.scheme
                or port is not None
                or not domain
                or has_magic(domain)
                or not domain.isascii()
            ):
                return []
            output.append(host)
    for host in hosts:
        if not host.startswith("*.") and not (
            policy.block
            and any(
                wildcard.startswith("*.")
                and intersect_network_hosts(wildcard, host) == host
                for wildcard in hosts
            )
        ):
            return []
    domains = list(dict.fromkeys(host.removeprefix("*.") for host in hosts))
    if requested is None:
        return domains
    if policy.block:
        return list(dict.fromkeys([*domains, *requested_domains]))
    intersection = []
    for allowed in domains:
        for requested_domain in requested_domains:
            if host := intersect_network_hosts(f"*.{allowed}", f"*.{requested_domain}"):
                intersection.append(host.removeprefix("*."))
    return list(dict.fromkeys(intersection))


CHAT_CLIENT_TOOL_TYPES = ("function", "custom")


CHAT_SAFE_CONTENT_TYPES = ("text", "refusal", "input_audio", "image_url", "file")


class ChatPolicy(RequestPolicy):
    def tool(self, tool, path: str) -> bool:
        return (
            isinstance(tool, dict)
            and tool.get("type", "function") in CHAT_CLIENT_TOOL_TYPES
        )

    def blocked(self, value, path: str) -> str | None:
        # Chat content parts are flat; nested lists and non-object parts are invalid.
        kind = value.get("type") if isinstance(value, dict) else None
        if kind not in CHAT_SAFE_CONTENT_TYPES:
            return f"{path}.type"
        if kind == "image_url":
            image = value.get("image_url") or {}
            url = image.get("url") if isinstance(image, dict) else image
            if self.blocked_url(url):
                return f"{path}.image_url.url"
        if kind == "file":
            file = value.get("file")
            if not isinstance(file, dict):
                return f"{path}.file"
            if file.get("file_id"):
                return f"{path}.file.file_id"
            data = file.get("file_data")
            if data is None:
                return None
            if not isinstance(data, str):
                return f"{path}.file.file_data"
            try:
                parsed = urlsplit(data)
            except ValueError:
                return f"{path}.file.file_data"
            if (parsed.scheme or parsed.netloc) and self.blocked_url(data):
                return f"{path}.file.file_data"
        return None

    def inspect(self, body: dict) -> None:
        for key in ("web_search_options", "plugins"):
            if body.get(key) is not None:
                self.violations.append(key)
        audio = body.get("audio")
        if (
            isinstance(audio, dict)
            and isinstance(audio.get("voice"), dict)
            and audio["voice"].get("id")
        ):
            self.violations.append("audio.voice.id")
        tools = self.tools(body.get("tools"))
        self.choice(body.get("tool_choice"), tools)
        for index, message in enumerate(body.get("messages") or []):
            if not isinstance(message, dict):
                continue
            if isinstance(message.get("audio"), dict) and message["audio"].get("id"):
                self.violations.append(f"messages[{index}].audio.id")
            content = message.get("content")
            if isinstance(content, list):
                for part_index, part in enumerate(content):
                    self.content(part, f"messages[{index}].content[{part_index}]")


RESPONSES_CLIENT_TOOL_TYPES = (
    "function",
    "custom",
    "local_shell",
    "apply_patch",
    "computer",
    "computer_use_preview",
)


RESPONSES_WEB_SEARCH_TOOL_TYPE = re.compile(
    r"web_search(?:_\d{4}_\d{2}_\d{2})?"
).fullmatch


RESPONSES_SAFE_INPUT_TYPES = (
    "input_text",
    "input_file",
    "input_image",
    "computer_screenshot",
    "output_text",
    "refusal",
    "computer_call",
    "function_call",
    "custom_tool_call",
    "reasoning",
    "compaction",
    "tool_search_call",
    "local_shell_call",
    "local_shell_call_output",
    "shell_call",
    "shell_call_output",
    "apply_patch_call",
    "apply_patch_call_output",
    "compaction_trigger",
)


RESPONSES_TEXT_TOOL_OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")


class ResponsesPolicy(RequestPolicy):
    def tool(self, tool, path: str) -> bool:
        if not isinstance(tool, dict):
            return False
        kind = tool.get("type")
        if isinstance(kind, str) and RESPONSES_WEB_SEARCH_TOOL_TYPE(kind):
            key = "blocked_domains" if self.policy.block else "allowed_domains"
            filters = tool.get("filters")
            if not isinstance(filters, dict):
                return False
            requested = filters.get(key)
            domains = provider_domains(self.policy, requested)
            return (
                bool(domains)
                and len(domains) <= 100
                and isinstance(requested, list)
                and set(domains) == set(requested)
            )
        if kind == "namespace":
            before = len(self.violations)
            self.tools(tool.get("tools"), f"{path}.tools")
            return len(self.violations) == before
        environment = tool.get("environment")
        return (
            kind in RESPONSES_CLIENT_TOOL_TYPES
            or kind == "tool_search"
            and tool.get("execution") == "client"
            or kind == "shell"
            and isinstance(environment, dict)
            and environment.get("type") == "local"
        )

    def blocked_part(self, value: dict, path: str) -> str | None:
        kind = value.get("type")
        if kind in ("input_file", "input_image", "computer_screenshot"):
            if value.get("file_id"):
                return f"{path}.file_id"
            url_field = "file_url" if kind == "input_file" else "image_url"
            if kind != "input_file" or url_field in value:
                if self.blocked_url(value.get(url_field)):
                    return f"{path}.{url_field}"
            elif not isinstance(value.get("file_data"), str):
                return f"{path}.file_data"

        if (
            kind == "reasoning"
            and value.get("id")
            and not value.get("encrypted_content")
        ):
            return f"{path}.id"
        if kind == "item_reference" or kind is None and set(value) == {"id"}:
            return f"{path}.id"

        if kind == "tool_search_call" and value.get("execution") != "client":
            return f"{path}.execution"
        if kind == "shell_call":
            environment = value.get("environment")
            if not (
                isinstance(environment, dict) and environment.get("type") == "local"
            ):
                return f"{path}.environment"
        if kind in ("additional_tools", "tool_search_output"):
            if kind == "tool_search_output" and value.get("execution") != "client":
                return f"{path}.execution"
            # Nested tool lists are atomic: report their first violation.
            probe = ResponsesPolicy(self.policy)
            probe.tools(value.get("tools"), f"{path}.tools")
            return next(iter(probe.violations), None)

        if kind in ("computer_call_output", *RESPONSES_TEXT_TOOL_OUTPUT_TYPES):
            return self.blocked(value.get("output"), f"{path}.output")
        if kind in (None, "message") and "role" in value and "content" in value:
            return self.blocked(value["content"], f"{path}.content")
        return None if kind in RESPONSES_SAFE_INPUT_TYPES else f"{path}.type"

    def inspect(self, body: dict) -> None:
        for key in ("previous_response_id", "conversation", "prompt", "plugins"):
            if body.get(key) is not None:
                self.violations.append(key)
        self.content(body.get("input"), "input")
        tools = self.tools(body.get("tools"))
        self.choice(body.get("tool_choice"), tools)


ANTHROPIC_CLIENT_TOOL_TYPE = re.compile(
    r"(?:bash|text_editor|computer|memory)_\d{8}"
).fullmatch


ANTHROPIC_WEB_TOOL_TYPE = re.compile(r"web_(?:search|fetch)_\d{8}").fullmatch


ANTHROPIC_CONTENT_WRAPPERS = (
    "tool_result",
    "code_execution_tool_result",
    "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result",
    "web_search_tool_result",
    "web_fetch_tool_result",
    "tool_search_tool_result",
    "mcp_tool_result",
    "advisor_tool_result",
    "code_execution_result",
    "bash_code_execution_result",
    "encrypted_code_execution_result",
    "web_fetch_result",
)


ANTHROPIC_SAFE_CONTENT_TYPES = (
    "text",
    "image",
    "document",
    "tool_reference",
    "thinking",
    "redacted_thinking",
    "tool_use",
    "search_result",
    "server_tool_use",
    "mid_conv_system",
    "compaction",
    "fallback",
    "mcp_tool_use",
    "web_search_result",
    "web_search_tool_result_error",
    "web_fetch_tool_result_error",
    "tool_search_tool_search_result",
    "tool_search_tool_result_error",
    "code_execution_tool_result_error",
    "bash_code_execution_tool_result_error",
    "text_editor_code_execution_tool_result_error",
    "text_editor_code_execution_create_result",
    "text_editor_code_execution_str_replace_result",
    "text_editor_code_execution_view_result",
    "advisor_result",
    "advisor_redacted_result",
    "advisor_tool_result_error",
)


class AnthropicPolicy(RequestPolicy):
    wrappers = ANTHROPIC_CONTENT_WRAPPERS

    def tool(self, tool, path: str) -> bool:
        if not isinstance(tool, dict):
            return False
        kind = tool.get("type")
        if isinstance(kind, str) and ANTHROPIC_WEB_TOOL_TYPE(kind):
            key = "blocked_domains" if self.policy.block else "allowed_domains"
            other = "allowed_domains" if self.policy.block else "blocked_domains"
            requested = tool.get(key)
            domains = provider_domains(self.policy, requested)
            return (
                tool.get(other) in (None, [])
                and tool.get("allowed_callers", ["direct"]) == ["direct"]
                and isinstance(requested, list)
                and bool(domains)
                and set(domains) == set(requested)
            )
        return (
            isinstance(kind, str)
            and bool(ANTHROPIC_CLIENT_TOOL_TYPE(kind))
            or kind in (None, "custom")
            and "input_schema" in tool
        )

    def blocked_part(self, value: dict, path: str) -> str | None:
        kind = value.get("type")
        if kind in ("image", "document"):
            source_path = f"{path}.source"
            source = value.get("source") or {}
            if not isinstance(source, dict):
                return source_path
            source_kind = source.get("type")
            if source_kind == "content":
                return self.blocked(source.get("content"), f"{source_path}.content")
            if source_kind == "url" and self.blocked_url(source.get("url")):
                return f"{source_path}.url"
            if source_kind == "file":
                return (
                    f"{source_path}.file_id"
                    if source.get("file_id")
                    else f"{source_path}.type"
                )
            if source_kind not in ("base64", "text", "url"):
                return f"{source_path}.type"

        if kind in (
            "container_upload",
            "code_execution_output",
            "bash_code_execution_output",
        ) and value.get("file_id"):
            return f"{path}.file_id"
        if kind in self.wrappers:
            return self.blocked(value.get("content"), f"{path}.content")
        return None if kind in ANTHROPIC_SAFE_CONTENT_TYPES else f"{path}.type"

    def inspect(self, body: dict) -> None:
        for key in ("container", "mcp_servers"):
            if body.get(key):
                self.violations.append(key)
        self.content(body.get("system"), "system")
        for index, message in enumerate(body.get("messages") or []):
            if isinstance(message, dict):
                self.content(message.get("content"), f"messages[{index}].content")
        tools = self.tools(body.get("tools"))
        self.choice(body.get("tool_choice"), tools)


_POLICIES = {
    "/chat/completions": ChatPolicy,
    "/responses": ResponsesPolicy,
    "/v1/messages": AnthropicPolicy,
}


def check_request(path: str, body: dict, policy: NetworkPolicyConfig) -> None:
    if not policy.network_restricted:
        return
    inspector = _POLICIES.get(path)
    if inspector is None:
        raise ProviderPolicyError(["unsupported protocol"])
    check = inspector(policy)
    check.inspect(body)
    if check.violations:
        raise ProviderPolicyError(check.violations)
