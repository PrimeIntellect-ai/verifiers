"""What a network-restricted runtime may send to its provider.

Provider-side tools and remote references run outside a runtime's egress controls, so the
interception server mediates a restricted rollout's native requests before they leave (see
`mediate`). Tools that would execute at the provider are removed, or, for web search, kept
with domain filters that express the policy; content the provider would fetch stays only when
the policy permits its host; references to stored provider state are dropped, and so is any
content type not known to be safe. When something is removed, the earliest user message says
so. Each dialect's wire rules are one `RequestFilter` subclass.
"""

import json
import re
from glob import has_magic
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, ValidationError

from verifiers.v1.configs.runtime import (
    NetworkPolicyConfig,
    intersect_network_hosts,
    parse_network_rule,
)
from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.base import Dialect, RawRequest
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.dialects.responses import ResponsesDialect

PROVIDER_CAPABILITY_POLICY_CODE = "provider_capability_unavailable"
CAPABILITY_NOTICE = (
    "Some request content or provider-side capabilities were omitted because they are "
    "blocked by the network policy or cannot enforce it."
)


class RequestFilter:
    """One request's omissions and blocked URLs; subclasses define native wire rules."""

    wrappers: tuple[str, ...] = ()

    def __init__(self, policy: NetworkPolicyConfig):
        self.policy = policy
        self.blocked_urls: list[str] = []
        self.capabilities: list[str] = []

    def mediate_request(self, body: RawRequest) -> RawRequest:
        """Filter `body` in place; omissions land on `capabilities`."""
        raise NotImplementedError

    def blocked_url(self, value: object) -> bool:
        if not isinstance(value, str):
            return True
        if value.lower().startswith("data:"):
            return False
        try:
            url = AnyHttpUrl(value)
        except ValidationError:
            return True
        blocked = not self.policy.permits(url.scheme, url.host.strip("[]"), url.port)
        if blocked:
            self.blocked_urls.append(value)
        return blocked

    def blocked(self, value, path: str) -> str | None:
        """Find the first forbidden part, treating nested content as one unit."""
        if isinstance(value, list):
            for index, item in enumerate(value):
                if blocked := self.blocked(item, f"{path}[{index}]"):
                    return blocked
            return None
        if not isinstance(value, dict):
            return None
        caller = value.get("caller")
        if caller is not None and not (
            isinstance(caller, dict) and caller.get("type") == "direct"
        ):
            return f"{path}.caller.type"
        return self.blocked_part(value, path)

    def blocked_part(self, value: dict, path: str) -> str | None:
        raise NotImplementedError

    def mediate(self, value, path: str):
        """Remove forbidden parts while retaining supported wrapper blocks."""
        if not isinstance(value, list):
            if blocked := self.blocked(value, path):
                self.capabilities.append(blocked)
                return ""
            return value
        mediated = []
        for index, block in enumerate(value):
            item_path = f"{path}[{index}]"
            wrapper = isinstance(block, dict) and block.get("type") in self.wrappers
            scan = {**block, "content": []} if wrapper else block
            if blocked := self.blocked(scan, item_path):
                self.capabilities.append(blocked)
                continue
            if wrapper:
                self.content(block, "content", f"{item_path}.content")
            mediated.append(block)
        return mediated

    def content(self, parent: dict, key: str, path: str) -> bool:
        """Rewrite a content field only when filtering removes something."""
        before = len(self.capabilities)
        content = self.mediate(parent.get(key), path)
        changed = len(self.capabilities) != before
        if changed:
            parent[key] = content or ""
        return changed

    def tools(self, value, path: str = "tools") -> list[dict]:
        if value is not None and not isinstance(value, list):
            self.capabilities.append(path)
            return []
        tools = []
        for index, tool in enumerate(value or []):
            if (filtered := self.tool(tool, f"{path}[{index}]")) is not None:
                tools.append(filtered)
        return tools

    def tool(self, value, path: str) -> dict | None:
        raise NotImplementedError


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


def append_user_notice(
    messages: list,
    *,
    blocked_urls: list[str],
    text_type: str = "text",
    message_type: str | None = None,
) -> None:
    """Explain an actual policy-driven omission in the earliest user input."""
    notice = CAPABILITY_NOTICE
    if blocked_urls:
        notice += "\nBlocked URLs: " + ", ".join(
            json.dumps(url) for url in dict.fromkeys(blocked_urls)
        )
        notice += "\nCircumventing this block is forbidden."
    part = {"type": text_type, "text": notice}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [*content, part]
        elif isinstance(content, str):
            message["content"] = f"{content}\n\n{notice}" if content else notice
        else:
            message["content"] = [part]
        return
    message = {"role": "user", "content": [part]}
    if message_type is not None:
        message["type"] = message_type
    messages.append(message)


# --- Chat Completions ----------------------------------------------------------------------

# Client tools return calls to the harness; every other type may execute at the provider.
_CHAT_CLIENT_TOOL_TYPES = ("function", "custom")
_CHAT_SAFE_CONTENT_TYPES = ("text", "refusal", "input_audio", "image_url", "file")


class ChatFilter(RequestFilter):
    def tool(self, tool, path: str) -> dict | None:
        if (
            isinstance(tool, dict)
            and tool.get("type", "function") in _CHAT_CLIENT_TOOL_TYPES
        ):
            return tool
        self.capabilities.append(f"{path}.type")
        return None

    def blocked(self, value, path: str) -> str | None:
        # Chat content parts are flat; nested lists and non-object parts are invalid.
        kind = value.get("type") if isinstance(value, dict) else None
        if kind not in _CHAT_SAFE_CONTENT_TYPES:
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

    def mediate_request(self, body: RawRequest) -> RawRequest:
        capabilities = self.capabilities
        if body.pop("web_search_options", None) is not None:
            capabilities.append("web_search_options")
        if body.pop("plugins", None) is not None:
            capabilities.append("plugins")

        audio = body.get("audio")
        voice = audio.get("voice") if isinstance(audio, dict) else None
        if isinstance(voice, dict) and voice.get("id"):
            capabilities.append("audio.voice.id")
            body.pop("audio")
            modalities = body.get("modalities")
            if isinstance(modalities, list):
                body["modalities"] = [
                    item for item in modalities if item != "audio"
                ] or ["text"]

        raw_tools = body.get("tools")
        tools = self.tools(raw_tools)
        if "tools" in body:
            body["tools"] = tools

        choice = body.get("tool_choice")
        valid_choice = choice is None or (
            isinstance(choice, str) and choice in ("none", "auto", "required")
        )
        if isinstance(choice, dict):
            kind = choice.get("type", "function")
            valid_choice = any(
                kind == tool.get("type", "function")
                and isinstance(tool.get(kind), dict)
                and isinstance(choice.get(kind), dict)
                and tool[kind].get("name") == choice[kind].get("name")
                for tool in tools
            )
            if kind == "allowed_tools":
                allowed = choice.get("allowed_tools")
                allowed_tools = (
                    allowed.get("tools") if isinstance(allowed, dict) else None
                )
                valid_choice = isinstance(allowed_tools, list) and all(
                    isinstance(tool, dict)
                    and tool.get("type", "function") in _CHAT_CLIENT_TOOL_TYPES
                    for tool in allowed_tools
                )
        if raw_tools is not None and not tools and choice not in (None, "none"):
            valid_choice = False
        if not valid_choice:
            capabilities.append(
                "tool_choice.type" if isinstance(choice, dict) else "tool_choice"
            )
            body.pop("tool_choice", None)

        for message_index, message in enumerate(body.get("messages") or []):
            if not isinstance(message, dict):
                continue
            if isinstance(message.get("audio"), dict) and message["audio"].get("id"):
                capabilities.append(f"messages[{message_index}].audio.id")
                message.pop("audio")
                if message.get("content") is None:
                    message["content"] = ""
            content = message.get("content")
            if not isinstance(content, list):
                continue
            safe_content = self.mediate(content, f"messages[{message_index}].content")
            message["content"] = safe_content or ""

        if capabilities:
            append_user_notice(
                body.setdefault("messages", []), blocked_urls=self.blocked_urls
            )
        return body


# --- Responses -----------------------------------------------------------------------------

# Client tools return calls to the harness; every other type may execute at the provider.
_RESPONSES_CLIENT_TOOL_TYPES = (
    "function",
    "custom",
    "local_shell",
    "apply_patch",
    "computer",
    "computer_use_preview",
)
_WEB_SEARCH_TOOL_TYPE = re.compile(r"web_search(?:_\d{4}_\d{2}_\d{2})?").fullmatch
_SAFE_INPUT_TYPES = (
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
_TEXT_TOOL_OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")
BLANK_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M/wHwAF/gL+Xw4AAAAASUVORK5CYII="
)


class ResponsesFilter(RequestFilter):
    def tool(self, tool, path: str) -> dict | None:
        if not isinstance(tool, dict):
            self.capabilities.append(path)
            return None
        kind = tool.get("type")
        if isinstance(kind, str) and _WEB_SEARCH_TOOL_TYPE(kind):
            filter_key = "blocked_domains" if self.policy.block else "allowed_domains"
            filters = tool.get("filters")
            if filters is None:
                filters = {}
            if isinstance(filters, dict):
                domains = provider_domains(self.policy, filters.get(filter_key))
                if domains and len(domains) <= 100:
                    return {**tool, "filters": {**filters, filter_key: domains}}
        if kind == "namespace":
            nested = self.tools(tool.get("tools"), f"{path}.tools")
            return {**tool, "tools": nested} if nested else None
        environment = tool.get("environment")
        if (
            kind in _RESPONSES_CLIENT_TOOL_TYPES
            or kind == "tool_search"
            and tool.get("execution") == "client"
            or kind == "shell"
            and isinstance(environment, dict)
            and environment.get("type") == "local"
        ):
            return tool
        self.capabilities.append(f"{path}.type")
        return None

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
            # Nested tool lists are atomic: report only their first omission.
            probe = ResponsesFilter(self.policy)
            probe.tools(value.get("tools"), f"{path}.tools")
            return next(iter(probe.capabilities), None)

        if kind in ("computer_call_output", *_TEXT_TOOL_OUTPUT_TYPES):
            return self.blocked(value.get("output"), f"{path}.output")
        if kind in (None, "message") and "role" in value and "content" in value:
            return self.blocked(value["content"], f"{path}.content")
        return None if kind in _SAFE_INPUT_TYPES else f"{path}.type"

    def mediate_request(self, body: RawRequest) -> RawRequest:
        capabilities = self.capabilities
        for field in ("previous_response_id", "conversation", "prompt", "plugins"):
            if body.pop(field, None) is not None:
                capabilities.append(field)

        raw_input = body.get("input")
        if isinstance(raw_input, list):
            safe_input = []
            for item_index, item in enumerate(raw_input):
                item_path = f"input[{item_index}]"
                if not isinstance(item, dict):
                    safe_input.append(item)
                    continue
                kind = item.get("type")
                if kind in ("additional_tools", "tool_search_output"):
                    if blocked := self.blocked({**item, "tools": []}, item_path):
                        capabilities.append(blocked)
                        continue
                    item["tools"] = self.tools(item.get("tools"), f"{item_path}.tools")
                    if kind == "tool_search_output" or item["tools"]:
                        safe_input.append(item)
                    continue
                content_field = None
                if kind in _TEXT_TOOL_OUTPUT_TYPES:
                    content_field = "output"
                elif kind in (None, "message") and "content" in item:
                    content_field = "content"

                if content_field:
                    self.content(item, content_field, f"{item_path}.{content_field}")

                scan = {**item, content_field: []} if content_field else item
                blocked = self.blocked(scan, item_path)
                if blocked is None:
                    safe_input.append(item)
                else:
                    capabilities.append(blocked)
                    if kind == "computer_call_output" and blocked.startswith(
                        f"{item_path}.output"
                    ):
                        item["output"] = {
                            "type": "computer_screenshot",
                            "image_url": BLANK_PNG,
                        }
                        safe_input.append(item)
            body["input"] = safe_input
        elif blocked := self.blocked(raw_input, "input"):
            capabilities.append(blocked)
            body["input"] = []

        tools = self.tools(body.get("tools"))
        if "tools" in body:
            body["tools"] = tools
            if not tools:
                body.pop("tool_choice", None)

        choice = body.get("tool_choice")
        valid_choice = choice is None or (
            isinstance(choice, str) and choice in ("none", "auto", "required")
        )
        if isinstance(choice, dict):
            kind = choice.get("type")
            valid_choice = any(
                tool.get("type") == kind
                and ("name" not in choice or tool.get("name") == choice["name"])
                for tool in tools
            )
            if kind == "allowed_tools":
                # Tool-choice validation reports the choice as one capability.
                choice_filter = ResponsesFilter(self.policy)
                choice_tools = choice_filter.tools(
                    choice.get("tools"), "tool_choice.tools"
                )
                valid_choice = not choice_filter.capabilities
                body["tool_choice"] = {**choice, "tools": choice_tools}
        if not valid_choice:
            capabilities.append("tool_choice")
            body.pop("tool_choice")

        if not capabilities:
            return body

        input_items = body.get("input")
        if not isinstance(input_items, list):
            input_items = (
                []
                if input_items is None
                else [{"role": "user", "content": input_items}]
            )
        append_user_notice(
            input_items,
            blocked_urls=self.blocked_urls,
            text_type="input_text",
            message_type="message",
        )
        body["input"] = input_items
        return body


# --- Anthropic Messages --------------------------------------------------------------------

# These versioned tool families return calls to the harness; every other typed tool may execute
# at the provider. Anchoring the pattern keeps new versions client-side without treating an
# arbitrary dated provider tool as safe.
_ANTHROPIC_CLIENT_TOOL_TYPE = re.compile(
    r"(?:bash|text_editor|computer|memory)_\d{8}"
).fullmatch
_WEB_TOOL_TYPE = re.compile(r"web_(?:search|fetch)_\d{8}").fullmatch
_CONTENT_WRAPPERS = (
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
_ANTHROPIC_SAFE_CONTENT_TYPES = (
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


class AnthropicFilter(RequestFilter):
    wrappers = _CONTENT_WRAPPERS

    def tool(self, tool, path: str) -> dict | None:
        kind = tool.get("type") if isinstance(tool, dict) else None
        if isinstance(kind, str) and _WEB_TOOL_TYPE(kind):
            callers = tool.get("allowed_callers")
            filter_key = "blocked_domains" if self.policy.block else "allowed_domains"
            other_key = "allowed_domains" if self.policy.block else "blocked_domains"
            domains = (
                provider_domains(self.policy, tool.get(filter_key))
                # Anthropic does not support combining allow and block filters.
                if tool.get(other_key) in (None, [])
                and (
                    callers is None or isinstance(callers, list) and "direct" in callers
                )
                else []
            )
            if domains:
                web_tool = {**tool, filter_key: domains, "allowed_callers": ["direct"]}
                web_tool.pop(other_key, None)
                return web_tool
        if isinstance(tool, dict) and (
            (isinstance(kind, str) and _ANTHROPIC_CLIENT_TOOL_TYPE(kind))
            or (kind in (None, "custom") and "input_schema" in tool)
        ):
            return tool
        self.capabilities.append(f"{path}.type")
        return None

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
        return None if kind in _ANTHROPIC_SAFE_CONTENT_TYPES else f"{path}.type"

    def mediate_request(self, body: RawRequest) -> RawRequest:
        capabilities = self.capabilities
        for key in ("container", "mcp_servers"):
            if body.pop(key, None):
                capabilities.append(key)

        if self.content(body, "system", "system") and not body["system"]:
            body.pop("system")

        for message_index, message in enumerate(body.get("messages") or []):
            if not isinstance(message, dict):
                continue
            self.content(message, "content", f"messages[{message_index}].content")

        tools = self.tools(body.get("tools"))
        if "tools" in body:
            body["tools"] = tools

        choice = body.get("tool_choice")
        valid_choice = choice is None
        if isinstance(choice, dict):
            kind = choice.get("type")
            valid_choice = (
                kind == "none"
                or bool(tools)
                and (
                    kind in ("auto", "any")
                    or kind == "tool"
                    and any(tool.get("name") == choice.get("name") for tool in tools)
                )
            )
        if not valid_choice:
            capabilities.append(
                "tool_choice.type" if isinstance(choice, dict) else "tool_choice"
            )
            body.pop("tool_choice", None)

        if capabilities:
            append_user_notice(
                body.setdefault("messages", []), blocked_urls=self.blocked_urls
            )
        return body


_FILTERS: dict[type[Dialect], type[RequestFilter]] = {
    ChatDialect: ChatFilter,
    ResponsesDialect: ResponsesFilter,
    AnthropicDialect: AnthropicFilter,
}


def mediate(
    dialect: Dialect, body: RawRequest, policy: NetworkPolicyConfig
) -> tuple[RawRequest, list[str]]:
    """Filter blocked content and constrain provider tools to the network policy, in place.

    Provider tools execute outside runtime egress controls; remove them when their filters
    cannot express the policy. Add context only when something is removed. Returns the body and
    the removed paths, which never contain request values.
    """
    request_filter = _FILTERS[type(dialect)](policy)
    body = request_filter.mediate_request(body)
    return body, list(dict.fromkeys(request_filter.capabilities))
