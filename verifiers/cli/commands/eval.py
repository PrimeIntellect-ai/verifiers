"""Evaluation command module for external hosts.

This adapter preserves `vf-eval` behavior for local runs and adds hosted mode
support for prime-cli integrations via `--hosted`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import requests
from rich.console import Console

from verifiers.scripts import eval as vf_eval
from verifiers.utils.eval_utils import load_toml_config

console = Console()

DEFAULT_PRIME_API_BASE_URL = "https://api.primeintellect.ai"
DEFAULT_PRIME_FRONTEND_URL = "https://app.primeintellect.ai"
DEFAULT_PRIME_API_KEY_VAR = "PRIME_API_KEY"
INTERNAL_ENV_DISPLAY_HEADER = "X-Prime-Eval-Env-Display"

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"}
LOG_STATUSES = {"RUNNING", "COMPLETED", "FAILED"}

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
PROGRESS_BAR = re.compile(r".*\|[█▏▎▍▌▋▊▉ ]{10,}\|.*")
HOSTED_SLUG = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?:@[^\s@]+)?$")
HOSTED_AHEAD = re.compile(r"ahead of ([A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?:@[^\s@)]+)?)")

HOSTED_BOOL_FLAGS = {
    "--hosted",
    "--follow",
    "--allow-sandbox-access",
    "--allow-instances-access",
}
HOSTED_VALUE_FLAGS = {
    "--poll-interval",
    "--timeout-minutes",
    "--custom-secrets",
    "--eval-name",
}


class HostedEvalError(RuntimeError):
    """Raised when hosted evaluation setup or execution fails."""


def _strip_api_v1(url: str) -> str:
    return url.rstrip("/").removesuffix("/api/v1")


def _load_prime_config() -> dict[str, Any]:
    config_path = Path.home() / ".prime" / "config.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_api_base_url(config: dict[str, Any]) -> str:
    env_url = os.getenv("PRIME_API_BASE_URL") or os.getenv("PRIME_BASE_URL")
    if env_url:
        return _strip_api_v1(env_url)

    cfg_url = config.get("base_url")
    if isinstance(cfg_url, str) and cfg_url:
        return _strip_api_v1(cfg_url)

    return DEFAULT_PRIME_API_BASE_URL


def _resolve_frontend_url(config: dict[str, Any]) -> str:
    cfg_url = config.get("frontend_url")
    if isinstance(cfg_url, str) and cfg_url:
        return cfg_url.rstrip("/")
    return DEFAULT_PRIME_FRONTEND_URL


def _resolve_api_key(api_key_var: str | None, config: dict[str, Any]) -> str:
    if api_key_var:
        explicit = os.getenv(api_key_var)
        if explicit:
            return explicit

    fallback = os.getenv(DEFAULT_PRIME_API_KEY_VAR)
    if fallback:
        return fallback

    cfg_key = config.get("api_key")
    if isinstance(cfg_key, str) and cfg_key:
        return cfg_key

    if api_key_var:
        raise HostedEvalError(
            f"--api-key-var '{api_key_var}' is not set and PRIME_API_KEY is unavailable."
        )
    raise HostedEvalError(
        "No API key configured. Set PRIME_API_KEY or run 'prime login'."
    )


def _parse_headers(raw_headers: list[str] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_header in raw_headers or []:
        if ":" not in raw_header:
            raise HostedEvalError(
                f"--header must be 'Name: Value', got: {raw_header!r}"
            )
        name, value = raw_header.split(":", 1)
        header_name = name.strip()
        header_value = value.strip()
        if not header_name:
            raise HostedEvalError("--header name cannot be empty")
        headers[header_name] = header_value
    return headers


def _get_header(headers: dict[str, str], name: str) -> str | None:
    target = name.lower()
    for header_name, header_value in headers.items():
        if header_name.lower() == target:
            return header_value
    return None


def _request_json(
    method: str,
    base_url: str,
    endpoint: str,
    api_key: str,
    *,
    json_payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    normalized_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    url = f"{base_url.rstrip('/')}/api/v1{normalized_endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json_payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise HostedEvalError(f"API request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = ""
        try:
            body = response.json()
        except ValueError:
            body = None

        if isinstance(body, dict):
            detail_obj = body.get("detail")
            if isinstance(detail_obj, str):
                detail = detail_obj
            elif detail_obj is not None:
                detail = str(detail_obj)

        if not detail:
            detail = response.text.strip() or "Request failed"
        raise HostedEvalError(f"HTTP {response.status_code}: {detail}")

    try:
        data = response.json()
    except ValueError as exc:
        raise HostedEvalError("API returned non-JSON response") from exc

    if not isinstance(data, dict):
        raise HostedEvalError("API returned unexpected response shape")
    return data


def _is_help_request(argv: list[str]) -> bool:
    return any(arg in ("-h", "--help") for arg in argv)


def _strip_hosted_flags_for_help(argv: list[str]) -> list[str]:
    stripped: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in HOSTED_BOOL_FLAGS:
            i += 1
            continue
        if token in HOSTED_VALUE_FLAGS:
            i += 2
            continue
        if any(token.startswith(f"{flag}=") for flag in HOSTED_VALUE_FLAGS):
            i += 1
            continue
        stripped.append(token)
        i += 1
    return stripped


def _has_hosted_only_flags(argv: list[str]) -> bool:
    for token in argv:
        if token in HOSTED_VALUE_FLAGS or token in (
            "--follow",
            "--allow-sandbox-access",
            "--allow-instances-access",
        ):
            return True
        if any(token.startswith(f"{flag}=") for flag in HOSTED_VALUE_FLAGS):
            return True
    return False


def _run_vf_eval(argv: list[str]) -> None:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *argv]
        vf_eval.main()
    finally:
        sys.argv = original_argv


def _build_hosted_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("env_id_or_config", nargs="?")
    parser.add_argument("--hosted", action="store_true")
    parser.add_argument("--model", "-m", type=str, default=vf_eval.DEFAULT_MODEL)
    parser.add_argument("--num-examples", "-n", type=int, default=None)
    parser.add_argument("--rollouts-per-example", "-r", type=int, default=None)
    parser.add_argument("--env-args", "-a", type=str, default=None)
    parser.add_argument(
        "--env-dir-path", "-p", type=str, default=vf_eval.DEFAULT_ENV_DIR_PATH
    )
    parser.add_argument("--api-key-var", "-k", type=str, default=None)
    parser.add_argument("--header", action="append", default=None)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--timeout-minutes", type=int, default=None)
    parser.add_argument("--allow-sandbox-access", action="store_true")
    parser.add_argument("--allow-instances-access", action="store_true")
    parser.add_argument("--custom-secrets", type=str, default=None)
    parser.add_argument("--eval-name", type=str, default=None)
    return parser


def _is_slug_reference(value: str) -> bool:
    if "/" not in value:
        return False
    if value.startswith(("./", "../", "/")):
        return False
    return True


def _split_slug_and_version(value: str) -> tuple[str, str]:
    if "@" not in value:
        return value, "latest"
    slug, version = value.rsplit("@", 1)
    if not version:
        raise HostedEvalError(f"Invalid environment version in '{value}'")
    return slug, version


def _resolve_slug_from_header(display_header: str | None) -> tuple[str, str] | None:
    if not display_header:
        return None

    stripped = display_header.strip()
    if HOSTED_SLUG.match(stripped):
        return _split_slug_and_version(stripped)

    match = HOSTED_AHEAD.search(stripped)
    if match:
        return _split_slug_and_version(match.group(1))

    return None


def _resolve_slug_from_local_metadata(
    env_name: str, env_dir_path: str
) -> tuple[str, str] | None:
    env_root = Path(env_dir_path)
    local_dir = env_root / env_name.replace("-", "_")
    candidates = [
        local_dir / ".prime" / ".env-metadata.json",
        local_dir / ".env-metadata.json",
    ]

    metadata: dict[str, Any] | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            metadata = data
            break

    if metadata is None:
        return None

    owner = metadata.get("owner")
    name = metadata.get("name")
    if not isinstance(owner, str) or not owner:
        return None
    if not isinstance(name, str) or not name:
        return None

    version = metadata.get("version")
    if isinstance(version, str) and version:
        return f"{owner}/{name}", version
    return f"{owner}/{name}", "latest"


def _resolve_hosted_slug(
    env_ref: str, env_dir_path: str, headers: dict[str, str]
) -> tuple[str, str]:
    if _is_slug_reference(env_ref):
        return _split_slug_and_version(env_ref)

    header_slug = _resolve_slug_from_header(
        _get_header(headers, INTERNAL_ENV_DISPLAY_HEADER)
    )
    if header_slug is not None:
        return header_slug

    metadata_slug = _resolve_slug_from_local_metadata(env_ref, env_dir_path)
    if metadata_slug is not None:
        return metadata_slug

    raise HostedEvalError(
        "Hosted evaluations require an upstream environment slug (owner/name). "
        "Pass a slug directly or push/link the local environment first."
    )


def _parse_json_object(value: str | None, *, flag_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HostedEvalError(f"Error parsing {flag_name}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise HostedEvalError(f"{flag_name} must be a JSON object.")
    return parsed


def _extract_environment_id(env_response: dict[str, Any]) -> str:
    details = env_response.get("data", env_response)
    if not isinstance(details, dict):
        raise HostedEvalError("Environment lookup returned invalid response shape.")
    env_id = details.get("id")
    if not isinstance(env_id, str) or not env_id:
        raise HostedEvalError("Environment lookup did not return an environment id.")
    return env_id


def _status_color(status: str) -> str:
    return {
        "PENDING": "yellow",
        "RUNNING": "cyan",
        "COMPLETED": "green",
        "FAILED": "red",
        "TIMEOUT": "red",
        "CANCELLED": "yellow",
    }.get(status, "white")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def _filter_progress_bars(text: str) -> str:
    lines = text.splitlines()
    filtered: list[str] = []
    for line in lines:
        if PROGRESS_BAR.search(line) or re.search(r"\d+%\|", line):
            if "100%" in line:
                match = re.search(
                    r"([^|]*100%\|[█▏▎▍▌▋▊▉ ]+\|[^\n]*?)(?=\d+%\||$)",
                    line,
                )
                if match:
                    filtered.append(match.group(1).strip())
                else:
                    filtered.append(line)
            continue
        if line.strip():
            filtered.append(line)
    return "\n".join(filtered)


def _clean_logs(text: str) -> str:
    return _filter_progress_bars(_strip_ansi(text))


def _get_new_log_lines(previous: str, current: str) -> list[str]:
    previous_lines = previous.splitlines() if previous else []
    current_lines = current.splitlines()
    if not previous:
        return current_lines

    overlap = 0
    max_overlap = min(len(previous_lines), len(current_lines))
    for size in range(1, max_overlap + 1):
        if previous_lines[-size:] == current_lines[:size]:
            overlap = size
    return current_lines[overlap:]


def _print_hosted_help_footer() -> None:
    console.print(
        "\nHosted options:\n"
        "  --hosted                    Run the evaluation on Prime platform.\n"
        "  --poll-interval FLOAT       Polling interval in seconds (default: 10).\n"
        "  --follow                    Stream hosted status/logs until completion.\n"
        "  --timeout-minutes INT       Hosted evaluation timeout in minutes.\n"
        "  --allow-sandbox-access      Allow sandbox read/write access.\n"
        "  --allow-instances-access    Allow pod/instance management.\n"
        "  --custom-secrets JSON       Secret key/value mapping as JSON.\n"
        "  --eval-name TEXT            Custom hosted evaluation name."
    )


def _poll_hosted_evaluation(
    *,
    base_url: str,
    api_key: str,
    eval_id: str,
    poll_interval: float,
    follow_logs: bool,
) -> dict[str, Any]:
    last_status: str | None = None
    last_logs = ""

    while True:
        eval_data = _request_json("GET", base_url, f"/evaluations/{eval_id}", api_key)
        status = str(eval_data.get("status", "UNKNOWN")).upper()
        total_samples = eval_data.get("total_samples", 0)

        if status != last_status:
            color = _status_color(status)
            console.print(
                f"[{color}]Status: {status}[/{color}] | Samples: {total_samples}"
            )
            last_status = status

        if follow_logs and status in LOG_STATUSES:
            try:
                log_response = _request_json(
                    "GET",
                    base_url,
                    f"/hosted-evaluations/{eval_id}/logs",
                    api_key,
                )
                cleaned = _clean_logs(str(log_response.get("logs") or ""))
            except HostedEvalError:
                cleaned = ""

            if cleaned and cleaned != last_logs:
                for line in _get_new_log_lines(last_logs, cleaned):
                    console.print(line)
                last_logs = cleaned

        if status in TERMINAL_STATUSES:
            return eval_data

        time.sleep(max(poll_interval, 1.0))


def _resolve_eval_counts(
    env_ref: str,
    raw_num_examples: Any,
    raw_rollouts_per_example: Any,
) -> tuple[int, int]:
    if raw_num_examples is not None and (
        not isinstance(raw_num_examples, int) or isinstance(raw_num_examples, bool)
    ):
        raise HostedEvalError(
            f"num_examples for '{env_ref}' must be an integer, got {raw_num_examples!r}."
        )
    if raw_rollouts_per_example is not None and (
        not isinstance(raw_rollouts_per_example, int)
        or isinstance(raw_rollouts_per_example, bool)
    ):
        raise HostedEvalError(
            "rollouts_per_example for "
            f"'{env_ref}' must be an integer, got {raw_rollouts_per_example!r}."
        )

    defaults = vf_eval.get_env_eval_defaults(env_ref)
    num_examples = (
        raw_num_examples
        if raw_num_examples is not None
        else defaults.get("num_examples", vf_eval.DEFAULT_NUM_EXAMPLES)
    )
    rollouts_per_example = (
        raw_rollouts_per_example
        if raw_rollouts_per_example is not None
        else defaults.get("rollouts_per_example", vf_eval.DEFAULT_ROLLOUTS_PER_EXAMPLE)
    )
    return num_examples, rollouts_per_example


def _resolve_config_env_dir_path(
    env_dir_path: str | None, *, config_path: Path, fallback: str
) -> str:
    raw = env_dir_path if env_dir_path else fallback
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str((config_path.parent / path).resolve())


def _build_hosted_specs(
    args: argparse.Namespace, headers: dict[str, str]
) -> list[dict[str, Any]]:
    env_ref = args.env_id_or_config
    if not env_ref:
        raise HostedEvalError("Missing environment argument.")

    if not env_ref.endswith(".toml"):
        env_slug, requested_version = _resolve_hosted_slug(
            env_ref=env_ref,
            env_dir_path=args.env_dir_path,
            headers=headers,
        )
        parsed_env_args = _parse_json_object(args.env_args, flag_name="--env-args")
        num_examples, rollouts_per_example = _resolve_eval_counts(
            env_ref=env_ref,
            raw_num_examples=args.num_examples,
            raw_rollouts_per_example=args.rollouts_per_example,
        )
        return [
            {
                "source_env_ref": env_ref,
                "env_slug": env_slug,
                "requested_version": requested_version,
                "model": args.model,
                "num_examples": num_examples,
                "rollouts_per_example": rollouts_per_example,
                "env_args": parsed_env_args,
            }
        ]

    config_path = Path(env_ref)
    if not config_path.is_file():
        raise HostedEvalError(
            f"Hosted eval config file not found: {config_path}. "
            "Pass an existing TOML path."
        )

    try:
        raw_eval_configs = load_toml_config(config_path)
    except Exception as exc:
        raise HostedEvalError(f"Failed to parse hosted TOML config: {exc}") from exc

    specs: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_eval_configs, start=1):
        if not isinstance(raw, dict):
            raise HostedEvalError(
                f"Invalid eval entry at index {index} in {config_path}."
            )

        entry_env_ref = raw.get("env_id")
        if not isinstance(entry_env_ref, str) or not entry_env_ref:
            raise HostedEvalError(
                f"Hosted TOML eval entry {index} is missing a valid env_id."
            )

        if raw.get("endpoint_id") is not None and raw.get("model") is None:
            raise HostedEvalError(
                f"Hosted TOML eval entry {index} ({entry_env_ref}) sets endpoint_id "
                "without model. Set model explicitly for hosted mode."
            )

        raw_model = raw.get("model")
        if raw_model is not None and not isinstance(raw_model, str):
            raise HostedEvalError(
                f"model for '{entry_env_ref}' must be a string, got {raw_model!r}."
            )
        model = (
            raw_model
            if isinstance(raw_model, str) and raw_model
            else vf_eval.DEFAULT_MODEL
        )

        raw_env_args = raw.get("env_args")
        if raw_env_args is not None and not isinstance(raw_env_args, dict):
            raise HostedEvalError(
                f"env_args for '{entry_env_ref}' must be a JSON object/table."
            )

        num_examples, rollouts_per_example = _resolve_eval_counts(
            env_ref=entry_env_ref,
            raw_num_examples=raw.get("num_examples"),
            raw_rollouts_per_example=raw.get("rollouts_per_example"),
        )

        env_dir_path = _resolve_config_env_dir_path(
            raw.get("env_dir_path")
            if isinstance(raw.get("env_dir_path"), str)
            else None,
            config_path=config_path,
            fallback=args.env_dir_path,
        )
        env_slug, requested_version = _resolve_hosted_slug(
            env_ref=entry_env_ref,
            env_dir_path=env_dir_path,
            headers=headers,
        )
        specs.append(
            {
                "source_env_ref": entry_env_ref,
                "env_slug": env_slug,
                "requested_version": requested_version,
                "model": model,
                "num_examples": num_examples,
                "rollouts_per_example": rollouts_per_example,
                "env_args": raw_env_args,
            }
        )

    return specs


def _print_final_hosted_result(final_data: dict[str, Any]) -> str:
    final_status = str(final_data.get("status", "UNKNOWN")).upper()
    color = _status_color(final_status)
    console.print()
    console.print(f"[{color}]Final status: {final_status}[/{color}]")

    for key, label in (
        ("total_samples", "Total samples"),
        ("avg_score", "Average score"),
        ("min_score", "Min score"),
        ("max_score", "Max score"),
    ):
        value = final_data.get(key)
        if value is not None:
            console.print(f"{label}: {value}")

    error_message = final_data.get("error_message")
    if isinstance(error_message, str) and error_message:
        console.print(f"[red]Error:[/red] {error_message}")

    return final_status


def _run_hosted_eval(argv: list[str]) -> None:
    parser = _build_hosted_parser()
    args, _ = parser.parse_known_args(argv)

    config = _load_prime_config()
    parsed_headers = _parse_headers(args.header)
    api_key = _resolve_api_key(args.api_key_var, config)
    base_url = _resolve_api_base_url(config)
    frontend_url = _resolve_frontend_url(config)
    parsed_custom_secrets = _parse_json_object(
        args.custom_secrets, flag_name="--custom-secrets"
    )
    specs = _build_hosted_specs(args, parsed_headers)

    created: list[tuple[str, str | None]] = []
    for index, spec in enumerate(specs, start=1):
        env_slug = str(spec["env_slug"])
        requested_version = str(spec["requested_version"])
        owner, env_name = env_slug.split("/", 1)
        console.print(
            f"[dim]Using hosted environment: {env_slug}@{requested_version}[/dim]"
        )

        env_response = _request_json(
            "GET",
            base_url,
            f"/environmentshub/{owner}/{env_name}/@{requested_version}",
            api_key,
        )
        environment_id = _extract_environment_id(env_response)

        eval_config: dict[str, Any] = {
            "num_examples": int(spec["num_examples"]),
            "rollouts_per_example": int(spec["rollouts_per_example"]),
            "allow_sandbox_access": args.allow_sandbox_access,
            "allow_instances_access": args.allow_instances_access,
        }
        env_args = spec.get("env_args")
        if isinstance(env_args, dict) and env_args:
            eval_config["env_args"] = env_args
        if args.timeout_minutes is not None:
            eval_config["timeout_minutes"] = args.timeout_minutes
        if parsed_custom_secrets:
            eval_config["custom_secrets"] = parsed_custom_secrets

        payload: dict[str, Any] = {
            "environment_ids": [environment_id],
            "inference_model": str(spec["model"]),
            "eval_config": eval_config,
        }
        if args.eval_name:
            payload["name"] = (
                args.eval_name if len(specs) == 1 else f"{args.eval_name}-{index}"
            )

        create_response = _request_json(
            "POST",
            base_url,
            "/hosted-evaluations",
            api_key,
            json_payload=payload,
        )
        eval_id = create_response.get("evaluation_id")
        if not isinstance(eval_id, str) or not eval_id:
            raise HostedEvalError(
                "Hosted evaluation response did not include evaluation_id."
            )

        viewer_url = create_response.get("viewer_url")
        if not isinstance(viewer_url, str) or not viewer_url:
            viewer_url = f"{frontend_url}/dashboard/evaluations/{eval_id}"

        console.print(f"[green]Created hosted evaluation:[/green] {eval_id}")
        console.print(f"[green]View:[/green] {viewer_url}")
        created.append((eval_id, viewer_url))

    if not args.follow:
        console.print(
            "[dim]Use --follow to stream status and logs until completion.[/dim]"
        )
        return

    failed_evals: list[str] = []
    for eval_id, _ in created:
        if len(created) > 1:
            console.print(f"\n[bold]Following hosted evaluation {eval_id}[/bold]")

        final_data = _poll_hosted_evaluation(
            base_url=base_url,
            api_key=api_key,
            eval_id=eval_id,
            poll_interval=args.poll_interval,
            follow_logs=True,
        )
        final_status = _print_final_hosted_result(final_data)
        if final_status != "COMPLETED":
            failed_evals.append(eval_id)

    if failed_evals:
        raise HostedEvalError(
            "Hosted evaluation did not complete successfully: "
            + ", ".join(failed_evals)
        )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    if _is_help_request(args):
        try:
            _run_vf_eval(_strip_hosted_flags_for_help(args))
        except SystemExit as exc:
            if exc.code == 0:
                _print_hosted_help_footer()
            raise
        return

    parser = _build_hosted_parser()
    known, _ = parser.parse_known_args(args)

    if known.hosted:
        try:
            _run_hosted_eval(args)
        except HostedEvalError as exc:
            console.print(f"[red]Hosted evaluation failed:[/red] {exc}")
            raise SystemExit(1) from exc
        return

    if _has_hosted_only_flags(args):
        console.print(
            "[red]Hosted-only flags require --hosted.[/red] "
            "Use `prime eval run <env> --hosted ...`."
        )
        raise SystemExit(2)

    _run_vf_eval(args)


if __name__ == "__main__":
    main()
