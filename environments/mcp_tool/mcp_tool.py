from __future__ import annotations

import os
import asyncio
import inspect
import json
import logging
import os
import shlex
import time
import uuid
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Any as AnyType

from datasets import Dataset
from prime_cli.api.sandbox import AsyncSandboxClient, CreateSandboxRequest

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset

try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None
    
@dataclass
class SandboxConfig:
    image: str = "python:3.11-slim"
    cpu_cores: int = 1
    memory_gb: int = 2
    disk_size_gb: int = 10
    start_command: str = "tail -f /dev/null"
    api_key: Optional[str] = None


@dataclass
class MCPServerSpec:
    """
    Specification for a single MCP server we can launch via FastMCP.
    """
    name: str

    # Preferred: python script path inside the sandbox
    script: Optional[str] = None

    # Optional: remote MCP over HTTP/SSE (no local process)
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    # Fallback: explicit stdio command (Python-only; do not use Node here)
    command: Optional[List[str]] = None

    env: Dict[str, str] = field(default_factory=dict)
    workdir: Optional[str] = None

    # Timeouts (seconds)
    discovery_timeout_s: Optional[int] = None
    call_timeout_s: Optional[int] = None


@dataclass
class DiscoveredTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = None


class PrimeSandboxManager:
    """Manage lifecycle + command execution for a Prime sandbox."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.api_key = self.config.api_key or os.getenv("PRIME_API_KEY")
        self.client = AsyncSandboxClient(api_key=self.api_key)
        self.sandbox_id: Optional[str] = None
        self.logger = logging.getLogger("verifiers.envs.MCPToolEnv.SandboxManager")

    async def __aenter__(self) -> "PrimeSandboxManager":
        await self.create()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.cleanup()

    async def create(self) -> str:
        try:
            req = CreateSandboxRequest(
                name=f"mcp-fastmcp-{int(time.time())}",
                docker_image=self.config.image,
                cpu_cores=self.config.cpu_cores,
                memory_gb=self.config.memory_gb,
                disk_size_gb=self.config.disk_size_gb,
                start_command=self.config.start_command,
            )
            self.logger.info(f"Creating sandbox with image {self.config.image}")
            sandbox = await self.client.create(req)
            self.sandbox_id = sandbox.id
            await self.client.wait_for_creation(self.sandbox_id, max_attempts=90)
            self.logger.info(f"Sandbox ready: {self.sandbox_id}")
            return self.sandbox_id
        except Exception as e:
            # Check if it's a service unavailable error
            if "503" in str(e) or "Service unavailable" in str(e):
                self.logger.error("Prime Intellect sandbox service is currently unavailable (503). Please try again later.")
                raise RuntimeError("Prime Intellect sandbox service unavailable. The service may be down for maintenance.")
            self.logger.exception("Failed to create sandbox")
            raise

    async def run(self, command: str) -> Tuple[int, str, str]:
        """Run a shell command inside the sandbox."""
        if not self.sandbox_id:
            raise RuntimeError("Sandbox not created")
        try:
            res = await self.client.execute_command(
                sandbox_id=self.sandbox_id,
                command=command,
            )
            return res.exit_code, res.stdout, res.stderr
        except Exception as e:
            # Check if it's a service unavailable error
            if "503" in str(e) or "Service unavailable" in str(e):
                self.logger.error("Prime Intellect sandbox service is currently unavailable (503). Please try again later.")
                raise RuntimeError("Prime Intellect sandbox service unavailable. The service may be down for maintenance.")
            self.logger.exception("Sandbox command failed")
            raise

    async def upload_file(self, remote_path: str, content: str) -> None:
        """
        Upload a text file into the sandbox using a temporary local file.
        """
        if not self.sandbox_id:
            raise RuntimeError("Sandbox not created")
        
        try:
            # Create a temporary local file with the content
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                await self.client.upload_file(
                    sandbox_id=self.sandbox_id,
                    file_path=remote_path,
                    local_file_path=temp_file_path
                )
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
        except Exception:
            self.logger.exception("Failed to upload file")
            raise

    async def cleanup(self) -> None:
        if self.sandbox_id:
            try:
                await self.client.delete(self.sandbox_id)
                self.logger.info(f"Deleted sandbox: {self.sandbox_id}")
            except Exception:
                self.logger.exception("Failed to delete sandbox")
            finally:
                self.sandbox_id = None

FASTMCP_RUNNER_PATH = "/opt/mcp-runner/fastmcp_runner.py"
FASTMCP_FILE = "fastmcp_runner.py"

with open(os.path.join(os.path.dirname(__file__), "fastmcp_runner.py"), "r", encoding="utf-8") as _f:
    FASTMCP_RUNNER_SCRIPT = _f.read()

class MCPBootstrap:
    """Prepare the sandbox with FastMCP and our unified runner."""

    def __init__(self, sandbox: PrimeSandboxManager):
        self.sandbox = sandbox
        self.logger = logging.getLogger("verifiers.envs.MCPToolEnv.MCPBootstrap")

    async def provision(self) -> None:
        # Create virtual environment
        steps = [
            # Create venv with system site packages access (safer for containers)
            "python -m venv --system-site-packages /opt/mcp-venv",
            # Verify the venv works
            "/opt/mcp-venv/bin/python --version",
            # Install FastMCP in the venv (will use system packages as base)
            '/opt/mcp-venv/bin/python -m pip install --no-cache-dir "fastmcp>=2.2.0"',
            # Create directory for our runner script
            "mkdir -p /opt/mcp-runner",
        ]
        for step in steps:
            self.logger.info(f"[bootstrap] {step}")
            rc, out, err = await self.sandbox.run(step)
            if rc != 0:
                raise RuntimeError(f"Bootstrap step failed: {step}\n{err or out}")
            
        with open(os.path.join(os.path.dirname(__file__), FASTMCP_FILE), "r", encoding="utf-8") as _f:
            fastmcp_runner_script = _f.read()

        await self.sandbox.upload_file(FASTMCP_RUNNER_PATH, fastmcp_runner_script)

        rc, out, err = await self.sandbox.run(f"chmod +x {shlex.quote(FASTMCP_RUNNER_PATH)}")
        if rc != 0:
            raise RuntimeError(f"Failed to chmod runner: {err or out}")


REMOVED_KEYS = {
    "additionalProperties", "unevaluatedProperties", "deprecated",
    "readOnly", "writeOnly", "$defs", "definitions", "examples",
    "patternProperties", "$schema", "$id",
}

def _clean_schema_for_oai(schema: Any) -> Any:
    """
    Recursively sanitize a JSON Schema so strict validators accept it.
    Preserves: type, properties, required, enum, default, description, items, oneOf/anyOf/allOf.
    """
    if not isinstance(schema, dict):
        return schema

    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if k in REMOVED_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _clean_schema_for_oai(pv) for pk, pv in v.items()}
            continue
        if k in ("items",):
            out[k] = _clean_schema_for_oai(v)
            continue
        if k in ("oneOf", "anyOf", "allOf") and isinstance(v, list):
            out[k] = [_clean_schema_for_oai(x) for x in v]
            continue
        if isinstance(v, dict):
            out[k] = _clean_schema_for_oai(v)
        elif isinstance(v, list):
            out[k] = [_clean_schema_for_oai(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v

    if "properties" in out and "type" not in out:
        out["type"] = "object"

    return out


def _redact_env(env: Dict[str, str]) -> Dict[str, str]:
    redacted = {}
    for k, v in env.items():
        if any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            redacted[k] = "****"
        else:
            redacted[k] = v
    return redacted


async def _upload_json(sandbox: PrimeSandboxManager, obj: Dict[str, Any]) -> str:
    """
    Serialize JSON to a temp path in the sandbox using upload_file (no Base64).
    """
    content = json.dumps(obj, ensure_ascii=False)
    fname = f"/tmp/mcp_req_{uuid.uuid4().hex}.json"
    await sandbox.upload_file(fname, content)
    return fname

class MCPServerManager:
    """
    Register MCP servers and invoke them *inside the sandbox* via a small runner that uses FastMCP.Client.
    """

    def __init__(
        self,
        sandbox: PrimeSandboxManager,
        tool_discovery_timeout_s: int = 20,
        tool_call_timeout_s: int = 60,
        return_structured: bool = False,
    ):
        self.sandbox = sandbox
        self.logger = logging.getLogger("verifiers.envs.MCPToolEnv.MCPServerManager")
        self.tool_discovery_timeout_s = tool_discovery_timeout_s
        self.tool_call_timeout_s = tool_call_timeout_s
        self.return_structured = return_structured

        self._servers: Dict[str, MCPServerSpec] = {}
        self._tools_by_server: Dict[str, List[DiscoveredTool]] = {}
        # map "<server>:<tool>" -> (server, raw_tool_name)
        self._index: Dict[str, Tuple[str, str]] = {}

    def register(self, spec: MCPServerSpec) -> None:
        self._servers[spec.name] = spec

    def list_registered(self) -> List[str]:
        return list(self._servers.keys())

    def tools_index(self) -> Dict[str, Tuple[str, str]]:
        return dict(self._index)

    async def _runner(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uploads a JSON request file and executes the container runner, returns parsed JSON.
        """
        req_path = await _upload_json(self.sandbox, request)
        cmd = f"/opt/mcp-venv/bin/python {shlex.quote(FASTMCP_RUNNER_PATH)} --input-file {shlex.quote(req_path)}"
        self.logger.debug(f"[runner] {cmd}")
        rc, out, err = await self.sandbox.run(cmd)
        if rc != 0:
            raise RuntimeError(f"Runner failed (exit {rc}): {err or out}")

        try:
            return json.loads((out or "").strip() or "{}")
        except json.JSONDecodeError:
            prefix = (out or err or "")[:800]
            raise RuntimeError(f"Malformed runner output: {prefix}")

    @staticmethod
    def _server_payload(spec: MCPServerSpec) -> Dict[str, Any]:
        """
        Convert MCPServerSpec into the runner 'server' payload (fastmcp Client source).
        """
        server: Dict[str, Any] = {}
        if spec.script:
            server["type"] = "script"
            server["path"] = spec.script
        elif spec.url:
            server["type"] = "http"
            server["url"] = spec.url
            if spec.headers:
                server["headers"] = spec.headers
        elif spec.command:
            server["type"] = "stdio"
            server["command"] = spec.command[0]
            server["args"] = spec.command[1:]
        else:
            raise ValueError(f"ServerSpec {spec.name} is missing script/url/command")

        if spec.env:
            server["env"] = spec.env
        if spec.workdir:
            server["cwd"] = spec.workdir
        return server

    async def discover_tools(self, server_name: str) -> List[DiscoveredTool]:
        spec = self._servers[server_name]
        timeout = spec.discovery_timeout_s or self.tool_discovery_timeout_s
        req = {
            "mode": "list-tools",
            "server": self._server_payload(spec),
            "timeout": timeout,
        }
        self.logger.debug(f"[discover] {server_name} env={_redact_env(spec.env)} cwd={spec.workdir}")
        resp = await self._runner(req)

        if not resp.get("ok"):
            raise RuntimeError(f"Discovery error for {server_name}: {resp.get('error')}")

        tools: List[DiscoveredTool] = []
        for t in resp.get("tools", []):
            input_schema = _clean_schema_for_oai(t.get("inputSchema") or {})
            tools.append(
                DiscoveredTool(
                    name=t["name"],
                    description=t.get("description") or f"MCP tool: {t['name']}",
                    input_schema=input_schema,
                    meta=t.get("meta"),
                )
            )

        # Index tools with "<server>:<tool>"
        self._tools_by_server[server_name] = tools
        for t in tools:
            self._index[f"{server_name}:{t.name}"] = (server_name, t.name)
        self.logger.info(f"Discovered {len(tools)} tools from server '{server_name}'")
        return tools

    async def call_tool(self, namespaced_tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool and return normalized payload {text, structured_content, data, is_error}."""
        try:
            server_name, tool_name = self._index[namespaced_tool]
        except KeyError:
            return {"error": f"Unknown tool: {namespaced_tool}"}

        spec = self._servers[server_name]
        timeout = spec.call_timeout_s or self.tool_call_timeout_s

        # Reserved per-call timeout override
        if isinstance(arguments, dict) and "_timeout_s" in arguments:
            try:
                timeout = int(arguments.pop("_timeout_s")) or timeout
            except Exception:
                pass

        req = {
            "mode": "call-tool",
            "server": self._server_payload(spec),
            "tool_name": tool_name,
            "arguments": arguments or {},
            "timeout": timeout,
        }

        resp = await self._runner(req)
        if not resp.get("ok"):
            return {"error": resp.get("error")}

        return resp.get("result", {"text": "", "structured_content": None, "data": None, "is_error": False})


def _normalize_tool_func_name(namespaced: str) -> str:
    # Convert "<server>:<tool>" to a safe python identifier
    safe = namespaced.replace(":", "__").replace("/", "_")
    return f"mcp__{safe}"


def create_tool_wrapper(
    namespaced_tool: str,
    tool_meta: DiscoveredTool,
    server_mgr: MCPServerManager,
    return_structured: bool = False,
) -> Callable:
    """
    Create a sync wrapper that internally uses asyncio to call MCP tool via the container runner.
    By default returns plain text or a small JSON envelope; can return structured-only when allowed.
    """

    cleaned_schema = _clean_schema_for_oai(tool_meta.input_schema or {})
    props: Dict[str, Any] = cleaned_schema.get("properties", {}) or {}
    required = set(cleaned_schema.get("required", []) or [])

    validator = None
    if jsonschema and cleaned_schema.get("type") == "object":
        try:
            validator = jsonschema.Draft202012Validator(cleaned_schema)
        except Exception:
            validator = None

    async def _async_call(kwargs: Dict[str, Any]) -> str:
        if validator:
            errors = sorted(validator.iter_errors(kwargs), key=lambda e: e.path)
            if errors:
                msgs = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]
                return "Input validation failed:\n- " + "\n- ".join(msgs)

        result = await server_mgr.call_tool(namespaced_tool, kwargs or {})
        if "error" in result:
            return f"Tool error: {result['error']}"

        # Prefer FastMCP's structured data if present, else fall back to text
        data = result.get("data")  # hydrated Python-like object serialized by runner
        text = result.get("text") or ""

        if data is not None:
            if return_structured or server_mgr.return_structured:
                try:
                    return json.dumps(data, ensure_ascii=False)
                except Exception:
                    return json.dumps({"structured": data, "text": text}, ensure_ascii=False)
            return json.dumps({"structured": data, "text": text}, ensure_ascii=False)

        # If no data, at least include the runner's structured_content/text
        structured = result.get("structured_content")
        if structured is not None:
            return json.dumps({"structured": structured, "text": text}, ensure_ascii=False)

        return text

    params: List[inspect.Parameter] = []
    param_names: List[str] = []

    for prop_name, _ in props.items():
        param_names.append(prop_name)
        if prop_name in required:
            params.append(
                inspect.Parameter(
                    prop_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=AnyType,
                )
            )
        else:
            params.append(
                inspect.Parameter(
                    prop_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=AnyType,
                )
            )

    sig = inspect.Signature(params)

    def wrapper(*args, **kwargs) -> str:
        call_kwargs: Dict[str, Any] = {}
        for i, arg in enumerate(args):
            if i < len(param_names):
                call_kwargs[param_names[i]] = arg
        for key, value in kwargs.items():
            if key in param_names and value is not None:
                call_kwargs[key] = value
            elif key == "_timeout_s":
                call_kwargs[key] = value

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as ex:
                    fut = ex.submit(asyncio.run, _async_call(call_kwargs))
                    return fut.result()
            else:
                return loop.run_until_complete(_async_call(call_kwargs))
        except RuntimeError:
            return asyncio.run(_async_call(call_kwargs))

    wrapper.__name__ = _normalize_tool_func_name(namespaced_tool)
    wrapper.__signature__ = sig

    # Build a docstring from cleaned schema (include defaults & enums if present)
    doc = [tool_meta.description or f"MCP tool: {namespaced_tool}", "", "Args:"]
    for p, spec in props.items():
        typ = spec.get("type", "any")
        desc = spec.get("description", p)
        default = spec.get("default", None)
        enum = spec.get("enum", None)
        frag = f"  {p} ({typ}){' [required]' if p in required else ''}: {desc}"
        if default is not None:
            frag += f" (default: {default!r})"
        if enum:
            frag += f" (one of: {', '.join(map(str, enum))})"
        doc.append(frag)
    wrapper.__doc__ = "\n".join(doc) if props else (tool_meta.description or "MCP tool")

    return wrapper


class MCPToolEnv(vf.ToolEnv):
    """
    Spins up a Python-only Prime sandbox,
    installs FastMCP + a small runner,
    registers one or more MCP servers,
    discovers tools and exposes them to verifiers,
    and cleans up deterministically (auto_cleanup).
    """

    def __init__(
        self,
        mcp_servers: List[Dict[str, Any]],
        sandbox_config: Optional[Dict[str, Any]] = None,
        dataset: Optional[Dataset] = None,
        parser: Optional[vf.Parser] = None,
        rubric: Optional[vf.ToolRubric] = None,
        max_turns: int = 10,
        prime_api_key: Optional[str] = None,
        return_structured: bool = False,
        auto_cleanup: bool = True,
        **kwargs,
    ):
        cfg = SandboxConfig(**(sandbox_config or {}), api_key=prime_api_key or (sandbox_config or {}).get("api_key"))
        self.sandbox_mgr = PrimeSandboxManager(cfg)
        self.bootstrap = MCPBootstrap(self.sandbox_mgr)
        self.server_mgr = MCPServerManager(
            self.sandbox_mgr,
            return_structured=return_structured,
        )

        self.return_structured = return_structured
        self.auto_cleanup = auto_cleanup

        self._initialized = False
        self._cleanup_done = False
        self._active_rollouts = 0
        self._counter_lock = asyncio.Lock()
        self.logger = logging.getLogger("verifiers.envs.MCPToolEnv")

        super().__init__(tools=[], dataset=dataset, parser=parser, rubric=rubric, max_turns=max_turns, **kwargs)

        self._server_specs: List[MCPServerSpec] = [
            MCPServerSpec(
                name=s["name"],
                script=s.get("script"),
                url=s.get("url"),
                headers=s.get("headers"),
                command=s.get("command"),
                env=s.get("env", {}),
                workdir=s.get("workdir"),
                discovery_timeout_s=s.get("discovery_timeout_s"),
                call_timeout_s=s.get("call_timeout_s"),
            )
            for s in mcp_servers
        ]

    async def _upload_scripts_to_sandbox(self) -> None:
        """
        Upload script files to sandbox based on server specs.
        Handles multiple scenarios:
        1. Scripts in /workspace/* - looks for local files in environment directory
        2. Scripts with relative paths - looks for local files relative to environment directory  
        3. Scripts with absolute local paths - uploads from specified local path
        """
        for spec in self._server_specs:
            if not spec.script:
                continue
                
            script_path = spec.script
            local_source_path = None
            
            # Determine where to find the local script file
            if script_path.startswith("/workspace/"):
                # Convention: /workspace/foo.py -> look for foo.py in environment directory
                script_filename = os.path.basename(script_path)
                local_source_path = os.path.join(os.path.dirname(__file__), script_filename)
                # Ensure workspace directory exists
                await self.sandbox_mgr.run("mkdir -p /workspace")
                
            elif not script_path.startswith("/"):
                # Relative path - look in environment directory
                local_source_path = os.path.join(os.path.dirname(__file__), script_path)
                # Create parent directories in sandbox
                parent_dir = os.path.dirname(script_path)
                if parent_dir:
                    await self.sandbox_mgr.run(f"mkdir -p {shlex.quote(parent_dir)}")
                    
            elif os.path.exists(script_path):
                # Absolute local path that exists - upload directly
                local_source_path = script_path
                # Create parent directories in sandbox
                parent_dir = os.path.dirname(script_path)
                if parent_dir != "/":
                    await self.sandbox_mgr.run(f"mkdir -p {shlex.quote(parent_dir)}")
            
            # Upload the script if we found a local source
            if local_source_path and os.path.exists(local_source_path):
                try:
                    with open(local_source_path, 'r', encoding='utf-8') as f:
                        script_content = f.read()
                    
                    await self.sandbox_mgr.upload_file(script_path, script_content)
                    await self.sandbox_mgr.run(f"chmod +x {shlex.quote(script_path)}")
                    self.logger.info(f"Uploaded script: {local_source_path} -> {script_path}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to upload script {local_source_path} -> {script_path}: {e}")
                    
            elif spec.script:
                self.logger.warning(f"Script not found locally: {spec.script} (looked at: {local_source_path})")

    async def _initialize(self) -> None:
        if self._initialized:
            return

        # 1) Bring up sandbox
        await self.sandbox_mgr.create()

        # 2) Bootstrap FastMCP + runner
        await self.bootstrap.provision()

        # 3) Upload script files to sandbox when needed
        await self._upload_scripts_to_sandbox()

        # 4) Register servers & discover tools
        all_tools: List[Callable] = []
        for spec in self._server_specs:
            self.server_mgr.register(spec)
            tools = await self.server_mgr.discover_tools(spec.name)
            for t in tools:
                namespaced = f"{spec.name}:{t.name}"
                tool_fn = create_tool_wrapper(namespaced, t, self.server_mgr, return_structured=self.return_structured)
                all_tools.append(tool_fn)

        # 5) Surface tools to verifiers ToolEnv
        self.tools = all_tools
        from verifiers.utils.tool_utils import convert_func_to_oai_tool
        self.oai_tools = [convert_func_to_oai_tool(fn) for fn in self.tools]
        self.tool_map = {fn.__name__: fn for fn in self.tools}

        self._initialized = True
        self.logger.info(f"Initialized MCPToolEnv with {len(all_tools)} tools from {len(self._server_specs)} server(s)")

    async def rollout(self, *args, **kwargs) -> Tuple[vf.Messages, vf.State]:
        await self._initialize()
        async with self._counter_lock:
            self._active_rollouts += 1
        try:
            return await super().rollout(*args, **kwargs)
        finally:
            do_cleanup = False
            async with self._counter_lock:
                self._active_rollouts -= 1
                if self.auto_cleanup and not self._cleanup_done and self._active_rollouts <= 0:
                    do_cleanup = True
                    self._cleanup_done = True
            if do_cleanup:
                try:
                    await self.cleanup()
                except Exception:
                    self.logger.exception("Auto-cleanup failed")

    async def __aenter__(self) -> "MCPToolEnv":
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cleanup()

    def __del__(self):
        try:
            if getattr(self, "sandbox_mgr", None) and getattr(self.sandbox_mgr, "sandbox_id", None):
                async def _finalize():
                    try:
                        await self.cleanup()
                    except Exception:
                        pass
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import threading
                        threading.Thread(target=lambda: asyncio.run(_finalize()), daemon=True).start()
                    else:
                        asyncio.run(_finalize())
                except Exception:
                    pass
        except Exception:
            pass

    async def cleanup(self) -> None:
        """Cleanup the sandbox and resources (idempotent)."""
        try:
            await self.sandbox_mgr.cleanup()
            self._cleanup_done = True
            self.logger.info("MCPToolEnv cleanup completed")
        except Exception:
            self.logger.exception("Cleanup failed")


def load_environment(
    mcp_launch_commands: List[Dict[str, Any]],
    dataset_name: str = "gsm8k",
    split: str = "train",
    num_examples: int = 100,
    sandbox_config: Optional[Dict[str, Any]] = None,
    max_turns: int = 10,
    prime_api_key: Optional[str] = None,
    return_structured: bool = False,
    auto_cleanup: bool = True,
    **kwargs
) -> MCPToolEnv:
    """
    Load MCPToolEnv with specified MCP server configs (Python-only or HTTP).

    Args:
        mcp_launch_commands: List of MCP server configs (see MCPServerSpec). Examples:
            [{"name": "calc", "script": "/workspace/calc_server.py"}]
            [{"name": "weather", "url": "https://weather.example.com/mcp"}]
            [{"name": "py_stdio", "command": ["python", "/workspace/tools_server.py"]}]
    """
    dataset = load_example_dataset(dataset_name, split=split)
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    parser = vf.Parser()
    rubric = vf.ToolRubric(tools=[])

    env = MCPToolEnv(
        mcp_servers=mcp_launch_commands,
        sandbox_config=sandbox_config,
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        prime_api_key=prime_api_key,
        return_structured=return_structured,
        auto_cleanup=auto_cleanup,
        **kwargs,
    )
    return env