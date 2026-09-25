import json
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.harnesses.utils.launch import bundle_program
from verifiers.v1.tasksets.tau import data as data_helpers


class TauHarness(vf.Harness[vf.HarnessConfig]):
    """Expose per-role interception credentials to the official Tau runner."""

    NEEDS_CONTAINER = False
    connection: dict
    peers: dict[str, dict]

    async def session(
        self,
        ctx,
        trace,
        runtime,
        endpoint,
        secret,
        mcp_urls,
        data,
        tool_interception_url=None,
    ):
        self.connection = {
            "model": f"openai/{ctx.model.removeprefix('openai/')}",
            "args": {
                "api_base": endpoint,
                "api_key": secret,
                "temperature": 0,
                "num_retries": 0,
                "timeout": 86400,
            },
        }
        return await super().session(
            ctx, trace, runtime, endpoint, secret, mcp_urls, data, tool_interception_url
        )

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        dependency = "tau2" if data.runner == "synth" else "tau2[knowledge]"
        dependency += (
            f" @ https://github.com/{data.repository}/archive/{data.revision}.tar.gz"
        )
        metadata = '# /// script\n# requires-python = ">=3.12,<3.14"\n'
        dependencies = [dependency, "audioop-lts; python_version >= '3.13'"]
        if data.runner == "upstream":
            dependencies.append("websockets>=13")
        metadata += f"# dependencies = {json.dumps(dependencies)}\n# ///\n"
        source = bundle_program(
            metadata + Path(__file__).with_name("program.py").read_text(), data_helpers
        )
        output = f"tau-result-{trace.id}.json"
        request = {
            "data": data.model_dump(mode="json"),
            "connections": {"assistant": self.connection, **self.peers},
            "output": output,
        }
        program = await runtime.prepare_uv_script(source, self.config.resolved_env)
        result = await runtime.run_program(
            [*program, json.dumps(request)], self.config.resolved_env
        )
        if result.exit_code == 0:
            simulation = json.loads(await runtime.read(output))
            if simulation["reward_info"] is None:
                raise ValueError("Tau returned a simulation without an official reward")
            trace.info["tau"] = simulation
            trace.stop(f"tau_{simulation['termination_reason']}")
        return result
