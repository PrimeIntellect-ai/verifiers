# Architecture

verifiers is built out of the following parts: 

A server-backed evaluation or prime-rl **orchestrator** creates worker processes and distributes rollout requests among them. Each worker loads its taskset and harness, owns an **interception pool**, and creates the runtime used by each rollout it handles.

The orchestrator and workers are managed by verifiers and prime-rl themselves and thus offer few configurable knobs.

The **rollout** is the executable combination of one loaded task, the harness, and any tools. Each rollout has an independent trace and runtime state. verifiers has three different runtimes which you can use for most tasksets:
- The `subprocess` runtime runs the rollouts in Python subprocesses locally. Thus, it is meant for debugging purposes, as there might be side effects during runtime, such as one subprocess altering the config files of the harness, which then affects the other subprocesses.
- The `docker` runtime runs the rollouts in docker containers on your local machine.
- Sandbox runtimes, such as `prime` or `modal`, are meant for production, especially for training or higher concurrency evaluation. These runtimes run remotely.

The harness runs inside the rollout runtime to interact with the taskset. The harness does _not_ call the provider endpoint directly. Instead, model traffic goes through an **interception server** over a local connection or [Prime Tunnel](https://docs.primeintellect.ai/sandboxes/tunnel).

The interception server receives all these requests and then sends them over to the actual API, e.g. the OpenAI responses endpoint. It uses the endpoint that the harness expects, so Codex will use OpenAI Responses, while Claude Code will use the Anthropic Messages API.

The interception server, however, allows several things beyond just replaying the correct API response: 
- Traces are built live, thus allowing the collection of the trajectories as they happen
- Setting sampling parameters in harnesses that don't necessarily expose those settings
- Intercepting and rewriting tool responses or server-side web search results to block reward hacks
