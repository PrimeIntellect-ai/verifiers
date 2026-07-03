# Architecture

Verifiers is built out of the following parts: 

The overall evaluation or prime-rl **orchestrator**, which creates several worker processes. These **workers** load the several other rollout-related objects, such as the harness or taskset object, as well as an **interception server**. The worker also creates the **rollout runtime**, where the actual rollout happens.

The orchestrator and workers are managed by Verifiers and prime-rl themselves and thus offer little knobs to be configurable.

The **rollout** is the actual environment, i.e., the combination of the taskset, the harness and, optionally, the registered tools. Each of the rollouts is independent from the others. Verifiers has three different runtimes which you can use for most environments:
- The `subprocess` runtime runs the rollouts in Python subprocesses locally. Thus, it is meant for debugging purposes, as there might be side effects during runtime, such as one subprocess altering the config files of the harness, which then affects the other subprocesses.
- The `docker` runtime runs the rollouts in docker containers on your local machine.
- Sandbox runtimes, such as `prime` or `modal`, are meant for production, especially for training or higher concurrency evaluation. These runtimes run remotely.

The harness is placed inside the runtime to interact with the taskset. However, the harness itself does _not_ call the model API (such as the OpenAI API or the vLLM endpoint) directly. Instead, they are connected to an **interception server** via a tunnel (either locally or by using [Prime Tunnels](https://docs.primeintellect.ai/sandboxes/tunnel)).

The interception server receives all these requests and then sends them over to the actual API, e.g. the OpenAI responses endpoint. It uses the endpoint that the harness expects, so Codex will use OpenAI Responses, while Claude Code will use the Anthropic Messages API.

The interception server, however, allows several things beyond just replaying the correct API response: 
- Traces are built live, thus allowing the collection of the trajectories as they happen
- Setting sampling parameters in harnesses that don't necessarily expose those settings
- Intercepting and rewriting tool responses or server-side web search results to block reward hacks
