# ARC-AGI Multi-Agent Solver

## What It Is

A multi-strategy, multi-model ARC-AGI solver built as a single environment on top of the [verifiers](https://github.com/PrimeIntellect-ai/verifiers) multi-agent RL framework.

Originally designed by [Johan Land](https://x.com/arcprize/status/2018746794310766668) ([source repo](https://github.com/beetree/ARC-AGI)), scoring 94.5% at $11.4/task on ARC-AGI v1 and 72.9% at $38.9/task on v2. Re-implemented here on top of the verifiers framework to enable multi-agent RL training.

## How It Works

An orchestrator model sees an ARC task and attempts a quick solve. Its answer, along with answers from several parallel specialist strategies, are collected into a candidate pool. A council of judges then picks the two best candidates as final submissions.

## Architecture

The system is a pipeline of environments coordinated by a central `ArcPipelineEnv`. Each strategy is its own `MultiAgentEnv` subclass with a dedicated actor. The verifiers `Protocol` handles routing — when the pipeline spawns work, it sends `{"task": env_name}` and Protocol routes to the matching environment's `rollout()`.

### Solver Strategies

- **Shallow / Deep** — Standard LLM reasoning (shallow = fast, deep = extended thinking)
- **Codegen (v1b, v4)** — Model writes a `solver()` function, verified against all training pairs in a sandboxed subprocess, then run on the test input
- **Image** — Sends a rendered PNG of the training grids to a vision model (Qwen VL, etc.)
- **Objects Pipeline** — Extracts objects from grids, identifies transformations, applies them
- **Hint Extractor** — Generates hints about the task pattern, feeds them to a hint-aware solver

### Judging / Selection

- **Duo Pick Council** — 3 judges in parallel see all candidates + reasoning, each picks their top 2 grids. Weighted voting (1st pick = 2pts, 2nd = 1pt), top 2 by aggregate score become the final answers.
- **Logic + Consistency Judges** — Fallback path. Logic judge checks if reasoning produces the claimed grid. Consistency judge checks internal coherence. Scores aggregated with consensus vote count.
- **Consensus** — Pure vote counting across all solvers. Used as the default fallback if no judges are enabled.

## Execution Flow

```
Step 1 — Shallow Search
  Orchestrator solves (Turn 1), parse candidates
  Spawn: extra shallow + codegen (v1b & v4) + image + objects + hint
  All run in parallel via Protocol.spawn()

Step 2 — Consensus Check
  If strong agreement among candidates → skip to Finalize

Steps 3-4 — Extended Search (optional, off by default)
  More codegen + image, another consensus check

Step 5 — Deep Search
  Deep thinking models + more codegen + more image
  All in parallel

Finalize
  Duo pick council (default) OR logic/consistency judges OR consensus
  Return top 2 picks as attempt_1 and attempt_2
```

## Configuration

A single `MODEL_CONFIG` dict controls everything — which roles are active, which models they use, and how many parallel instances to spawn:

```python
MODEL_CONFIG = {
    "shallow":  [None],                  # None = use -m model
    "deep":     [None],
    "codegen":  [None],
    "image":    ["qwen3-vl-235b-i"],     # vision model
    "duo_pick": [None],
    # "judge":  [None],                  # disabled by default
}
```

- Key present = role enabled. Missing/commented = disabled.
- Tuple entries like `("sonnet", 3)` spawn 3 parallel instances of the same model.
- Multiple entries per role give model diversity (e.g., `"deep": ["sonnet", "gemini-3-pro"]`).

## Scoring (Eval)

Team reward — 1.0 if top pick correct, 0.5 if pass@2, 0.0 otherwise. Currently eval-only. Use `--debug` to see full per-actor details. The rubric is structured to be compatible with GRPO training in the future.

## Implementation

Everything lives in `arc_multiagent.py`:

| Section | What |
|---------|------|
| Grid utilities, prompt templates | Shared helpers for formatting ARC grids and building prompts |
| Codegen prompts (v1b, v4) | Two prompt styles for code generation |
| Sandbox execution | Subprocess-isolated Python execution for generated solver code |
| Image generation | Matplotlib rendering of ARC grids as PNG for vision models |
| Duo pick prompt | Prompt builder for the judge council |
| Model config + helpers | `ConfigEntry`, `_parse_entry()`, `_actor_id_for()` |
| 8 environment classes | SingleSolver, DeepThinking, HintSolver, HintExtractor, ObjectsPipeline, Judge, Codegen, ImageSolver, DuoPickJudge |
| ArcPipelineEnv | Central orchestrator — `on_turn_complete` runs the 5-step pipeline |
| Rubric, actor/env factories, `load_environment()` | Wiring it all together for `vf-eval` |


## Future Plans

- **Break up the monolith** — The single file works but will split into modules: `envs/`, `prompts/`, `config/`, `utils/`. etc. 
- **Taskset + Harness + Agent abstractions** — Aligning with the broader verifiers roadmap:
  - **Taskset** generalizes datasets — tasks define *what* should be done, the environment, and how it's evaluated (rubrics, tools, skills), but not *how* the agent works.
  - **Harness** encapsulates rollout logic (tools, skills, turn structure) without a model attached.
  - **Model + Harness = Agent.** Agent + Task = Rollout.
  - **Protocol** specifies how multiple *agents* (not just models) interact with a task. This separates the agent definition from the orchestration layer.
  - For ARC, this would mean: each solver strategy becomes a Harness, the ARC task data becomes a Taskset with task-specific tools (sandbox, image gen) and rubrics, and the pipeline Protocol wires agents together — cleanly decoupling what we currently have bundled in one file.
